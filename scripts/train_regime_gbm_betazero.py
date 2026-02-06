# train_regime_gbm.py
import os, json, random
import copy
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim


from src.ppo.update import ppo_update_joint
from src.ppo.agent import JointBandPolicy, ValueNetCLS, PPOConfig

from src.utils.training_utils import freeze_centers_in_stage2
from src.utils.rlopt_helpers import build_corr_from_pairs, build_cov

from src.regime_gbm.gbm_env import globalsetting
from src.ppo.rollout import rollout_joint
from src.regime_gbm.regime_gbm_env import RegimeGBMBandEnvMulti


def set_seed(seed: int, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

def load_regime_json(path: str, *, N: int):
    """
    Load regimes/P from JSON like:
      {"regimes":[{"beta":[...],"sigmas":[...],"R":[[...]]}, ...],
       "P":[[...],[...]],
       "dt": 0.003968... (optional)}
    Returns: (regimes_list, P_array, dt_or_None)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"regime json not found: {p}")
    obj = json.loads(p.read_text())
    regimes_in = obj["regimes"]
    P = np.asarray(obj["P"], float)
    dt = obj.get("dt", None)

    regimes = []
    for r in regimes_in:
        beta = np.asarray(r["beta"], float).reshape(-1)
        sigmas = np.asarray(r["sigmas"], float).reshape(-1)
        R = np.asarray(r["R"], float)
        if beta.size != N or sigmas.size != N:
            raise ValueError(f"regime dim mismatch: beta={beta.size}, sigmas={sigmas.size}, expected N={N}")
        if R.shape != (N, N):
            raise ValueError(f"R shape mismatch: {R.shape}, expected {(N,N)}")
        regimes.append(dict(beta=beta, sigmas=sigmas, R=R))
    if P.shape[0] != P.shape[1] or P.shape[0] != len(regimes):
        raise ValueError(f"P shape mismatch: {P.shape}, expected KxK with K={len(regimes)}")
    return regimes, P, dt

def _snapshot_trainable_params(model: torch.nn.Module):
    # clone trainable params only
    return {name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad}

def _param_update_norm(prev, model: torch.nn.Module) -> float:
    # L2 norm of (current - prev) over trainable params
    sq = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad:
            d = (p.detach() - prev[name]).float()
            sq += float((d*d).sum().cpu())
    return float(np.sqrt(sq))

@torch.no_grad()
def sample_s_stats(policy, obs_batch: torch.Tensor) -> dict:
    """
    obs_batch: torch.Tensor [B, obs_dim] on device
    Returns dict of s stats (using sample_s_only).
    """
    if hasattr(policy, "sample_s_only"):
        s_t, _, _ = policy.sample_s_only(obs_batch)
        s = s_t.detach().cpu().numpy()
    else:
        # fallback if you ever use older API
        _, s_t, _, _, _ = policy.sample_stage2(obs_batch)
        s = s_t.detach().cpu().numpy()
    return {
        "s_min": float(np.min(s)),
        "s_mean": float(np.mean(s)),
        "s_max": float(np.max(s)),
        "s_std": float(np.std(s)),
    }


def main():
    # -------------------------
    # Config (same as notebook)
    # -------------------------
    globalcfg = globalsetting(
        seed    = 42,
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        N_ASSETS = 5,
        years = 1,
        sigmas   = np.array([0.40, 0.30, 0.12, 0.22, 0.25], dtype=float),
        pair_rhos = {
            (0,1): 0.60,
            (1,3): -0.20,
            (2,4): 0.05,
        },
        DISCOUNT_BY_BANK = True,
        INIT_W0_UNIFORM  = True,
        BAND_SMOOTH_COEF = 0.0,
        TRADE_PEN_COEF   = 0.0,
        TARGET_ETA = 0.0,
        ALPHA = 1/3,
        STAGE1_WIDTH_COEF = 0.05,

        OBS_BETA_ZERO = True,
    )
    # --- episode-level regime randomization ---
    # Each episode samples {beta_k, sigmas_k, R_k} for all regimes and keeps them fixed within the episode.
    globalcfg.REGIME_EPISODE_RANDOMIZE = True
    globalcfg.REGIME_BETA_STD = 0        # std for beta perturbation
    globalcfg.REGIME_SIGMA_LOGSTD = 0.5    # log-std for sigma multiplicative noise
    globalcfg.REGIME_CORR_NOISE = 0.08      # additive noise on correlation matrix entries
    globalcfg.REGIME_BETA_CLIP = 0.999
    globalcfg.REGIME_SIGMA_CLIP = (1e-4, 10.0)
    set_seed(globalcfg.seed, globalcfg.device)

    # baseline R/Cov used for MV-center computations in rollout code
    R_base = build_corr_from_pairs(globalcfg.N_ASSETS, base_rho=0.20, pair_rhos=globalcfg.pair_rhos, make_psd=True)
    _ = build_cov(sigmas=globalcfg.sigmas, R=R_base, make_psd=True)

    target_choices = [0.02, 0.04, 0.06, 0.08]
    REGIME_JSON = os.environ.get("REGIME_JSON", "configs/regime_k2_hist.json")

    cfg = PPOConfig(
        horizon=globalcfg.T_days,
        gamma=1.0,
        gae_lambda=1.0,
        batch_episodes=64,
        epochs=4,
        minibatch_size=8192,
        lr_actor=1e-4,
        lr_critic=5e-4,
        lr_actor2=1e-4,
        lr_critic2=5e-4,
        clip_ratio=0.15,
        entropy_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    # -------------------------
    # Regimes (example)
    # -------------------------
    N = globalcfg.N_ASSETS

    def corr_from_base(base_rho, extra_pairs=None):
        extra_pairs = extra_pairs or {}
        return build_corr_from_pairs(N, base_rho=base_rho, pair_rhos=extra_pairs, make_psd=True)

    # Default (fallback) regimes if JSON is missing
    regimes_default = [
        dict(beta=np.ones(N)*0.6,
             sigmas=np.array([0.18, 0.15, 0.10, 0.12, 0.14]),
             R=corr_from_base(0.25, {(0,1):0.55})),
        dict(beta=np.ones(N)*(-0.2),
             sigmas=np.array([0.45, 0.35, 0.18, 0.28, 0.32]),
             R=corr_from_base(0.60, {(1,3):-0.15})),
    ]
    P_default = np.array([[0.97, 0.03],
                          [0.10, 0.90]], float)

    try:
        regimes, P, dt_json = load_regime_json(REGIME_JSON, N=N)
        print(f"[Regime] loaded from {REGIME_JSON} (K={len(regimes)})")
        if dt_json is not None and hasattr(globalcfg, "dt"):
            globalcfg.dt = float(dt_json)
            print(f"[Regime] set globalcfg.dt={globalcfg.dt} from JSON")
    except Exception as e:
        regimes, P = regimes_default, P_default
        print(f"[Regime] fallback to default regimes because: {e}")

    # ============================================================
    # Regime info into observation
    #   - Env appends soft regime prob gamma (len K) to obs global features.
    #   - Networks must be created with global_dim = 4 + K, and PPOConfig must match.
    # ============================================================
    K = int(len(regimes))
    globalcfg.REGIME_GAMMA_ON_OBS = True
    cfg.global_dim = 4 + K

    # env factory to inject into rollout
    #def env_ctor():
    #    return RegimeGBMBandEnvMulti(cfg=globalcfg, regimes=regimes, P=P, init_regime=None, R=R_base)
    
    def env_ctor(gcfg_ep=None, R_ep=None, seed=None):
        g_use = globalcfg if gcfg_ep is None else gcfg_ep
        if seed is not None:
            g_use = copy.copy(g_use)
            g_use.seed = int(seed)
        R_use = R_base if R_ep is None else R_ep
        return RegimeGBMBandEnvMulti(cfg=g_use, regimes=regimes, P=P, init_regime=None, R=R_use)

    # -------------------------
    # Networks / optim
    # -------------------------
    policy = JointBandPolicy(
        N, d_model=128, nlayers=2, nhead=4,
        use_cash_softmax=True,
        global_dim=cfg.global_dim,
    ).to(globalcfg.device)
    value  = ValueNetCLS(
        N, d_model=128, nlayers=2, nhead=4,
        global_dim=cfg.global_dim,
    ).to(globalcfg.device)

    opt_pi = optim.Adam(policy.parameters(), lr=cfg.lr_actor)
    opt_v  = optim.Adam(value.parameters(),  lr=cfg.lr_critic)

    # -------------------------
    # Stage 2 only (same as your notebook)
    # -------------------------
    print("[Stage 2] start (regime-switching GBM)")
    freeze_centers_in_stage2(policy)
    policy.use_cash_softmax = True
    opt_pi = optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=cfg.lr_actor)

    stages = [
        ([0.975],  8, 32, 0.015),
        ([0.99,0.995], 10, 48, 0.012),
        ([0.995,0.999,1.0], 12, 48, 0.010),
    ]

    for lam_list, updates, be, ent in stages:
        cfg.entropy_coef = ent
        cfg.batch_episodes = be
        print(f"[Stage 2] λ choices={lam_list}, updates={updates}, batch_eps={be}, ent={ent}")
        for it in range(updates):
            batch = rollout_joint(
                policy, value, cfg,
                gcfg=globalcfg,
                lam_choices=lam_list,
                target_choices=None, #target_choices,
                stage=2,
                R=R_base,
                env_ctor=env_ctor,     # ★ ここだけ追加
            )
            # ---- s statistics on the rollout batch obs (cheap) ----
            obs = batch["obs"]  # [T_total, obs_dim] on device
            B = min(obs.shape[0], 2048)
            idx = torch.randint(0, obs.shape[0], (B,), device=obs.device)
            s_stats = sample_s_stats(policy, obs[idx])

            # ---- update-norm tracking ----
            prev_params_epoch = _snapshot_trainable_params(policy)

            ppo_update_joint(
                policy, value, opt_pi, opt_v, cfg, batch,
                env_cfg=globalcfg, stage=2, width_prior_w=0.02
            )

            upd_norm = _param_update_norm(prev_params_epoch, policy)

            if (it+1) % 2 == 0:
                print(
                    f"  upd {it+1:02d}: "
                    f"mean_annual_ret={batch['rew_ep_mean']/globalcfg.years:.4f} "
                    f"| upd_norm={upd_norm:.3e} "
                    f"| s(min/mean/max/std)=({s_stats['s_min']:.3f}/{s_stats['s_mean']:.3f}/{s_stats['s_max']:.3f}/{s_stats['s_std']:.3f})"
                )
                print(
                    f"... "
                    f"| b(min/mean/max/zero%)=({batch.get('b_min_mean', float('nan')):.3e}/"
                    f"{batch.get('b_mean_mean', float('nan')):.3e}/"
                    f"{batch.get('b_max_mean', float('nan')):.3e}/"
                    f"{100*batch.get('b_zero_frac_mean', float('nan')):.1f}%) "
                    f"| trade_rate={batch.get('trade_rate_mean', float('nan')):.3f} "
                    f"| avg_turnover={batch.get('avg_turnover_mean', float('nan')):.3e}"
                    )
                print(f"... | minm_zero={batch.get('minm_zero_mean', float('nan')):.3f}"
                    f" delta_zero={batch.get('delta_zero_mean', float('nan')):.3f}"
                    f" s_zero={batch.get('s_zero_mean', float('nan')):.3f}"
                    f" b_zero={batch.get('b_zero_mean', float('nan')):.3f}"
                    )



    # -------------------------
    # Save
    # -------------------------
    outdir = "checkpoints_regime_A2"
    os.makedirs(outdir, exist_ok=True)

    torch.save(policy.state_dict(), os.path.join(outdir, "policy_stage2_A2_regime_beta_zero.pt"))
    torch.save(value.state_dict(),  os.path.join(outdir, "value_stage2_A2_regime_beta_zero.pt"))
    meta = dict(
        N_ASSETS=N,
        target_choices=target_choices,
        regimes=[dict(beta=r["beta"].tolist(), sigmas=r["sigmas"].tolist()) for r in regimes],
        P=P.tolist(),
        stages=stages,
    )
    with open(os.path.join(outdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Checkpoint] saved to {outdir}/")


if __name__ == "__main__":
    main()
