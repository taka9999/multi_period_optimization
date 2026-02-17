# train_regime_gbm.py
import os, json, random
import copy
from pathlib import Path
from functools import partial

import numpy as np
import torch
import torch.optim as optim

from src.ppo.update import ppo_update_joint
from src.ppo.agent import JointBandPolicy, ValueNetCLS, PPOConfig
from src.utils.training_utils import freeze_width_head_in_stage1,freeze_centers_in_stage2, make_market_sampler, MarketSamplerConfig

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


def _proj_to_corr_psd(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetrize -> eigen clip -> renormalize to correlation (diag=1)."""
    M = np.asarray(M, float)
    M = 0.5 * (M + M.T)

    # eigen clip to PSD
    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    Mp = (V * w) @ V.T

    # renormalize to correlation
    d = np.sqrt(np.clip(np.diag(Mp), eps, None))
    Mp = Mp / (d[:, None] * d[None, :])
    Mp = np.clip(Mp, -0.999, 0.999)
    np.fill_diagonal(Mp, 1.0)

    # PSD guard again (tiny)
    Mp = 0.5 * (Mp + Mp.T)
    w2, V2 = np.linalg.eigh(Mp)
    if np.min(w2) < eps:
        w2 = np.clip(w2, eps, None)
        Mp = (V2 * w2) @ V2.T
        d = np.sqrt(np.clip(np.diag(Mp), eps, None))
        Mp = Mp / (d[:, None] * d[None, :])
        np.fill_diagonal(Mp, 1.0)
    return Mp


def _mix_corr(R_base: np.ndarray, rng: np.random.Generator,
              noise_std: float,
              mean_corr_shift: float = 0.0,
              hi_cond: bool = False) -> np.ndarray:
    """
    - Add symmetric noise, keep diag=1.
    - Optionally shift average correlation up/down (mean_corr_shift).
    - Optionally create high condition-number structure (hi_cond=True).
    """
    N = R_base.shape[0]
    R = np.array(R_base, float)

    # 1) base mean-corr shift via shrinkage to/from rank-one
    #    R' = (1-a)R + a*J  (J has ones everywhere -> high avg corr)
    if mean_corr_shift != 0.0:
        a = float(np.clip(mean_corr_shift, -0.95, 0.95))
        J = np.ones((N, N), float)
        np.fill_diagonal(J, 1.0)
        if a > 0:
            R = (1 - a) * R + a * J
        else:
            # negative shift: shrink toward identity (lower avg corr)
            a2 = -a
            I = np.eye(N)
            R = (1 - a2) * R + a2 * I

    # 2) noise
    E = rng.normal(0.0, noise_std, size=(N, N))
    E = 0.5 * (E + E.T)
    np.fill_diagonal(E, 0.0)
    R = R + E
    np.fill_diagonal(R, 1.0)

    # 3) optional high condition number: add a strong 1-factor component
    if hi_cond:
        v = rng.normal(size=N)
        v /= (np.linalg.norm(v) + 1e-12)
        # add factor vv^T (in corr space), then project back
        strength = 0.8  # tune: bigger -> higher cond
        R = (1 - strength) * R + strength * (np.outer(v, v))
        np.fill_diagonal(R, 1.0)

    return _proj_to_corr_psd(R)

def market_sampler(rng: np.random.Generator, k: int, base_sigmas=None, base_R=None, globalcfg=None):
    """
    Returns:
      R_ep:  (N,N) correlation matrix (PSD, diag=1)
      sigmas_ep: (N,) positive vols
    """
    N = base_sigmas.size

    # --- sigma: lognormal multiplicative noise ---
    logstd = float(getattr(globalcfg, "REGIME_SIGMA_LOGSTD", 0.4))  # reuse your knob
    z = rng.normal(0.0, logstd, size=N)
    sig = base_sigmas * np.exp(z)

    lo, hi = getattr(globalcfg, "REGIME_SIGMA_CLIP", (1e-4, 10.0))
    sig = np.clip(sig, lo, hi)

    # --- correlation: mixture of scenarios ---
    # probabilities (tune)
    p_hi_corr = 0.25
    p_lo_corr = 0.25
    p_hi_cond = 0.15
    # rest: "normal"

    u = rng.random()
    noise_std = float(getattr(globalcfg, "REGIME_CORR_NOISE", 0.08))

    if u < p_hi_corr:
        # higher average correlation
        R = _mix_corr(base_R, rng, noise_std=noise_std, mean_corr_shift=+0.5, hi_cond=False)
    elif u < p_hi_corr + p_lo_corr:
        # lower average correlation (more idiosyncratic)
        R = _mix_corr(base_R, rng, noise_std=noise_std, mean_corr_shift=-0.5, hi_cond=False)
    elif u < p_hi_corr + p_lo_corr + p_hi_cond:
        # high condition number / factor dominance
        R = _mix_corr(base_R, rng, noise_std=noise_std, mean_corr_shift=+0.2, hi_cond=True)
    else:
        # normal: small perturbation around base
        R = _mix_corr(base_R, rng, noise_std=noise_std, mean_corr_shift=0.0, hi_cond=False)
    return R, sig

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
        INIT_W0_UNIFORM = True,
        BAND_SMOOTH_COEF = 0.0,
        TRADE_PEN_COEF = 0.0,
        ALPHA = 1/3,
        STAGE1_WIDTH_COEF = 0,

        # LQ / MV-style reward parameters
        ALLOW_CASH_IN_MV = False,
        MV_USE_TARGET = False,   # whether to use target return constraint in MV center
        RISK_GAMMA = 0.0,
        TARGET_ETA = 0.0,        # eta in hinge penalty eta*[target - mu^T w]_+
        REGIME_GAMMA_ON_OBS = False,
        OBS_BETA_ZERO = True,

        ROLL_COV_SUMMARY_ON_OBS = True,
        ROLL_OBS_LOOKBACK = 21,
        ROLL_TOP_EIGS = 5,
        )
    
    # --- episode-level regime randomization ---
    # Each episode samples {beta_k, sigmas_k, R_k} for all regimes and keeps them fixed within the episode.
    globalcfg.REGIME_EPISODE_RANDOMIZE = False
    globalcfg.REGIME_BETA_STD = 0.25        # std for beta perturbation
    globalcfg.REGIME_SIGMA_LOGSTD = 0.4    # log-std for sigma multiplicative noise
    globalcfg.REGIME_CORR_NOISE = 0.08      # additive noise on correlation matrix entries
    globalcfg.REGIME_BETA_CLIP = 0.999
    globalcfg.REGIME_SIGMA_CLIP = (1e-4, 10.0)
    set_seed(globalcfg.seed, globalcfg.device)

    # baseline R/Cov used for MV-center computations in rollout code
    R_base = build_corr_from_pairs(globalcfg.N_ASSETS, base_rho=0.20, pair_rhos=globalcfg.pair_rhos, make_psd=True)
    _ = build_cov(sigmas=globalcfg.sigmas, R=R_base, make_psd=True)

    target_choices = [0.04, 0.06, 0.08, 0.1]
    REGIME_JSON = os.environ.get("REGIME_JSON", "configs/regime_k2_hist.json")

    cfg = PPOConfig(
        horizon=globalcfg.T_days,
        gamma=1.0,
        gae_lambda=1.0,
        batch_episodes=32,
        epochs=4,
        minibatch_size=4096,
        lr_actor=1e-3,
        lr_critic=5e-3,
        lr_actor2=1e-3,
        lr_critic2=5e-3,
        clip_ratio=0.7,
        entropy_coef=0.5,
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
    #if globalcfg.REGIME_GAMMA_ON_OBS:
    #    K = int(len(regimes))
    #    cfg.global_dim = 4 + K
    #else:
    #    cfg.global_dim = 4
    cfg.per_asset_dim = int(getattr(globalcfg, "PER_ASSET_DIM", 5))

    roll_global_dim = 0
    if getattr(globalcfg, "ROLL_COV_SUMMARY_ON_OBS", False):
        roll_global_dim = 2 + int(getattr(globalcfg, "ROLL_TOP_EIGS", 2))

    K = int(len(regimes)) if getattr(globalcfg, "REGIME_GAMMA_ON_OBS", False) else 0
    cfg.global_dim = 4 + roll_global_dim + K
    
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
        per_dim=cfg.per_asset_dim,
    ).to(globalcfg.device)
    value  = ValueNetCLS(N, d_model=128, nlayers=2, nhead=4,global_dim=cfg.global_dim,per_dim=cfg.per_asset_dim,).to(globalcfg.device)

    opt_pi = optim.Adam(policy.parameters(), lr=cfg.lr_actor)
    opt_v  = optim.Adam(value.parameters(),  lr=cfg.lr_critic)
    
    BASE_SIGMAS = np.asarray(globalcfg.sigmas, float).reshape(-1)
    BASE_R = np.asarray(R_base, float)

    ms_cfg = MarketSamplerConfig(
        sigma_logstd=float(getattr(globalcfg, "REGIME_SIGMA_LOGSTD", 0.4)),
        sigma_clip=tuple(getattr(globalcfg, "REGIME_SIGMA_CLIP", (1e-4, 10.0))),
        corr_noise_std=float(getattr(globalcfg, "REGIME_CORR_NOISE", 0.08)),
    )
    market_sampler_fn = make_market_sampler(BASE_SIGMAS, BASE_R, cfg=ms_cfg)

    # -------------------------
    #  Stage 1: frictionless (λ=1.0, tiny width)
    # -------------------------
    print("[Stage 1] warmup (beta -> m)")
    policy.use_cash_softmax = False
    freeze_width_head_in_stage1(policy)
    for it in range(6):
        batch = rollout_joint(policy, value, cfg,
                          gcfg=globalcfg,
                          lam_choices=[1.0],
                          target_choices=target_choices,
                          stage=1,
                          batch_episodes=32,
                          R=R_base,
                          center_mode="policy",
                          market_sampler=market_sampler_fn,)

        assert isinstance(batch, dict) and all(k in batch for k in ["obs","m","s","logp","adv","ret"]), \
            f"Bad batch keys: {None if batch is None else list(batch.keys())}"
        ppo_update_joint(policy, value, opt_pi, opt_v, cfg, batch, env_cfg=globalcfg, stage=1, width_prior_w=0)

    # -------------------------
    # Stage 2
    # -------------------------
    print("[Stage 2] start (regime-switching GBM)")
    freeze_centers_in_stage2(policy)
    policy.use_cash_softmax = True
    opt_pi = optim.Adam(filter(lambda p: p.requires_grad, policy.parameters()), lr=cfg.lr_actor)

    stages = [
        #([0.90],  6, 32, 0.020),
        ([0.95],  8, 32, 0.015),
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
                target_choices=target_choices,
                stage=2,
                R=R_base,
                center_mode="policy",
                market_sampler=market_sampler_fn,
                env_ctor=env_ctor,
            )
            ppo_update_joint(policy, value, opt_pi, opt_v, cfg, batch, env_cfg=globalcfg, stage=2, width_prior_w=0.00)
            if (it+1) % 2 == 0:
                print(f"  upd {it+1:02d}: mean_annual_ret={batch['rew_ep_mean']/globalcfg.years:.4f}")

    # -------------------------
    # Save
    # -------------------------
    outdir = "checkpoints_regime_A2"
    os.makedirs(outdir, exist_ok=True)

    torch.save(policy.state_dict(), os.path.join(outdir, "policy_2stage_A2_minvar_obsupdate.pt"))
    torch.save(value.state_dict(),  os.path.join(outdir, "value_2stage_A2_minvar_obsupdate.pt"))
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
