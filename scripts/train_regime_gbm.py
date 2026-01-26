# train_regime_gbm.py
import os, json, random
import copy

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


def main():
    # -------------------------
    # Config (same as notebook)
    # -------------------------
    globalcfg = globalsetting(
        seed    = 42,
        device  = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        N_ASSETS = 5,
        sigmas   = np.array([0.40, 0.30, 0.12, 0.22, 0.25], dtype=float),
        pair_rhos = {
            (0,1): 0.60,
            (1,3): -0.20,
            (2,4): 0.05,
        },
        DISCOUNT_BY_BANK = True,
        INIT_W0_UNIFORM  = True,
        BAND_SMOOTH_COEF = 0.0,
        TRADE_PEN_COEF   = 0.4,
        ALPHA = 1/3,
        STAGE1_WIDTH_COEF = 0.05,
    )
    # --- episode-level regime randomization ---
    # Each episode samples {beta_k, sigmas_k, R_k} for all regimes and keeps them fixed within the episode.
    globalcfg.REGIME_EPISODE_RANDOMIZE = True
    globalcfg.REGIME_BETA_STD = 0.05        # std for beta perturbation
    globalcfg.REGIME_SIGMA_LOGSTD = 0.10    # log-std for sigma multiplicative noise
    globalcfg.REGIME_CORR_NOISE = 0.02      # additive noise on correlation matrix entries
    globalcfg.REGIME_BETA_CLIP = 0.999
    globalcfg.REGIME_SIGMA_CLIP = (1e-4, 10.0)
    set_seed(globalcfg.seed, globalcfg.device)

    # baseline R/Cov used for MV-center computations in rollout code
    R_base = build_corr_from_pairs(globalcfg.N_ASSETS, base_rho=0.20, pair_rhos=globalcfg.pair_rhos, make_psd=True)
    _ = build_cov(sigmas=globalcfg.sigmas, R=R_base, make_psd=True)

    target_choices = [0.02, 0.04, 0.06, 0.08]

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

    regimes = [
        dict(
            beta=np.ones(N)*0.6,
            sigmas=np.array([0.18, 0.15, 0.10, 0.12, 0.14]),
            R=corr_from_base(0.25, {(0,1):0.55})
        ),
        dict(
            beta=np.ones(N)*(-0.2),
            sigmas=np.array([0.45, 0.35, 0.18, 0.28, 0.32]),
            R=corr_from_base(0.60, {(1,3):-0.15})
        )
    ]
    P = np.array([
        [0.97, 0.03],
        [0.10, 0.90]
    ], float)

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
    policy = JointBandPolicy(N, d_model=128, nlayers=2, nhead=4, use_cash_softmax=True).to(globalcfg.device)
    value  = ValueNetCLS(N, d_model=128, nlayers=2, nhead=4).to(globalcfg.device)

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
        ([0.90],  6, 32, 0.020),
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
                env_ctor=env_ctor,     # ★ ここだけ追加
            )
            ppo_update_joint(policy, value, opt_pi, opt_v, cfg, batch, env_cfg=globalcfg, stage=2, width_prior_w=0.02)
            if (it+1) % 2 == 0:
                print(f"  upd {it+1:02d}: mean_annual_ret={batch['rew_ep_mean']/globalcfg.years:.4f}")

    # -------------------------
    # Save
    # -------------------------
    outdir = "checkpoints_regime_A2"
    os.makedirs(outdir, exist_ok=True)

    torch.save(policy.state_dict(), os.path.join(outdir, "policy_stage2_A2_regime.pt"))
    torch.save(value.state_dict(),  os.path.join(outdir, "value_stage2_A2_regime.pt"))
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
