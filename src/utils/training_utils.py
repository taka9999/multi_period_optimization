from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

from src.regime_gbm.gbm_env import globalsetting
from src.ppo.agent import JointBandPolicy, ValueNetCLS, PPOConfig
from src.utils.rlopt_helpers  import build_corr_from_pairs

@dataclass
class MarketSamplerConfig:
    # sigma
    sigma_logstd: float = 0.4
    sigma_clip: tuple = (1e-4, 10.0)

    # corr noise
    corr_noise_std: float = 0.08

    # mixture probs
    p_hi_corr: float = 0.25
    p_lo_corr: float = 0.25
    p_hi_cond: float = 0.15

    # scenario strengths
    mean_corr_shift_hi: float = 0.5
    mean_corr_shift_lo: float = -0.5
    mean_corr_shift_hi_cond: float = 0.2
    cond_factor_strength: float = 0.8


def _proj_to_corr_psd(M: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    M = np.asarray(M, float)
    M = 0.5 * (M + M.T)

    w, V = np.linalg.eigh(M)
    w = np.clip(w, eps, None)
    Mp = (V * w) @ V.T

    d = np.sqrt(np.clip(np.diag(Mp), eps, None))
    Mp = Mp / (d[:, None] * d[None, :])
    Mp = np.clip(Mp, -0.999, 0.999)
    np.fill_diagonal(Mp, 1.0)

    # PSD guard again
    Mp = 0.5 * (Mp + Mp.T)
    w2, V2 = np.linalg.eigh(Mp)
    if np.min(w2) < eps:
        w2 = np.clip(w2, eps, None)
        Mp = (V2 * w2) @ V2.T
        d = np.sqrt(np.clip(np.diag(Mp), eps, None))
        Mp = Mp / (d[:, None] * d[None, :])
        np.fill_diagonal(Mp, 1.0)
    return Mp


def _mix_corr(
    R_base: np.ndarray,
    rng: np.random.Generator,
    noise_std: float,
    mean_corr_shift: float = 0.0,
    hi_cond: bool = False,
    cond_factor_strength: float = 0.8,
) -> np.ndarray:
    N = R_base.shape[0]
    R = np.array(R_base, float)

    # shift avg corr up/down
    if mean_corr_shift != 0.0:
        a = float(np.clip(mean_corr_shift, -0.95, 0.95))
        if a > 0:
            J = np.ones((N, N), float)
            np.fill_diagonal(J, 1.0)
            R = (1 - a) * R + a * J
        else:
            a2 = -a
            I = np.eye(N)
            R = (1 - a2) * R + a2 * I

    # symmetric noise
    E = rng.normal(0.0, noise_std, size=(N, N))
    E = 0.5 * (E + E.T)
    np.fill_diagonal(E, 0.0)
    R = R + E
    np.fill_diagonal(R, 1.0)

    # high condition number via 1-factor component
    if hi_cond:
        v = rng.normal(size=N)
        v /= (np.linalg.norm(v) + 1e-12)
        strength = float(np.clip(cond_factor_strength, 0.0, 0.95))
        R = (1 - strength) * R + strength * np.outer(v, v)
        np.fill_diagonal(R, 1.0)

    return _proj_to_corr_psd(R)


def make_market_sampler(
    base_sigmas: np.ndarray,
    base_R: np.ndarray,
    cfg: Optional[MarketSamplerConfig] = None,
) -> Callable[[np.random.Generator, int], Tuple[np.ndarray, np.ndarray]]:
    """
    Returns a callable market_sampler(rng, k) -> (R_ep, sigmas_ep)
    """
    base_sigmas = np.asarray(base_sigmas, float).reshape(-1)
    base_R = np.asarray(base_R, float)
    N = base_sigmas.size

    if cfg is None:
        cfg = MarketSamplerConfig()

    def sampler(rng: np.random.Generator, k: int):
        # sigma: lognormal
        z = rng.normal(0.0, float(cfg.sigma_logstd), size=N)
        sig = base_sigmas * np.exp(z)
        lo, hi = cfg.sigma_clip
        sig = np.clip(sig, lo, hi)

        # corr: mixture
        u = rng.random()
        noise_std = float(cfg.corr_noise_std)

        if u < cfg.p_hi_corr:
            R = _mix_corr(base_R, rng, noise_std=noise_std,
                         mean_corr_shift=cfg.mean_corr_shift_hi,
                         hi_cond=False,
                         cond_factor_strength=cfg.cond_factor_strength)
        elif u < cfg.p_hi_corr + cfg.p_lo_corr:
            R = _mix_corr(base_R, rng, noise_std=noise_std,
                         mean_corr_shift=cfg.mean_corr_shift_lo,
                         hi_cond=False,
                         cond_factor_strength=cfg.cond_factor_strength)
        elif u < cfg.p_hi_corr + cfg.p_lo_corr + cfg.p_hi_cond:
            R = _mix_corr(base_R, rng, noise_std=noise_std,
                         mean_corr_shift=cfg.mean_corr_shift_hi_cond,
                         hi_cond=True,
                         cond_factor_strength=cfg.cond_factor_strength)
        else:
            R = _mix_corr(base_R, rng, noise_std=noise_std,
                         mean_corr_shift=0.0,
                         hi_cond=False,
                         cond_factor_strength=cfg.cond_factor_strength)

        return R, sig

    return sampler


def warmup_joint_beta_to_m(
    policy: JointBandPolicy,
    steps=800,
    bs=8192,
    lr=5e-4,
    lam_value=1.0,
    cfg: globalsetting=None,
    R: np.ndarray=None,
    target_choices=None,
):
    if cfg is None:
        raise ValueError("warmup_joint_beta_to_m: cfg(globalsetting) must be provided")
    device = cfg.device


    opt = torch.optim.Adam([p for n,p in policy.named_parameters() if "log_std" not in n], lr=lr)
    bce = nn.BCEWithLogitsLoss()  # ロジット vs 目標m（sigmoid内部でlogit化）

    policy.train()
    for _ in range(steps):
        # 合成バッチ
        beta = torch.empty(bs, cfg.N_ASSETS, device=device).uniform_(-0.95, 0.95)
        w    = torch.rand(bs, cfg.N_ASSETS, device=device)
        w    = (w / (w.sum(dim=1, keepdim=True)+1e-12)) * torch.rand(bs,1, device=device)  # sum<=1
        lam  = torch.full((bs,), float(lam_value), device=device)
        # target (annual) and per-step target used in global features
        if target_choices is None or len(target_choices) == 0:
            target_ann = torch.full((bs,), float(getattr(cfg, "TARGET_RET_ANN", 0.06)), device=device)
        else:
            choices = torch.tensor(target_choices, dtype=torch.float32, device=device)
            idx = torch.randint(0, choices.numel(), (bs,), device=device)
            target_ann = choices[idx]
        target_dt = target_ann * float(cfg.dt_day)

        if R is None:
            R_use = build_corr_from_pairs(cfg.N_ASSETS, base_rho=0.20, pair_rhos=cfg.pair_rhos, make_psd=True)
        else:
            R_use = R

        # 観測ベクトル N*5+4 を生成
        with torch.no_grad():
            sigma = torch.tensor(cfg.sigmas, dtype=torch.float32, device=device).unsqueeze(0).expand(bs, -1)
            R_t   = torch.tensor(R_use,      dtype=torch.float32, device=device)                  # [N,N]
            Rw    = torch.matmul(w, R_t.T)                                               # [B,N]
            per   = torch.stack([beta, w, sigma, Rw, lam.view(-1,1).expand(-1,cfg.N_ASSETS)], dim=-1)  # [B,N,5]
            per_f = per.view(bs, -1)                                                     # [B,N*5]
            Dsig  = torch.diag_embed(sigma)                                              # [B,N,N]
            Cov   = torch.matmul(torch.matmul(Dsig, R_t.expand(bs,-1,-1)), Dsig)         # [B,N,N]
            port_var = (w.unsqueeze(1) @ Cov @ w.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # [B]
            rw_norm = Rw.norm(dim=1)
            glob = torch.stack([lam, target_dt, port_var, rw_norm], dim=1)              # [B,4]
            obs = torch.cat([per_f, glob], dim=1)                                        # [B,N*5+4]

        # 前向き
        cls_out, tok_out = policy.body(per, glob)     # per=[B,N,5], glob=[B,4]
        m_mu = policy.head_m(tok_out).squeeze(-1)     # ロジット

        # 目標 m*（フリクションレス目安）
        # Heuristic warmup target for center (still a *training aid*):
        # tilt towards risky assets as target increases.
        base = ((beta + 0.95)/(2*0.95)).clamp(1e-4, 1-1e-4)
        scale = (target_ann / (float(getattr(cfg, "TARGET_RET_ANN", 0.06)) + 1e-12)).clamp(0.5, 1.5)
        m_star = (base * scale.view(-1,1)).clamp(1e-4, 1-1e-4)

        # 合計≤1のソフト制約（ロジットをsigmoidしてから）
        m_sig  = torch.sigmoid(m_mu)
        excess = torch.clamp(m_sig.sum(dim=1) - 1.0, min=0.0)
        simplex_pen = 1e-2 * (excess**2).mean()

        loss = bce(m_mu, m_star) + simplex_pen
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

def freeze_width_head_in_stage1(policy: JointBandPolicy):
    for n,p in policy.named_parameters():
        if 'head_s' in n or 'log_std_s' in n:
            p.requires_grad_(False)

def freeze_centers_in_stage2(policy: JointBandPolicy):
    for n, p in policy.named_parameters():
        if 'head_m' in n or 'log_std_m' in n:
            p.requires_grad_(False)
