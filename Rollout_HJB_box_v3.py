from __future__ import annotations

import numpy as np
import torch
import cvxpy as cp
from typing import List, Optional, Dict, Tuple

from GBMEnv_LQ_v4 import GBMBandEnvMulti, globalsetting
from RLopt_helpers import clamp01_vec, build_corr_from_pairs
from PPO_agent_v3 import JointBandPolicy, ValueNetCLS, PPOConfig


def compute_gae(rews: torch.Tensor, vals: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Episode-wise GAE."""
    T = len(rews)
    adv = torch.zeros(T, dtype=torch.float32)
    lastgaelam = 0.0
    nextvalue = 0.0
    nextnonterm = 0.0
    for t in reversed(range(T)):
        delta = rews[t] + gamma * nextvalue * nextnonterm - vals[t]
        lastgaelam = float(delta) + gamma * lam * nextnonterm * lastgaelam
        adv[t] = lastgaelam
        nextvalue = float(vals[t])
        nextnonterm = 0.0 if bool(dones[t]) else 1.0
    ret = adv + vals
    return adv, ret


def make_env(env_cfg: globalsetting, R: np.ndarray) -> GBMBandEnvMulti:
    return GBMBandEnvMulti(cfg=env_cfg, R=R)


# --- MV center solver (cached) ---
_MV_CACHE: Dict[Tuple[Tuple[float, ...], float, bool], np.ndarray] = {}

def mv_center_qp(Cov: np.ndarray, sigmas: np.ndarray, beta: np.ndarray, target_ann: float, *, allow_cash: bool = True, round_nd: int = 4, solver: str = "OSQP") -> np.ndarray:
    """
    Long-only MV center (discounted-by-bank world):
      mu_eff = (sigmas**2) * beta
      min w^T Cov w
      s.t. mu_eff^T w >= target_ann, w>=0, sum(w)<=1 (cash) or ==1 (no cash).
    Target is clipped to max(mu_eff) to avoid infeasibility.
    """
    beta = np.asarray(beta, float).reshape(-1)
    sigmas = np.asarray(sigmas, float).reshape(-1)
    Cov = np.asarray(Cov, float)
    mu_eff = (sigmas**2) * beta
    max_mu = float(np.max(mu_eff))
    targ = float(min(float(target_ann), max_mu)) if np.isfinite(max_mu) else float(target_ann)
    key = (tuple(np.round(beta, round_nd).tolist()), float(np.round(targ, round_nd)), bool(allow_cash))
    if key in _MV_CACHE:
        return _MV_CACHE[key].copy()
    n = len(mu_eff)
    w = cp.Variable(n)
    obj = cp.Minimize(cp.quad_form(w, Cov))
    cons = [w >= 0, mu_eff @ w >= targ]
    cons += [cp.sum(w) <= 1.0] if allow_cash else [cp.sum(w) == 1.0]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=getattr(cp, solver), verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        ww = np.zeros(n, float)
        ww[int(np.argmax(mu_eff))] = 1.0
    else:
        ww = np.maximum(0.0, np.array(w.value, dtype=float).reshape(-1))
        s = float(ww.sum())
        if allow_cash:
            if s > 1.0 + 1e-10:
                ww /= max(s, 1e-12)
        else:
            if s <= 1e-12:
                ww[:] = 0.0
                ww[int(np.argmax(mu_eff))] = 1.0
            else:
                ww /= s
    ww = clamp01_vec(ww)
    _MV_CACHE[key] = ww.copy()
    return ww

def compute_Dii(w: np.ndarray, Cov: np.ndarray) -> np.ndarray:
    """Return diag(D(w)) for the risky-weight process in the no-trade region.

    Setup (matching GBMEnv_LQ_v3):
      - risky weights w_i = S_i / (sum S + cash)
      - discounted-by-bank wealth (so cash is constant between trades)
      - risky prices follow correlated GBM with covariance Cov

    For the weight SDE  dw = b(w) dt + A(w) dW, the diffusion covariance is

        D(w) := A(w)A(w)^T = (diag(w) - w w^T) Cov (diag(w) - w w^T).

    The diagonal has the convenient closed form

        D_ii(w) = w_i^2 * (Cov_ii - 2 (Cov w)_i + w^T Cov w).

    This quantity *does* depend on cross-asset correlation via Cov.
    """
    w = np.asarray(w, dtype=float).reshape(-1)
    Cov = np.asarray(Cov, dtype=float)
    if Cov.shape[0] != Cov.shape[1] or Cov.shape[0] != w.size:
        raise ValueError("compute_Dii: shape mismatch between w and Cov")

    Cw = Cov @ w
    wCw = float(w @ Cw)
    Dii = (w**2) * (np.diag(Cov) - 2.0 * Cw + wCw)
    return np.maximum(Dii, 0.0)


def compute_delta_box(
    w_star: np.ndarray,
    Cov: np.ndarray,
    gamma: float,
    lam: float | np.ndarray,
    *,
    scale: float = 1.0,
    eps: float = 1e-12,
    clip: tuple[float, float] = (0.0, 1.0),
    corr_mode: str = "full",
) -> np.ndarray:
    """HJB/QVI-inspired box half-width multiplier per asset.

    Uses the classic small proportional-cost scaling (1/3 power):

        delta_i ∝ ( kappa_i * D_ii(w*) / Gamma_ii )^{1/3}

    where
      - kappa_i is the proportional *sell* cost rate (sell-only wedge): kappa_i = 1 - lam_i
      - D_ii(w*) is diag diffusion variance of the weight process at the center
      - Gamma_ii is local curvature; we approximate  Gamma ≈ gamma * Cov  ⇒ Gamma_ii ≈ gamma * Cov_ii

    Output is dimensionless, typically used as a multiplier in:
        b = 0.95 * s * delta * min(m, 1-m)
    """
    w_star = np.asarray(w_star, dtype=float).reshape(-1)
    Cov = np.asarray(Cov, dtype=float) if corr_mode == "full" else np.asarray(np.diag(np.diag(Cov)))
    N = w_star.size

    lam_vec = np.asarray(lam, dtype=float)
    if lam_vec.ndim == 0:
        lam_vec = np.full(N, float(lam_vec), dtype=float)
    else:
        lam_vec = lam_vec.reshape(-1)
        if lam_vec.size != N:
            raise ValueError("compute_delta_box: lam must be scalar or length N")

    kappa = np.maximum(0.0, 1.0 - lam_vec)  # sell-only cost rate
    Dii = compute_Dii(w_star, Cov)
    Gamma_ii = np.maximum(eps, float(gamma) * np.maximum(eps, np.diag(Cov)))

    delta = scale * np.power((kappa * Dii) / Gamma_ii, 1.0 / 3.0)
    lo, hi = clip
    return np.clip(delta, lo, hi)

def compute_delta_rotated(
    m_star: np.ndarray,
    Cov: np.ndarray,
    gamma: float,
    lam: float,
    *,
    scale: float = 1.0,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build Level-B (U, delta_z) using eigendecomposition of a* = Q(m)CovQ(m).
    delta_z_k ∝ (kappa * eig_a_k / Gamma_kk)^{1/3},
    where Gamma_kk ≈ gamma * (U^T Cov U)_kk.

    Returns (U, delta_z).
    """
    m_star = np.asarray(m_star, float).reshape(-1)
    Cov = np.asarray(Cov, float)
    N = m_star.size

    kappa = max(0.0, 1.0 - float(lam))

    # a* = Q Cov Q
    Q = np.diag(m_star) - np.outer(m_star, m_star)
    a = Q @ Cov @ Q
    a = 0.5 * (a + a.T)

    eigvals, U = np.linalg.eigh(a)
    eigvals = np.maximum(eigvals, 0.0)

    Cov_rot = U.T @ Cov @ U
    Gamma_kk = np.maximum(eps, float(gamma) * np.maximum(eps, np.diag(Cov_rot)))

    delta_z = scale * np.power((kappa * eigvals) / Gamma_kk, 1.0/3.0)
    delta_z = np.maximum(0.0, delta_z)
    return U, delta_z


@torch.no_grad()
def _sample_pair_rhos(rng: np.random.Generator, n_assets: int, n_edges: int, rho_low: float, rho_high: float) -> Dict[Tuple[int, int], float]:
    if n_edges <= 0:
        return {}
    pairs = [(i, j) for i in range(n_assets) for j in range(i + 1, n_assets)]
    n_edges = min(n_edges, len(pairs))
    idx = rng.choice(len(pairs), size=n_edges, replace=False)
    out: Dict[Tuple[int, int], float] = {}
    for k in idx:
        i, j = pairs[int(k)]
        out[(i, j)] = float(rng.uniform(rho_low, rho_high))
    return out


@torch.no_grad()
def rollout_joint(
    policy: JointBandPolicy,
    valuef: ValueNetCLS,
    cfg: PPOConfig,
    *,
    gcfg: globalsetting,
    lam_choices: List[float],
    target_choices: Optional[List[float]] = None,
    stage: int = 2,
    batch_episodes: Optional[int] = None,
    R: Optional[np.ndarray] = None,
    mv_allow_cash: bool = True,
    mv_round_nd: int = 4,
    mv_solver: str = "OSQP",
    corr_mode: str = "fixed",
    base_rho_low: float = -0.2,
    base_rho_high: float = 0.5,
    n_pair_edges: int = 2,
) -> Dict[str, torch.Tensor]:
    """
    MV center is computed outside the policy; policy outputs width only (s).
    Returned batch matches PPO_update expectations: obs, m, s, logp, adv, ret.
    """
    be = cfg.batch_episodes if batch_episodes is None else int(batch_episodes)
    N = gcfg.N_ASSETS
    rng = np.random.default_rng(getattr(gcfg, "seed", None))

    if corr_mode not in ("fixed", "random"):
        raise ValueError(f"rollout_joint: corr_mode must be 'fixed' or 'random', got {corr_mode}")

    if corr_mode == "fixed":
        if R is None:
            R = build_corr_from_pairs(
                N,
                base_rho=0.20,
                pair_rhos=getattr(gcfg, "pair_rhos", {}),
                make_psd=True,
            )

    obs_buf, m_buf, s_buf, logp_buf, adv_buf, ret_buf = [], [], [], [], [], []
    rew_ep = []

    for _ in range(be):
        beta = np.random.uniform(-0.95, 0.95, size=N)
        lam = 1.0 if stage == 1 else float(np.random.choice(lam_choices))
        target_ann = (float(np.random.choice(target_choices)) if (target_choices is not None and len(target_choices) > 0) else float(getattr(gcfg, "TARGET_RET_ANN", 0.06)))
        target_ann = max(target_ann, 0.0)

        if corr_mode == "random":
            base_rho = float(rng.uniform(base_rho_low, base_rho_high))
            pair_rhos = _sample_pair_rhos(
                rng,
                n_assets=N,
                n_edges=int(n_pair_edges),
                rho_low=base_rho_low,
                rho_high=base_rho_high,
            )
            R_ep = build_corr_from_pairs(N, base_rho=base_rho, pair_rhos=pair_rhos, make_psd=True)
        else:
            R_ep = np.asarray(R, float)

        env = make_env(gcfg, R_ep)
        obs = env.reset(beta=beta, lam=lam, target_ret=target_ann, w0=None)
        # Align to env-internal target (avoids mismatch)
        target_ann_eff = float(env.target_ret_ann)
        m_star = mv_center_qp(env.Cov, gcfg.sigmas, beta, target_ann_eff, allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver)

        ep_obs, ep_m, ep_s, ep_lp, ep_val, ep_rew, ep_done = [], [], [], [], [], [], []
        for t in range(env.T):
            o = torch.tensor(obs, dtype=torch.float32, device=gcfg.device).unsqueeze(0)
            v_t = valuef(o)

            if stage == 1:
                # stage1: fixed tiny width; no PPO on s (logp=0)
                s_np = np.full(N, 0.5, dtype=float)
                s_pre = torch.tensor(s_np, dtype=torch.float32, device=gcfg.device).unsqueeze(0)
                logp_use = torch.zeros(1, device=gcfg.device)
            else:
                s_t, logp_use, s_pre = policy.sample_s_only(o)
                s_np = s_t.squeeze(0).detach().cpu().numpy()

            m = m_star
            minm = np.minimum(m, 1.0 - m)
            if stage == 1:
                b = gcfg.STAGE1_WIDTH_COEF * 0.95 * minm
            else:
                #lam_scalar = float(obs[N*5 + 0])
                #g = (max(0.0, 1.0 - lam_scalar) + 1e-8) ** gcfg.ALPHA
                #b = 0.95 * s_np * g * minm
                # HJB/QVI-inspired correlation-aware width prior (box approximation).
                # In this codebase, `lam` is the sell-proceeds wedge in (0,1]; proportional sell cost is kappa = 1-lam.
                lam_scalar = float(obs[N*5 + 0])
                delta = compute_delta_box(
                    w_star=m_star,
                    Cov=env.Cov,
                    gamma=float(getattr(gcfg, "RISK_GAMMA", 1.0)),
                    lam=lam_scalar,
                    # Tune this if bands are too wide/narrow
                    scale=1.0,
                    clip=(0.0, 1.0),
                )
                b = 0.95 * s_np * delta * minm


            A = clamp01_vec(m - b)
            B = np.maximum(A + 1e-6, m + b)

            obs, r, done, _ = env.step(A, B, use_trade_penalty=(stage == 2))

            ep_obs.append(o.squeeze(0).detach().cpu())
            ep_m.append(torch.tensor(m, dtype=torch.float32))
            ep_s.append(s_pre.squeeze(0).detach().cpu())
            ep_lp.append(logp_use.squeeze(0).detach().cpu())
            ep_val.append(v_t.squeeze(0).detach().cpu())
            ep_rew.append(float(r))
            ep_done.append(bool(done))
            if done:
                break

        if len(ep_rew) == 0:
            continue
        rew_ep.append(float(np.sum(ep_rew)))

        obs_ep = torch.stack(ep_obs)
        m_ep   = torch.stack(ep_m)
        s_ep   = torch.stack(ep_s)
        lp_ep  = torch.stack(ep_lp)
        val_ep = torch.stack(ep_val)
        rew_t  = torch.tensor(ep_rew, dtype=torch.float32)
        done_t = torch.tensor(ep_done, dtype=torch.bool)

        adv_ep, ret_ep = compute_gae(rew_t, val_ep, done_t, cfg.gamma, cfg.gae_lambda)
        obs_buf.append(obs_ep)
        m_buf.append(m_ep)
        s_buf.append(s_ep)
        logp_buf.append(lp_ep)
        adv_buf.append(adv_ep)
        ret_buf.append(ret_ep)

    if len(obs_buf) == 0:
        raise RuntimeError("rollout_joint: collected 0 episodes")

    obs = torch.cat(obs_buf).to(gcfg.device)
    m   = torch.cat(m_buf).to(gcfg.device)
    s   = torch.cat(s_buf).to(gcfg.device)
    logp= torch.cat(logp_buf).to(gcfg.device)
    adv = torch.cat(adv_buf).to(gcfg.device)
    ret = torch.cat(ret_buf).to(gcfg.device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return dict(obs=obs, m=m, s=s, logp=logp, adv=adv, ret=ret, rew_ep_mean=float(np.mean(rew_ep)), rew_ep_std=float(np.std(rew_ep)))


@torch.no_grad()
def rollout_joint_levelB(
    policy: JointBandPolicy,
    valuef: ValueNetCLS,
    cfg: PPOConfig,
    *,
    gcfg: globalsetting,
    lam_choices: List[float],
    target_choices: Optional[List[float]] = None,
    batch_episodes: Optional[int] = None,
    R: Optional[np.ndarray] = None,
    mv_allow_cash: bool = True,
    mv_round_nd: int = 4,
    mv_solver: str = "OSQP",
    qp_solver: str = "OSQP",
    corr_mode: str = "fixed",
    base_rho_low: float = -0.2,
    base_rho_high: float = 0.5,
    n_pair_edges: int = 2,
) -> Dict[str, torch.Tensor]:
    """
    Stage-2 PPO batch collection under Level-B executor:
      - center m* from MV-QP (same as rollout_joint)
      - widths s from policy
      - execute with rotated-box projection: env.step_rotated_box(m*, U, b_z)
    Output batch matches ppo_update_joint expectations: obs, m, s, logp, adv, ret.
    """
    be = cfg.batch_episodes if batch_episodes is None else int(batch_episodes)
    N = gcfg.N_ASSETS
    rng = np.random.default_rng(getattr(gcfg, "seed", None))

    if corr_mode not in ("fixed", "random"):
        raise ValueError(f"rollout_joint_levelB: corr_mode must be 'fixed' or 'random', got {corr_mode}")

    if corr_mode == "fixed":
        if R is None:
            R = build_corr_from_pairs(
                N,
                base_rho=0.20,
                pair_rhos=getattr(gcfg, "pair_rhos", {}),
                make_psd=True,
            )

    obs_buf, m_buf, s_buf, logp_buf, adv_buf, ret_buf = [], [], [], [], [], []
    rew_ep = []

    for _ in range(be):
        beta = np.random.uniform(-0.95, 0.95, size=N)
        lam = float(np.random.choice(lam_choices))
        target_ann = (float(np.random.choice(target_choices))
                      if (target_choices is not None and len(target_choices) > 0)
                      else float(getattr(gcfg, "TARGET_RET_ANN", 0.06)))
        target_ann = max(target_ann, 0.0)

        if corr_mode == "random":
            base_rho = float(rng.uniform(base_rho_low, base_rho_high))
            pair_rhos = _sample_pair_rhos(
                rng,
                n_assets=N,
                n_edges=int(n_pair_edges),
                rho_low=base_rho_low,
                rho_high=base_rho_high,
            )
            R_ep = build_corr_from_pairs(N, base_rho=base_rho, pair_rhos=pair_rhos, make_psd=True)
        else:
            R_ep = np.asarray(R, float)

        env = make_env(gcfg, R_ep)
        obs = env.reset(beta=beta, lam=lam, target_ret=target_ann, w0=None)

        # align to env-internal target
        target_ann_eff = float(env.target_ret_ann)

        # MV center (same as A2 rollout)
        m_star = mv_center_qp(env.Cov, gcfg.sigmas, beta, target_ann_eff,
                              allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver)

        # rotated-basis prior computed ONCE per episode (fast + stable)
        U, delta_z = compute_delta_rotated(
            m_star=m_star,
            Cov=env.Cov,
            gamma=float(getattr(gcfg, "RISK_GAMMA", 1.0)),
            lam=lam,
            scale=1.0,
        )

        ep_obs, ep_m, ep_s, ep_lp, ep_val, ep_rew, ep_done = [], [], [], [], [], [], []
        for t in range(env.T):
            o = torch.tensor(obs, dtype=torch.float32, device=gcfg.device).unsqueeze(0)
            v_t = valuef(o)

            # sample s only (stage2)
            s_t, logp_use, s_pre = policy.sample_s_only(o)
            s_np = s_t.squeeze(0).detach().cpu().numpy()

            # total width in rotated coordinates
            b_z = 0.95 * s_np * delta_z

            obs, r, done, _ = env.step_rotated_box(
                m=m_star, U=U, b_z=b_z,
                allow_cash=mv_allow_cash,
                solver=qp_solver,
                use_trade_penalty=True,
            )

            ep_obs.append(o.squeeze(0).detach().cpu())
            ep_m.append(torch.tensor(m_star, dtype=torch.float32))
            ep_s.append(s_pre.squeeze(0).detach().cpu())
            ep_lp.append(logp_use.squeeze(0).detach().cpu())
            ep_val.append(v_t.squeeze(0).detach().cpu())
            ep_rew.append(float(r))
            ep_done.append(bool(done))
            if done:
                break

        if len(ep_rew) == 0:
            continue
        rew_ep.append(float(np.sum(ep_rew)))

        obs_ep = torch.stack(ep_obs)
        m_ep   = torch.stack(ep_m)
        s_ep   = torch.stack(ep_s)
        lp_ep  = torch.stack(ep_lp)
        val_ep = torch.stack(ep_val)
        rew_t  = torch.tensor(ep_rew, dtype=torch.float32)
        done_t = torch.tensor(ep_done, dtype=torch.bool)

        adv_ep, ret_ep = compute_gae(rew_t, val_ep, done_t, cfg.gamma, cfg.gae_lambda)
        obs_buf.append(obs_ep)
        m_buf.append(m_ep)
        s_buf.append(s_ep)
        logp_buf.append(lp_ep)
        adv_buf.append(adv_ep)
        ret_buf.append(ret_ep)

    if len(obs_buf) == 0:
        raise RuntimeError("rollout_joint_levelB: collected 0 episodes")

    obs = torch.cat(obs_buf).to(gcfg.device)
    m   = torch.cat(m_buf).to(gcfg.device)
    s   = torch.cat(s_buf).to(gcfg.device)
    logp= torch.cat(logp_buf).to(gcfg.device)
    adv = torch.cat(adv_buf).to(gcfg.device)
    ret = torch.cat(ret_buf).to(gcfg.device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return dict(
        obs=obs, m=m, s=s, logp=logp, adv=adv, ret=ret,
        rew_ep_mean=float(np.mean(rew_ep)), rew_ep_std=float(np.std(rew_ep))
    )


@torch.no_grad()
def rollout_eval_levelB(
    policy: JointBandPolicy,
    cfg: PPOConfig,
    *,
    gcfg: globalsetting,
    lam_choices: List[float],
    target_choices: Optional[List[float]] = None,
    batch_episodes: int = 16,
    R: np.ndarray,
    mv_allow_cash: bool = True,
    mv_round_nd: int = 4,
    mv_solver: str = "OSQP",
    qp_solver: str = "OSQP",
) -> Dict[str, float]:
    """
    Evaluate A2-trained policy under Level-B executor (rotated-box projection + sell/buy).
    """
    if R is None:
        raise ValueError("rollout_eval_levelB: R must be provided")

    N = gcfg.N_ASSETS
    rew_ep = []
    lret_ep = []

    for _ in range(batch_episodes):
        beta = np.random.uniform(-0.95, 0.95, size=N)
        lam = float(np.random.choice(lam_choices))
        target_ann = (float(np.random.choice(target_choices)) if (target_choices is not None and len(target_choices) > 0)
                      else float(getattr(gcfg, "TARGET_RET_ANN", 0.06)))
        target_ann = max(target_ann, 0.0)

        env = make_env(gcfg, R)
        obs = env.reset(beta=beta, lam=lam, target_ret=target_ann, w0=None)
        target_ann_eff = float(env.target_ret_ann)

        m_star = mv_center_qp(env.Cov, gcfg.sigmas, beta, target_ann_eff,
                              allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver)

        ep_rew = 0.0
        ep_lret = 0.0

        # rotated-basis width prior
        U, delta_z = compute_delta_rotated(
            m_star=m_star,
            Cov=env.Cov,
            gamma=float(getattr(gcfg, "RISK_GAMMA", 1.0)),
            lam=lam,
            scale=1.0,
        )

        for t in range(env.T):
            o = torch.tensor(obs, dtype=torch.float32, device=gcfg.device).unsqueeze(0)
            # width scale from trained policy (stage2 head_s)
            s_t, _, _ = policy.sample_s_only(o)
            s_np = s_t.squeeze(0).detach().cpu().numpy()


            # total width in z-space
            # (use same "min(m,1-m)" shrink, but in z basis we can just use scalar safety factor)
            b_z = 0.95 * s_np * delta_z

            obs, r, done, lret = env.step_rotated_box(
                m=m_star,
                U=U,
                b_z=b_z,
                allow_cash=mv_allow_cash,
                solver=qp_solver,
                use_trade_penalty=True,
            )
            ep_rew += float(r)
            ep_lret += float(lret)
            if done:
                break

        rew_ep.append(ep_rew)
        lret_ep.append(ep_lret)

    return dict(
        rew_ep_mean=float(np.mean(rew_ep)),
        rew_ep_std=float(np.std(rew_ep)),
        lret_ep_mean=float(np.mean(lret_ep)),
        lret_ep_std=float(np.std(lret_ep)),
    )
