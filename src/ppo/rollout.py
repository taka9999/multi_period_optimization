from __future__ import annotations

from typing import List, Optional, Dict, Tuple, Callable
import copy

import numpy as np
import torch
import cvxpy as cp

from src.regime_gbm.gbm_env import GBMBandEnvMulti, globalsetting
from src.utils.rlopt_helpers import clamp01_vec
from src.ppo.agent import JointBandPolicy, ValueNetCLS, PPOConfig


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

def _call_env_ctor(env_ctor, *, gcfg_ep, R_ep, seed_ep):
    """Call env_ctor in a backward-compatible way.
    Supported signatures:
      env_ctor(gcfg_ep, R_ep, seed=seed_ep)
      env_ctor(gcfg_ep, R_ep, seed_ep)
      env_ctor(seed=seed_ep)
      env_ctor(seed_ep)
      env_ctor()
    """
    if env_ctor is None:
        raise ValueError("env_ctor is None")
    try:
        return env_ctor(gcfg_ep, R_ep, seed=seed_ep)
    except TypeError:
        pass
    try:
        return env_ctor(gcfg_ep, R_ep, seed_ep)
    except TypeError:
        pass
    try:
        return env_ctor(seed=seed_ep)
    except TypeError:
        pass
    try:
        return env_ctor(seed_ep)
    except TypeError:
        pass
    return env_ctor()


def _cov_cache_key(Cov: np.ndarray, *, round_nd: int) -> bytes:
    """Hashable cache key for Cov.

    IMPORTANT: if you domain-randomize (R, sigmas), Cov changes episode-by-episode.
    The MV-center cache MUST depend on Cov; otherwise it will silently reuse the
    wrong center.
    """
    Cov_r = np.round(np.asarray(Cov, float), round_nd).astype(np.float32, copy=False)
    return Cov_r.tobytes(order="C")


# --- MV center solver (cached) ---
_MV_CACHE: Dict[Tuple[bytes, Tuple[float, ...], float, bool], np.ndarray] = {}

def mv_center_qp(
    Cov, sigmas, beta, target_ann,
    allow_cash=True,
    round_nd=6,
    solver="OSQP",
):
    """
    min w^T Cov w
    s.t. (optional) mu_eff^T w >= target_ann
         w >= 0, sum(w) <= 1 (cash allowed)
    If target_ann is None: returns GMV (no return constraint).
    """

    Cov = np.asarray(Cov, float)
    sigmas = np.asarray(sigmas, float).reshape(-1)
    n = len(sigmas)

    # --- caching key robust ---
    cov_key = _cov_cache_key(Cov, round_nd=round_nd)
    targ_key = None if target_ann is None else float(np.round(float(target_ann), round_nd))

    # beta is only relevant when target constraint is active
    if target_ann is None:
        beta_key = None
    else:
        if beta is None:
            raise ValueError("mv_center_qp: beta must be provided when target_ann is not None.")
        beta = np.asarray(beta, float).reshape(-1)
        beta_key = tuple(np.round(beta, round_nd).tolist())

    key = (cov_key, beta_key, targ_key, bool(allow_cash))
    if key in _MV_CACHE:
        return _MV_CACHE[key].copy()

    # --- decision variable ---
    w = cp.Variable(n)

    # --- objective (variance) ---
    obj = cp.Minimize(cp.quad_form(w, Cov))

    # --- constraints ---
    cons = [w >= 0]
    if allow_cash:
        cons += [cp.sum(w) <= 1]
    else:
        cons += [cp.sum(w) == 1]

    # --- optional return constraint ---
    if target_ann is not None:
        mu_eff = (sigmas**2) * np.asarray(beta, float).reshape(-1)
        max_mu = float(np.max(mu_eff))
        targ = float(min(float(target_ann), max_mu)) if np.isfinite(max_mu) else float(target_ann)
        cons += [mu_eff @ w >= targ]

    prob = cp.Problem(obj, cons)

    try:
        prob.solve(solver=getattr(cp, solver), verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        # fallback: all cash or uniform tiny risky (pick your preference)
        ww = np.zeros(n, float)
    else:
        ww = np.asarray(w.value, float).reshape(-1)

    ww = np.clip(ww, 0.0, 1.0)
    if allow_cash:
        ssum = ww.sum()
        if ssum > 1.0:
            ww /= ssum
    else:
        ssum = ww.sum()
        if ssum <= 0:
            ww[:] = 1.0 / n
        else:
            ww /= ssum

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

    # eigh returns ascending eigenvalues; we want principal directions first
    eigvals, U = np.linalg.eigh(a)
    eigvals = np.maximum(eigvals, 0.0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    U = U[:, order]

    Cov_rot = U.T @ Cov @ U
    Gamma_kk = np.maximum(eps, float(gamma) * np.maximum(eps, np.diag(Cov_rot)))

    delta_z = scale * np.power((kappa * eigvals) / Gamma_kk, 1.0/3.0)
    delta_z = np.maximum(0.0, delta_z)
    return U, delta_z

def apply_topk_s(s:np.ndarray, *, topk: int | None) -> np.ndarray:
    s = np.asarray(s, dtype = float).reshape(-1)
    if topk is None:
        return s
    k = int(topk)
    if k <= 0:
        return np.ones_like(s)
    if k >= s.size:
        return s
    out = np.ones_like(s)
    out[:k] = s[:k]
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
    market_sampler: Optional[Callable[[np.random.Generator, int], Tuple[np.ndarray, np.ndarray]]] = None,
    base_seed: int = 1234,
    seed_offset: int = 0,
    mv_allow_cash: bool = True,
    mv_round_nd: int = 4,
    mv_solver: str = "OSQP",
    corr_mode: str = "full",
    env_ctor=None,
) -> Dict[str, torch.Tensor]:
    """
    MV center is computed outside the policy; policy outputs width only (s).
    Returned batch matches PPO_update expectations: obs, m, s, logp, adv, ret.
    """
    if R is None and market_sampler is None:
        raise ValueError("rollout_joint: provide either R or market_sampler")
    be = cfg.batch_episodes if batch_episodes is None else int(batch_episodes)
    N = gcfg.N_ASSETS

    obs_buf, m_buf, s_buf, logp_buf, adv_buf, ret_buf = [], [], [], [], [], []
    rew_ep = []

    for k in range(be):
        rng = np.random.default_rng(int(base_seed) + int(seed_offset) + int(k))
        seed_ep = int(base_seed) + int(seed_offset) + int(k)

        # --- domain randomization (optional) ---
        if market_sampler is not None:
            R_ep, sigmas_ep = market_sampler(rng, k)
            gcfg_ep = copy.copy(gcfg)
            gcfg_ep.seed = int(seed_ep)
            gcfg_ep.sigmas = np.asarray(sigmas_ep, float).reshape(-1)
        else:
            R_ep = np.asarray(R, float)
            gcfg_ep = copy.copy(gcfg)
            gcfg_ep.seed = int(seed_ep)

        beta = rng.uniform(-0.95, 0.95, size=N)
        lam = 1.0 if stage == 1 else float(rng.choice(lam_choices))
        target_ann = (float(rng.choice(target_choices)) if (target_choices is not None and len(target_choices) > 0) else float(getattr(gcfg_ep, "TARGET_RET_ANN", 0.06)))
        target_ann = max(target_ann, 0.0)

        if env_ctor is None:
            env = make_env(gcfg_ep, R_ep)
        else:
            env = _call_env_ctor(env_ctor, gcfg_ep=gcfg_ep, R_ep=R_ep, seed_ep=seed_ep)

        obs = env.reset(beta=beta, lam=lam, target_ret=target_ann, w0=None)
        # Align to env-internal target (avoids mismatch)
        target_ann_eff = float(env.target_ret_ann)
        m_star = mv_center_qp(env.Cov, gcfg_ep.sigmas, beta, target_ann_eff, allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver)

        ep_obs, ep_m, ep_s, ep_lp, ep_val, ep_rew, ep_done = [], [], [], [], [], [], []
        for t in range(env.T):
            o = torch.tensor(obs, dtype=torch.float32, device=gcfg_ep.device).unsqueeze(0)
            v_t = valuef(o)

            if stage == 1:
                # stage1: fixed tiny width; no PPO on s (logp=0)
                s_np = np.full(N, 0.5, dtype=float)
                s_pre = torch.tensor(s_np, dtype=torch.float32, device=gcfg_ep.device).unsqueeze(0)
                logp_use = torch.zeros(1, device=gcfg_ep.device)
            else:
                s_t, logp_use, s_pre = policy.sample_s_only(o)
                s_np = s_t.squeeze(0).detach().cpu().numpy()

            m = m_star
            minm = np.minimum(m, 1.0 - m)
            if stage == 1:
                b = gcfg_ep.STAGE1_WIDTH_COEF * 0.95 * minm
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
                    gamma=float(getattr(gcfg_ep, "RISK_GAMMA", 1.0)),
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
    market_sampler: Optional[Callable[[np.random.Generator, int], Tuple[np.ndarray, np.ndarray]]] = None,
    base_seed: int = 1234,
    seed_offset: int = 0,
    mv_allow_cash: bool = True,
    mv_round_nd: int = 4,
    mv_solver: str = "OSQP",
    qp_solver: str = "OSQP",
    topk: int | None = None,
    env_ctor=None,
) -> Dict[str, torch.Tensor]:
    """
    Stage-2 PPO batch collection under Level-B executor:
      - center m* from MV-QP (same as rollout_joint)
      - widths s from policy
      - execute with rotated-box projection: env.step_rotated_box(m*, U, b_z)
    Output batch matches ppo_update_joint expectations: obs, m, s, logp, adv, ret.
    """
    if R is None and market_sampler is None:
        raise ValueError("rollout_joint_levelB: provide either R or market_sampler")
    be = cfg.batch_episodes if batch_episodes is None else int(batch_episodes)
    N = gcfg.N_ASSETS

    obs_buf, m_buf, s_buf, logp_buf, adv_buf, ret_buf = [], [], [], [], [], []
    rew_ep = []

    for k in range(be):
        rng = np.random.default_rng(int(base_seed) + int(seed_offset) + int(k))
        seed_ep = int(base_seed) + int(seed_offset) + int(k)

        if market_sampler is not None:
            R_ep, sigmas_ep = market_sampler(rng, k)
            gcfg_ep = copy.copy(gcfg)
            gcfg_ep.seed = int(seed_ep)
            gcfg_ep.sigmas = np.asarray(sigmas_ep, float).reshape(-1)
        else:
            R_ep = np.asarray(R, float)
            gcfg_ep = copy.copy(gcfg)
            gcfg_ep.seed = int(seed_ep)

        beta = rng.uniform(-0.95, 0.95, size=N)
        lam = float(rng.choice(lam_choices))
        target_ann = (float(rng.choice(target_choices))
                      if (target_choices is not None and len(target_choices) > 0)
                      else float(getattr(gcfg_ep, "TARGET_RET_ANN", 0.06)))
        target_ann = max(target_ann, 0.0)

        if env_ctor is None:
            env = make_env(gcfg_ep, R_ep)
        else:
            env = _call_env_ctor(env_ctor, gcfg_ep=gcfg_ep, R_ep=R_ep, seed_ep=seed_ep)

        obs = env.reset(beta=beta, lam=lam, target_ret=target_ann, w0=None)

        # align to env-internal target
        target_ann_eff = float(env.target_ret_ann)

        # MV center (same as A2 rollout)
        m_star = mv_center_qp(env.Cov, gcfg_ep.sigmas, beta, target_ann_eff,
                              allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver)

        # rotated-basis prior computed ONCE per episode (fast + stable)
        U, delta_z = compute_delta_rotated(
            m_star=m_star,
            Cov=env.Cov,
            gamma=float(getattr(gcfg_ep, "RISK_GAMMA", 1.0)),
            lam=lam,
            scale=1.0,
        )

        ep_obs, ep_m, ep_s, ep_lp, ep_val, ep_rew, ep_done = [], [], [], [], [], [], []
        for t in range(env.T):
            o = torch.tensor(obs, dtype=torch.float32, device=gcfg_ep.device).unsqueeze(0)
            v_t = valuef(o)

            # sample s only (stage2)
            s_t, logp_use, s_pre = policy.sample_s_only(o)
            s_np = s_t.squeeze(0).detach().cpu().numpy()

            s_eff = apply_topk_s(s_np, topk=topk)
            b_z = 0.95 * s_eff * delta_z

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
    R: Optional[np.ndarray] = None,
    market_sampler: Optional[Callable[[np.random.Generator, int], Tuple[np.ndarray, np.ndarray]]] = None,
    base_seed: int = 777,
    seed_offset: int = 0,
    mv_allow_cash: bool = True,
    mv_round_nd: int = 4,
    mv_solver: str = "OSQP",
    qp_solver: str = "OSQP",
    topk: int | None = None,
    env_ctor=None,
) -> Dict[str, float]:
    """
    Evaluate A2-trained policy under Level-B executor (rotated-box projection + sell/buy).
    """
    #if R is None and market_sampler is None:
    #    raise ValueError("rollout_eval_levelB: provide either R or market_sampler")
    

    N = gcfg.N_ASSETS
    rew_ep = []
    lret_ep = []
    
    R_ep = np.asarray(R, float) if R is not None else np.eye(N, dtype=float)

    for k in range(int(batch_episodes)):
        rng = np.random.default_rng(int(base_seed) + int(seed_offset) + int(k))
        seed_ep = int(base_seed) + int(seed_offset) + int(k)

        if market_sampler is not None:
            R_ep, sigmas_ep = market_sampler(rng, k)
            gcfg_ep = copy.copy(gcfg)
            gcfg_ep.seed = int(seed_ep)
            gcfg_ep.sigmas = np.asarray(sigmas_ep, float).reshape(-1)
        else:
            #R_ep = np.asarray(R, float)
            # If env_ctor is used, R_ep is irrelevant; keep a dummy for readability
            R_ep = np.asarray(R, float) if R is not None else np.eye(N, dtype=float)
            gcfg_ep = copy.copy(gcfg)
            gcfg_ep.seed = int(seed_ep)

        beta = rng.uniform(-0.95, 0.95, size=N)
        lam = float(rng.choice(lam_choices))
        target_ann = (float(rng.choice(target_choices)) if (target_choices is not None and len(target_choices) > 0)
                      else float(getattr(gcfg_ep, "TARGET_RET_ANN", 0.06)))
        target_ann = max(target_ann, 0.0)

        if env_ctor is None:
            env = make_env(gcfg_ep, R_ep)
        else:
            # NOTE: env_ctor is expected to close over regime_json / P / etc.
            # and return a ready-to-use env instance.
            env = _call_env_ctor(env_ctor, gcfg_ep=gcfg_ep, R_ep=R_ep, seed_ep=seed_ep)
        obs = env.reset(beta=beta, lam=lam, target_ret=target_ann, w0=None)
        target_ann_eff = float(env.target_ret_ann)

        beta_for_mv = np.asarray(getattr(env, "beta", beta),float).reshape(-1)
        m_star = mv_center_qp(
            env.Cov, gcfg_ep.sigmas, beta_for_mv, target_ann_eff,
            allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver
        )

        #m_star = mv_center_qp(env.Cov, gcfg_ep.sigmas, beta, target_ann_eff,
        #                      allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver)

        ep_rew = 0.0
        ep_lret = 0.0

        # rotated-basis width prior
        U, delta_z = compute_delta_rotated(
            m_star=m_star,
            Cov=env.Cov,
            gamma=float(getattr(gcfg_ep, "RISK_GAMMA", 1.0)),
            lam=lam,
            scale=1.0,
        )

        for t in range(env.T):
            o = torch.tensor(obs, dtype=torch.float32, device=gcfg_ep.device).unsqueeze(0)
            # width scale from trained policy (stage2 head_s)
            s_t, _, _ = policy.sample_s_only(o)
            s_np = s_t.squeeze(0).detach().cpu().numpy()

            s_eff = apply_topk_s(s_np, topk=topk)
            b_z = 0.95 * s_eff * delta_z

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
