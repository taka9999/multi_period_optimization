from __future__ import annotations

from typing import List, Optional, Dict, Tuple, Callable
import copy
import warnings

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

def _use_target_constraint(gcfg) -> bool:
    # explicit override wins
    if hasattr(gcfg, "MV_USE_TARGET"):
        return bool(getattr(gcfg, "MV_USE_TARGET"))
    # default heuristic
    return float(getattr(gcfg, "TARGET_ETA", 1.0)) > 0.0

# --- MV center solver (cached) ---
_MV_CACHE: Dict[Tuple[bytes, Tuple[float, ...], float, bool], np.ndarray] = {}

def mv_center_qp_old(
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

# --- MV center solver (cached) ---
# _MV_CACHE: (Cov,beta,target,allow_cash) -> solution w*  (existing)
# plus: _MV_PROB_CACHE: (n,allow_cash,has_target) -> reusable CVXPY problem

_MV_PROB_CACHE = {}

class _MVProblem:
    def __init__(self, n: int, allow_cash: bool, has_target: bool):
        self.n = int(n)
        self.allow_cash = bool(allow_cash)
        self.has_target = bool(has_target)

        self.w = cp.Variable(self.n)

        # Parameters (updated each call)
        self.Cov = cp.Parameter((self.n, self.n), PSD=True)
        if self.has_target:
            self.mu = cp.Parameter(self.n)
            self.target = cp.Parameter(nonneg=True)

        cons = [self.w >= 0]
        if self.allow_cash:
            cons += [cp.sum(self.w) <= 1]
        else:
            cons += [cp.sum(self.w) == 1]

        if self.has_target:
            cons += [self.mu @ self.w >= self.target]

        obj = cp.Minimize(cp.quad_form(self.w, self.Cov))
        self.prob = cp.Problem(obj, cons)

    def solve(self, Cov, mu=None, target=None, *, solver="OSQP"):
        #self.Cov.value = np.asarray(Cov, float)
        # Symmetrize (and tiny jitter if needed) to avoid PSD=True parameter errors
        C = np.asarray(Cov, dtype=float)
        C = 0.5 * (C + C.T)
        self.Cov.value = C
        if self.has_target:
            self.mu.value = np.asarray(mu, float).reshape(-1)
            self.target.value = float(target)

        # warm_start=True: 前回 w.value を初期値として使う
        try:
            self.prob.solve(
                solver=getattr(cp, solver),
                warm_start=True,
                verbose=False,
            )
        except Exception:
            # fallback
            self.prob.solve(solver=cp.SCS, warm_start=True, verbose=False)

        if self.w.value is None:
            return None
        return np.asarray(self.w.value, float).reshape(-1)


def _get_mv_prob(n: int, *, allow_cash: bool, has_target: bool) -> _MVProblem:
    key = (int(n), bool(allow_cash), bool(has_target))
    prob = _MV_PROB_CACHE.get(key)
    if prob is None:
        prob = _MVProblem(n=int(n), allow_cash=allow_cash, has_target=has_target)
        _MV_PROB_CACHE[key] = prob
    return prob

def mv_center_qp(
    Cov, sigmas, beta, target_ann,
    allow_cash=False,
    round_nd=6,
    solver="OSQP",
):
    """
    min w^T Cov w
    s.t. (optional) mu_eff^T w >= target_ann
         w >= 0, sum(w) <= 1 (cash allowed)
    If target_ann is None: returns GMV (no return constraint).

    Speedups:
      - solution cache (_MV_CACHE): if hit => no solve
      - problem cache (_MV_PROB_CACHE): if miss => reuse cvxpy Problem + warm-start
    """
    Cov = np.asarray(Cov, float)
    sigmas = np.asarray(sigmas, float).reshape(-1)
    n = len(sigmas)

    # --- robust cache key (same spirit as current) ---
    cov_key = _cov_cache_key(Cov, round_nd=round_nd)
    targ_key = None if target_ann is None else float(np.round(float(target_ann), round_nd))

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

    # --- build mu/target if needed ---
    has_target = (target_ann is not None)
    if has_target:
        mu_eff = (sigmas**2) * np.asarray(beta, float).reshape(-1)
        max_mu = float(np.max(mu_eff))
        targ = float(min(float(target_ann), max_mu)) if np.isfinite(max_mu) else float(target_ann)
        targ = max(0.0, targ)
    else:
        mu_eff = None
        targ = None

    # --- reuse cvxpy problem + warm-start ---
    prob = _get_mv_prob(n, allow_cash=allow_cash, has_target=has_target)
    ww = prob.solve(Cov, mu=mu_eff, target=targ, solver=solver)

    if ww is None:
        ww = np.zeros(n, float)

    # normalize / clip (same as current)
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

    Setup:
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
    gamma_eff = 1 + gamma
    Gamma_ii = np.maximum(eps, float(gamma_eff) * np.maximum(eps, np.diag(Cov)))

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
    # --- added knobs ---
    eig_floor_rel: float = 1e-10,   # relative floor vs trace(a)/N
    eig_floor_abs: float = 0.0,     # optional absolute floor (rarely needed)
    jitter_cov: float = 0.0,        # e.g. 1e-10 if Cov is ill-conditioned
    debug: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    # Force stable float64 and sanitize weights (important near the simplex boundary).
    m_star = np.asarray(m_star, dtype=np.float64).reshape(-1)
    Cov = np.asarray(Cov, dtype=np.float64)
    N = m_star.size

    kappa = max(0.0, 1.0 - float(lam))
    # Small numeric drift can push weights slightly outside [0,1] or violate sum<=1 (cash).
    # Clipping here does not change the intended center materially but improves stability.
    m_star = np.clip(m_star, 0.0, 1.0)
    s = float(m_star.sum())
    if s > 1.0 + 1e-12:
        m_star = m_star / s

    # a* = Q Cov Q (symmetrized; use float64 for stable eigh)
    Q = np.diag(m_star) - np.outer(m_star, m_star)
    a = Q @ Cov @ Q
    a = 0.5 * (a + a.T)

    tr_a = float(np.trace(a))
    tr_C = float(np.trace(Cov))
    if tr_a <= max(1e-18, eps) * max(1.0, tr_C):
        # Optional debug:
        # print(f"[delta_rot] degenerate: trace(a)={tr_a:.3e}, trace(Cov)={tr_C:.3e} -> U=I, delta=0")
        return np.eye(N, dtype=np.float64), np.zeros(N, dtype=np.float64)

    # Regular case: do NOT floor eigenvalues (keeps theory intact).
    eigvals, U = np.linalg.eigh(a)
    eigvals = np.maximum(eigvals, 0.0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    U = U[:, order]

    # ---- stability floor (key patch) ----
    tr = float(np.trace(a))
    scale_ref = tr / max(1, N)              # typical eigenvalue scale
    floor_rel = float(eig_floor_rel) * max(scale_ref, 0.0)
    floor = max(float(eig_floor_abs), floor_rel, 0.0)

    # apply floor only to very small eigenvalues
    if floor > 0.0:
        eigvals = np.maximum(eigvals, floor)

    Cov_rot = U.T @ Cov @ U
    gamma_eff = 1.0 + float(gamma)
    Gamma_kk = np.maximum(eps, gamma_eff * np.maximum(eps, np.diag(Cov_rot)))

    delta_z = scale * np.power((kappa * eigvals) / Gamma_kk, 1.0 / 3.0)
    delta_z = np.maximum(0.0, delta_z)

    if debug:
        print(f"[delta_rot] trace(a)={tr:.3e} scale_ref={scale_ref:.3e} floor={floor:.3e}")
        print(f"[delta_rot] eig(min/med/max)={eigvals.min():.3e}/{np.median(eigvals):.3e}/{eigvals.max():.3e}")
        print(f"[delta_rot] delta(min/med/max)={delta_z.min():.3e}/{np.median(delta_z):.3e}/{delta_z.max():.3e}")
        print("[delta_rot dbg] lam", lam, "kappa", 1-lam)
        print("[delta_rot dbg] m*: min/med/max/sum",
            m_star.min(), np.median(m_star), m_star.max(), m_star.sum())
        print("[delta_rot dbg] trace(a)", np.trace(a), "||a||_F", np.linalg.norm(a))
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
    mv_allow_cash: bool = False,
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

    TOPK = 2   # for debug

    obs_buf, m_buf, s_buf, logp_buf, adv_buf, ret_buf = [], [], [], [], [], []
    rew_ep = []
    diag = {
            "b_min": [], "b_mean": [], "b_max": [], "b_med": [], "b_zero_frac": [],
            "trade_rate": [], "avg_turnover": [], "turnover_sum": [], "steps": [],
            }
    diag.update(dict(
        gap_over_mean = [],gap_under_mean = [],gap_any_mean = [],
        sqrtDii_mean = [],sqrtDii_med  = [],sqrtDii_max  = [],
        ))

    diag_ep = {
        "minm_zero": [], "minm_min": [], "minm_med": [], "minm_max": [],
        "delta_zero": [], "delta_min": [], "delta_med": [], "delta_max": [],
        "s_zero": [], "s_min": [], "s_mean": [], "s_max": [],
        "b_zero": [],
    }

    for k in range(be):
        rng = np.random.default_rng(int(base_seed) + int(seed_offset) + int(k))
        seed_ep = int(base_seed) + int(seed_offset) + int(k)
        gap_over_list  = []
        gap_under_list = []
        gap_any_list   = []

        sqrtDii_mean_list = []
        sqrtDii_med_list  = []
        sqrtDii_max_list  = []

        gap_post_list = []

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

        if _use_target_constraint(gcfg_ep) and (target_choices is not None and len(target_choices) > 0):
             target_ann = max(float(rng.choice(target_choices)), 0.0)
        elif _use_target_constraint(gcfg_ep):
             target_ann = max(float(getattr(gcfg_ep, "TARGET_RET_ANN", 0.06)), 0.0)
        else:
            target_ann = None

        if env_ctor is None:
            env = make_env(gcfg_ep, R_ep)
        else:
            env = _call_env_ctor(env_ctor, gcfg_ep=gcfg_ep, R_ep=R_ep, seed_ep=seed_ep)

        obs = env.reset(beta=beta, lam=lam, target_ret=target_ann, w0=None)
        target_ann_eff = (float(env.target_ret_ann) if target_ann is not None else None)
        sigmas_for_mv = np.asarray(getattr(env, "sigmas", gcfg_ep.sigmas), float).reshape(-1)
        beta_for_mv = np.asarray(getattr(env, "beta", beta), float).reshape(-1)
        m_star = mv_center_qp(env.Cov, sigmas_for_mv, beta_for_mv, target_ann_eff,
                              allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver)
        if k == 0:
            ms = m_star
            print(f"[mv dbg] sum={ms.sum():.6f} "
                f"nnz(>1e-8)={int((ms>1e-8).sum())}/{ms.size} "
                f"min/med/max={ms.min():.3e}/{np.median(ms):.3e}/{ms.max():.3e}")
            print(
                "[Cov dbg]",
                "sigma=", np.round(env.sigmas, 3),
                "Cov_diag=", np.round(np.diag(env.Cov), 4),
                "eigmin=", np.linalg.eigvalsh(env.Cov)[0]
                )

        ep_obs, ep_m, ep_s, ep_lp, ep_val, ep_rew, ep_done = [], [], [], [], [], [], []
        b_min_list, b_mean_list, b_max_list, b_med_list = [], [], [], []
        b_zero_frac_list = []
        turnover_sum = 0.0
        trade_count = 0
        step_count = 0
        
        # --- per-episode diagnostics (accumulate per step) ---
        minm_zero_list = []
        minm_min_list, minm_med_list, minm_max_list = [], [], []

        delta_zero_list = []
        delta_min_list, delta_med_list, delta_max_list = [], [], []

        s_zero_list = []
        s_min_list, s_mean_list, s_max_list = [], [], []

        b_zero_list = []
        
        gap_topk_list = []
        sqrtDii_topk_list = []

        absdev_mean_list = []
        absdev_med_list  = []
        absdev_max_list  = []

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

                # --- s stats ---
                s_arr = np.asarray(s_np, float).reshape(-1)
                s_zero_list.append(float(np.mean(s_arr <= 1e-6)))
                s_min_list.append(float(s_arr.min()))
                s_mean_list.append(float(s_arr.mean()))
                s_max_list.append(float(s_arr.max()))


            m = m_star
            minm = np.minimum(m, 1.0 - m)
            # --- minm stats ---
            minm_arr = np.asarray(minm, float).reshape(-1)
            minm_zero_list.append(float(np.mean(minm_arr <= 1e-12)))
            minm_min_list.append(float(minm_arr.min()))
            minm_med_list.append(float(np.median(minm_arr)))
            minm_max_list.append(float(minm_arr.max()))

            if k == 0 and t == 0:
                ms = m_star
                near0 = int(np.sum(ms < 1e-8))
                near1 = int(np.sum(ms > 1-1e-8))
                print(f"[mv dbg] allow_cash={mv_allow_cash} target={target_ann_eff} "
                      f"m*: sum={ms.sum():.6f} min/med/max={ms.min():.3e}/{np.median(ms):.3e}/{ms.max():.3e} "
                      f"near0={near0} near1={near1}"
                      )
    
            if stage == 1:
                b = gcfg_ep.STAGE1_WIDTH_COEF * 0.95 * minm
            else:
                #lam_scalar = float(obs[N*5 + 0])
                #g = (max(0.0, 1.0 - lam_scalar) + 1e-8) ** gcfg.ALPHA
                #b = 0.95 * s_np * g * minm
                # HJB/QVI-inspired correlation-aware width prior (box approximation).
                # In this codebase, `lam` is the sell-proceeds wedge in (0,1]; proportional sell cost is kappa = 1-lam.
                per_dim = int(getattr(gcfg_ep, "PER_ASSET_DIM", 5))
                lam_scalar = float(obs[N*per_dim + 0])
                #lam_scalar = float(obs[N*5 + 0])
                delta = compute_delta_box(
                    w_star=m_star,
                    Cov=env.Cov,
                    gamma=float(getattr(gcfg_ep, "RISK_GAMMA", 0.0)),
                    lam=lam_scalar,
                    # Tune this if bands are too wide/narrow
                    scale=1.0,
                    clip=(0.0, 1.0),
                )
                # --- delta stats ---
                delta_arr = np.asarray(delta, float).reshape(-1)
                delta_zero_list.append(float(np.mean(delta_arr <= 1e-12)))
                delta_min_list.append(float(delta_arr.min()))
                delta_med_list.append(float(np.median(delta_arr)))
                delta_max_list.append(float(delta_arr.max()))

                b = 0.95 * s_np * delta * minm
                b_arr = np.asarray(b, float).reshape(-1)
                b_zero_list.append(float(np.mean(b_arr <= 1e-12)))

            # ===== DEBUG: 最初の episode & 最初の step だけ =====
            if stage == 2 and k == 0 and t == 0:
                print("[dbg] gamma =", float(getattr(gcfg_ep, "RISK_GAMMA", 0.0)))
                print("[dbg] delta(min/med/max) =",
                    float(delta.min()),
                    float(np.median(delta)),
                    float(delta.max()))
                print("[dbg] s_np(min/mean/max) =",
                    float(s_np.min()),
                    float(s_np.mean()),
                    float(s_np.max()))
                print("[dbg] b(min/med/max) =",
                    float(b.min()),
                    float(np.median(b)),
                    float(b.max()))
            # ===============================================

            A = clamp01_vec(m - b)
            B = np.maximum(A + 1e-6, m + b)

            Y_prev = env.S.sum() + env.C
            w_prev = env.S / (Y_prev + 1e-30)

            # ===== |w - m*| diagnostics =====
            abs_dev = np.abs(w_prev - m_star)

            # ===== outside-gap diagnostics =====
            gap_over  = float(np.max(w_prev - B))
            gap_under = float(np.max(A - w_prev))
            gap_any   = max(gap_over, gap_under)

            gap_over_list.append(max(0.0, gap_over))
            gap_under_list.append(max(0.0, gap_under))
            gap_any_list.append(max(0.0, gap_any))

            # ===== theoretical daily scale =====
            Dii = compute_Dii(m_star, env.Cov)
            sqrtDii = np.sqrt(Dii / 252.0)

            sqrtDii_mean_list.append(float(np.mean(sqrtDii)))
            sqrtDii_med_list.append(float(np.median(sqrtDii)))
            sqrtDii_max_list.append(float(np.max(sqrtDii)))


            obs, r, done, _ = env.step(A, B, use_trade_penalty=(stage == 2))

            # ===== step-after gap (post-trade) =====
            Y_post = env.S.sum() + env.C
            w_post = env.S / (Y_post + 1e-30)

            # ===== top-k diagnostics (pre-trade) =====
            idx_topk = np.argsort(m_star)[::-1][:TOPK]

            gap_topk = np.maximum(w_prev - B, A - w_prev)
            gap_topk = gap_topk[idx_topk]
            sqrtDii_topk = sqrtDii[idx_topk]


            w_trade = getattr(env, "w_post_trade", None)
            if w_trade is not None:
                to_trade = 0.5 * np.abs(w_trade - w_prev).sum()
            else:
                # fallback（案A）
                Y_next = env.S.sum() + env.C
                w_next = env.S / (Y_next + 1e-30)
                to_trade = 0.5 * np.abs(w_next - w_prev).sum()
            
            # ===== step-after gap (post-TRADE, pre-RETURN) =====
            if w_trade is not None:
                gap_over_post  = float(np.max(w_trade - B))
                gap_under_post = float(np.max(A - w_trade))
                gap_any_post   = max(gap_over_post, gap_under_post)
            else:
                gap_any_post = np.nan

            turnover_sum += to_trade
            step_count += 1
            if to_trade > 1e-4:
                trade_count += 1

            # b statistics (for this step)
            b_arr = np.asarray(b, float).reshape(-1)
            b_min_list.append(float(b_arr.min()))
            b_mean_list.append(float(b_arr.mean()))
            b_max_list.append(float(b_arr.max()))
            b_med_list.append(float(np.median(b_arr)))
            b_zero_frac_list.append(float(np.mean(b_arr <= 1e-12)))

            gap_post_list.append(max(0.0, gap_any_post))
            gap_topk_list.append(float(np.max(gap_topk)))
            sqrtDii_topk_list.append(float(np.mean(sqrtDii_topk)))
            absdev_mean_list.append(float(np.mean(abs_dev)))
            absdev_med_list.append(float(np.median(abs_dev)))
            absdev_max_list.append(float(np.max(abs_dev)))

            # --- episode-level summaries ---
            if len(minm_zero_list) > 0:
                out_minm = dict(
                    minm_zero=float(np.mean(minm_zero_list)),
                    minm_min=float(np.mean(minm_min_list)),
                    minm_med=float(np.mean(minm_med_list)),
                    minm_max=float(np.mean(minm_max_list)),
                )
            else:
                out_minm = None

            if len(delta_zero_list) > 0:
                out_delta = dict(
                    delta_zero=float(np.mean(delta_zero_list)),
                    delta_min=float(np.mean(delta_min_list)),
                    delta_med=float(np.mean(delta_med_list)),
                    delta_max=float(np.mean(delta_max_list)),
                )
            else:
                out_delta = None

            out_s = dict(
                s_zero=float(np.mean(s_zero_list)),
                s_min=float(np.mean(s_min_list)),
                s_mean=float(np.mean(s_mean_list)),
                s_max=float(np.mean(s_max_list)),
            )
            out_bzero = float(np.mean(b_zero_list)) if len(b_zero_list) > 0 else float("nan")


            ep_obs.append(o.squeeze(0).detach().cpu())
            ep_m.append(torch.tensor(m, dtype=torch.float32))
            ep_s.append(s_pre.squeeze(0).detach().cpu())
            ep_lp.append(logp_use.squeeze(0).detach().cpu())
            ep_val.append(v_t.squeeze(0).detach().cpu())
            ep_rew.append(float(r))
            ep_done.append(bool(done))
            if done:
                break

        if step_count > 0:
            diag["b_min"].append(float(np.mean(b_min_list)))
            diag["b_mean"].append(float(np.mean(b_mean_list)))
            diag["b_max"].append(float(np.mean(b_max_list)))
            diag["b_med"].append(float(np.mean(b_med_list)))
            diag["b_zero_frac"].append(float(np.mean(b_zero_frac_list)))
            diag["trade_rate"].append(float(trade_count / step_count))
            diag["avg_turnover"].append(float(turnover_sum / step_count))
            diag["turnover_sum"].append(float(turnover_sum))
            diag["steps"].append(int(step_count))
            diag["gap_over_mean"].append(float(np.mean(gap_over_list)))
            diag["gap_under_mean"].append(float(np.mean(gap_under_list)))
            diag["gap_any_mean"].append(float(np.mean(gap_any_list)))

            diag["sqrtDii_mean"].append(float(np.mean(sqrtDii_mean_list)))
            diag["sqrtDii_med"].append(float(np.mean(sqrtDii_med_list)))
            diag["sqrtDii_max"].append(float(np.mean(sqrtDii_max_list)))

            diag.setdefault("gap_post_mean", []).append(float(np.mean(gap_post_list)))
            diag.setdefault("gap_topk_mean", []).append(float(np.mean(gap_topk_list)))
            diag.setdefault("sqrtDii_topk_mean", []).append(float(np.mean(sqrtDii_topk_list)))
            diag.setdefault("absdev_mean", []).append(float(np.mean(absdev_mean_list)))
            diag.setdefault("absdev_med",  []).append(float(np.mean(absdev_med_list)))
            diag.setdefault("absdev_max",  []).append(float(np.mean(absdev_max_list)))

        if out_minm is not None:
            for k0, v0 in out_minm.items():
                diag_ep[k0].append(v0)
        if out_delta is not None:
            for k0, v0 in out_delta.items():
                diag_ep[k0].append(v0)
        for k0, v0 in out_s.items():
            diag_ep[k0].append(v0)
        diag_ep["b_zero"].append(out_bzero)


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

    avg_trade_rate = float(np.mean(diag["trade_rate"])) if len(diag["trade_rate"]) > 0 else 0.0
    avg_trunover = float(np.mean(diag["avg_turnover"])) if len(diag["avg_turnover"]) > 0 else 0.0
    print(f"average trade rate: {avg_trade_rate:.4f}, average turnover: {avg_trunover:.6f}")
    gapovermean = float(np.mean(diag["gap_over_mean"])) if len(diag["gap_over_mean"]) > 0 else 0.0
    gapundermean = float(np.mean(diag["gap_under_mean"])) if len(diag["gap_under_mean"]) > 0 else 0.0
    gapanymean = float(np.mean(diag["gap_any_mean"])) if len(diag["gap_any_mean"]) > 0 else 0.0
    print(f"gap over mean: {gapovermean:.6e}, gap under mean: {gapundermean:.6e}, gap any mean: {gapanymean:.6e}")
    sqrtDii_mean = float(np.mean(diag["sqrtDii_mean"])) if len(diag["sqrtDii_mean"]) > 0 else 0.0
    sqrtDii_med = float(np.mean(diag["sqrtDii_med"])) if len(diag["sqrtDii_med"]) > 0 else 0.0
    sqrtDii_max = float(np.mean(diag["sqrtDii_max"])) if len(diag["sqrtDii_max"]) > 0 else 0.0
    print(f"sqrtDii mean: {sqrtDii_mean:.6e}, median: {sqrtDii_med:.6e}, max: {sqrtDii_max:.6e}")

    gappostmean = float(np.mean(diag.get("gap_post_mean", [0.0]))) if len(diag.get("gap_post_mean", [])) > 0 else 0.0
    gaptopkmean = float(np.mean(diag.get("gap_topk_mean", [0.0]))) if len(diag.get("gap_topk_mean", [])) > 0 else 0.0
    sqrtDiitopkmean = float(np.mean(diag.get("sqrtDii_topk_mean", [0.0]))) if len(diag.get("sqrtDii_topk_mean", [])) > 0 else 0.0
    print(f"gap post mean: {gappostmean:.6e}, gap topk mean: {gaptopkmean:.6e}, sqrtDii topk mean: {sqrtDiitopkmean:.6e}")
    absdevmean = float(np.mean(diag.get("absdev_mean", [0.0]))) if len(diag.get("absdev_mean", [])) > 0 else 0.0
    absdevmed = float(np.mean(diag.get("absdev_med", [0.0]))) if len(diag.get("absdev_med", [])) > 0 else 0.0
    absdevmax = float(np.mean(diag.get("absdev_max", [0.0]))) if len(diag.get("absdev_max", [])) > 0 else 0.0
    print(f"absdev mean: {absdevmean:.6e}, median: {absdevmed:.6e}, max: {absdevmax:.6e}")

    obs = torch.cat(obs_buf).to(gcfg.device)
    m   = torch.cat(m_buf).to(gcfg.device)
    s   = torch.cat(s_buf).to(gcfg.device)
    logp= torch.cat(logp_buf).to(gcfg.device)
    adv = torch.cat(adv_buf).to(gcfg.device)
    ret = torch.cat(ret_buf).to(gcfg.device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    out = dict(
        obs=obs, m=m, s=s, logp=logp, adv=adv, ret=ret,
        rew_ep_mean=float(np.mean(rew_ep)), rew_ep_std=float(np.std(rew_ep)),
    )

    if len(diag["steps"]) > 0:
        out.update(dict(
            b_min_mean=float(np.mean(diag["b_min"])),
            b_mean_mean=float(np.mean(diag["b_mean"])),
            b_max_mean=float(np.mean(diag["b_max"])),
            b_med_mean=float(np.mean(diag["b_med"])),
            b_zero_frac_mean=float(np.mean(diag["b_zero_frac"])),
            trade_rate_mean=float(np.mean(diag["trade_rate"])),
            avg_turnover_mean=float(np.mean(diag["avg_turnover"])),
        ))
        # --- add zero-rate diagnostics (batch mean over episodes) ---
    if len(diag_ep["s_zero"]) > 0:
        out.update(dict(
            minm_zero_mean=float(np.mean(diag_ep["minm_zero"])) if len(diag_ep["minm_zero"]) else float("nan"),
            minm_min_mean=float(np.mean(diag_ep["minm_min"])) if len(diag_ep["minm_min"]) else float("nan"),
            minm_med_mean=float(np.mean(diag_ep["minm_med"])) if len(diag_ep["minm_med"]) else float("nan"),
            minm_max_mean=float(np.mean(diag_ep["minm_max"])) if len(diag_ep["minm_max"]) else float("nan"),

            delta_zero_mean=float(np.mean(diag_ep["delta_zero"])) if len(diag_ep["delta_zero"]) else float("nan"),
            delta_min_mean=float(np.mean(diag_ep["delta_min"])) if len(diag_ep["delta_min"]) else float("nan"),
            delta_med_mean=float(np.mean(diag_ep["delta_med"])) if len(diag_ep["delta_med"]) else float("nan"),
            delta_max_mean=float(np.mean(diag_ep["delta_max"])) if len(diag_ep["delta_max"]) else float("nan"),

            s_zero_mean=float(np.mean(diag_ep["s_zero"])),
            s_min_mean=float(np.mean(diag_ep["s_min"])),
            s_mean_mean=float(np.mean(diag_ep["s_mean"])),
            s_max_mean=float(np.mean(diag_ep["s_max"])),

            b_zero_mean=float(np.mean(diag_ep["b_zero"])) if len(diag_ep["b_zero"]) else float("nan"),
        ))
    if len(diag["gap_any_mean"]) > 0:
        out.update(dict(
            gap_over_mean=float(np.mean(diag["gap_over_mean"])),
            gap_under_mean=float(np.mean(diag["gap_under_mean"])),
            gap_any_mean=float(np.mean(diag["gap_any_mean"])),

            sqrtDii_mean=float(np.mean(diag["sqrtDii_mean"])),
            sqrtDii_med =float(np.mean(diag["sqrtDii_med"])),
            sqrtDii_max =float(np.mean(diag["sqrtDii_max"])),
        ))

    out.update(dict(
        gap_post_mean=float(np.mean(diag.get("gap_post_mean", [0.0]))),

        gap_topk_mean=float(np.mean(diag.get("gap_topk_mean", [0.0]))),
        sqrtDii_topk_mean=float(np.mean(diag.get("sqrtDii_topk_mean", [0.0]))),

        absdev_mean=float(np.mean(diag.get("absdev_mean", [0.0]))),
        absdev_med =float(np.mean(diag.get("absdev_med",  [0.0]))),
        absdev_max =float(np.mean(diag.get("absdev_max",  [0.0]))),
    ))

    return out


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
    mv_allow_cash: bool = False,
    mv_round_nd: int = 4,
    mv_solver: str = "OSQP",
    qp_solver: str = "OSQP",
    topk: int | None = None,
    env_ctor=None,
    center_policy: JointBandPolicy | None = None,
    center_mode: str = "policy",
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

    cm = str(center_mode).lower()
    if cm not in ("mv", "policy"):
        raise ValueError(f"center_mode must be 'mv' or 'policy', got {center_mode}")

    def _normalize_center(m: np.ndarray) -> np.ndarray:
        m = np.asarray(m, float).reshape(-1)
        m = np.maximum(0.0, m)
        s = float(m.sum())
        if not np.isfinite(s) or s <= 1e-12:
            warnings.warn("[rollout_levelB] bad m_star from policy; fallback to uniform.")
            return np.full(N, 1.0 / N, dtype=float)
        # centerは通常 simplex 上に置きたい（allow_cash=Trueでも中心を1に揃えるのが無難）
        return (m / s)

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
        #target_ann = (float(rng.choice(target_choices))
        #              if (target_choices is not None and len(target_choices) > 0)
        #              else float(getattr(gcfg_ep, "TARGET_RET_ANN", 0.06)))
        #target_ann = max(target_ann, 0.0)
        # If we want pure GMV (no return constraint), pass target=None.
        if _use_target_constraint(gcfg_ep) and (target_choices is not None and len(target_choices) > 0):
             target_ann = max(float(rng.choice(target_choices)), 0.0)
        elif _use_target_constraint(gcfg_ep):
             target_ann = max(float(getattr(gcfg_ep, "TARGET_RET_ANN", 0.06)), 0.0)
        else:
             target_ann = None

        if env_ctor is None:
            env = make_env(gcfg_ep, R_ep)
        else:
            env = _call_env_ctor(env_ctor, gcfg_ep=gcfg_ep, R_ep=R_ep, seed_ep=seed_ep)

        obs = env.reset(beta=beta, lam=lam, target_ret=target_ann, w0=None)

        # align to env-internal target
        target_ann_eff = (float(env.target_ret_ann) if target_ann is not None else None)

        # MV center (same as A2 rollout)
        #m_star = mv_center_qp(env.Cov, gcfg_ep.sigmas, beta, target_ann_eff,
        #                      allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver)
        #sigmas_for_mv = np.asarray(getattr(env, "sigmas", gcfg_ep.sigmas), float).reshape(-1)
        #beta_for_mv = np.asarray(getattr(env, "beta", beta), float).reshape(-1)
        #m_star = mv_center_qp(env.Cov, sigmas_for_mv, beta_for_mv, target_ann_eff,
        #                       allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver)

        # ------------------------------------------------------------
        # Center m_star (episode-fixed)
        # ------------------------------------------------------------
        cpol = center_policy if center_policy is not None else policy

        if cm == "mv":
            sigmas_for_mv = np.asarray(getattr(env, "sigmas", gcfg_ep.sigmas), float).reshape(-1)
            beta_for_mv   = np.asarray(getattr(env, "beta", beta), float).reshape(-1)
            m_star = mv_center_qp(
                env.Cov, sigmas_for_mv, beta_for_mv, target_ann_eff,
                allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver
            )
        else:
            # center_policy(A2) から m を1回だけ決定論で取得して固定
            o0 = torch.tensor(obs, dtype=torch.float32, device=gcfg_ep.device).unsqueeze(0)
            m_t, _, _, _, _, _ = cpol.sample(o0, deterministic=True)   # m_t: [1,N]（すでに射影済）
            m_star = m_t.squeeze(0).detach().cpu().numpy()


        # rotated-basis prior computed ONCE per episode (fast + stable)
        U, delta_z = compute_delta_rotated(
            m_star=m_star,
            Cov=env.Cov,
            gamma=float(getattr(gcfg_ep, "RISK_GAMMA", 0.0)),
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

            # ===== DEBUG: 最初の episode & 最初の step だけ =====
            if k == 0 and t == 0:
                print("[dbg] gamma =", float(getattr(gcfg_ep, "RISK_GAMMA", 0.0)))
                print("[dbg] delta_z(min/med/max) =",
                    float(delta_z.min()),
                    float(np.median(delta_z)),
                    float(delta_z.max()))
                print("[dbg] s_eff(min/mean/max) =",
                    float(s_eff.min()),
                    float(s_eff.mean()),
                    float(s_eff.max()))
                print("[dbg] b_z(min/med/max) =",
                    float(b_z.min()),
                    float(np.median(b_z)),
                    float(b_z.max()))
            # ===============================================

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
    mv_allow_cash: bool = False,
    mv_round_nd: int = 4,
    mv_solver: str = "OSQP",
    qp_solver: str = "OSQP",
    topk: int | None = None,
    env_ctor=None,
    center_mode: str = "policy",
    center_policy: Optional[JointBandPolicy] = None,
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
        #target_ann = (float(rng.choice(target_choices)) if (target_choices is not None and len(target_choices) > 0)
        #              else float(getattr(gcfg_ep, "TARGET_RET_ANN", 0.06)))
        #target_ann = max(target_ann, 0.0)
        if _use_target_constraint(gcfg_ep) and (target_choices is not None and len(target_choices) > 0):
             target_ann = max(float(rng.choice(target_choices)), 0.0)
        elif _use_target_constraint(gcfg_ep):
             target_ann = max(float(getattr(gcfg_ep, "TARGET_RET_ANN", 0.06)), 0.0)
        else:
             target_ann = None

        if env_ctor is None:
            env = make_env(gcfg_ep, R_ep)
        else:
            # NOTE: env_ctor is expected to close over regime_json / P / etc.
            # and return a ready-to-use env instance.
            env = _call_env_ctor(env_ctor, gcfg_ep=gcfg_ep, R_ep=R_ep, seed_ep=seed_ep)
        obs = env.reset(beta=beta, lam=lam, target_ret=target_ann, w0=None)
        target_ann_eff = (float(env.target_ret_ann) if target_ann is not None else None)

        sigmas_for_mv = np.asarray(getattr(env, "sigmas", gcfg_ep.sigmas), float).reshape(-1)
        beta_for_mv = np.asarray(getattr(env, "beta", beta),float).reshape(-1)
        m_star = mv_center_qp(
            env.Cov, sigmas_for_mv, beta_for_mv, target_ann_eff,
            allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver
        )

        #m_star = mv_center_qp(env.Cov, gcfg_ep.sigmas, beta, target_ann_eff,
        #                      allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver)

        cm = str(center_mode).lower().strip()
        if cm == "mv":
            sigmas_for_mv = np.asarray(getattr(env, "sigmas", gcfg_ep.sigmas), float).reshape(-1)
            beta_for_mv   = np.asarray(getattr(env, "beta", beta), float).reshape(-1)
            m_star = mv_center_qp(
                env.Cov, sigmas_for_mv, beta_for_mv, target_ann_eff,
                allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver
            )
        elif cm in ("policy", "a2", "center"):
            pol_c = center_policy if center_policy is not None else policy
            o0 = torch.tensor(obs, dtype=torch.float32, device=gcfg_ep.device).unsqueeze(0)
            with torch.no_grad():
                # deterministic=True で m だけ “固定センター” として使う
                m_t, _, _, _, _, _ = pol_c.sample(o0, deterministic=True)
            m_star = m_t.squeeze(0).detach().cpu().numpy()
            # 安全に正規化（cash可否に合わせる）
            m_star = np.clip(m_star, 0.0, 1.0)
            s = float(np.sum(m_star))
            if not np.isfinite(s) or s <= 1e-12:
                m_star = np.ones_like(m_star) / len(m_star)
            else:
                if mv_allow_cash:
                    if s > 1.0:
                        m_star = m_star / s
                else:
                    m_star = m_star / s
        else:
            raise ValueError(f"unknown center_method={center_mode}")

        ep_rew = 0.0
        ep_lret = 0.0

        # rotated-basis width prior
        U, delta_z = compute_delta_rotated(
            m_star=m_star,
            Cov=env.Cov,
            gamma=float(getattr(gcfg_ep, "RISK_GAMMA", 0.0)),
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

            # ===== DEBUG: 最初の episode & 最初の step だけ =====
            if k == 0 and t == 0:
                print("[dbg] gamma =", float(getattr(gcfg_ep, "RISK_GAMMA", 0.0)))
                print("[dbg] delta_z(min/med/max) =",
                    float(delta_z.min()),
                    float(np.median(delta_z)),
                    float(delta_z.max()))
                print("[dbg] s_eff(min/mean/max) =",
                    float(s_eff.min()),
                    float(s_eff.mean()),
                    float(s_eff.max()))
                print("[dbg] b_z(min/med/max) =",
                    float(b_z.min()),
                    float(np.median(b_z)),
                    float(b_z.max()))
            # ===============================================

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
