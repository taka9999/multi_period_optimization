"""
End-to-end script:
  load asset_returns.pkl (log returns)
  choose columns to match globalcfg.N_ASSETS
  load trained policies (A2 / B2)
  run historical evaluation (MV daily, MV monthly, RL A2, RL B2)
  compute annualized mean/vol + Sharpe
  plot arithmetic & geometric frontiers

Assumptions:
  - You have these available in your project:
      * globalcfg (or you can replace with your cfg object)
      * JointBandPolicy
      * compute_delta_box, apply_topk_s, clamp01_vec
      * HistoricalBandEnvMulti (from HistoricalEnv_LQ_v2.py in this chat)
  - asset_returns.pkl contains log returns with columns:
      ['LargeCap','MidCap','SmallCap','EAFE','EM','REIT','HighYield','Treasury',
       'Corporate','AggBond','Commodity','Gold','TBill']
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cvxpy as cp
import torch

from src.historical.env import HistoricalBandEnvMulti            # ←そのまま
from src.utils.rlopt_helpers import clamp01_vec
from src.ppo.agent import JointBandPolicy
from src.ppo.rollout import compute_delta_box, apply_topk_s, mv_center_qp
from src.regime_gbm.gbm_env import globalsetting

globalcfg = globalsetting()   # <- instantiate

# ============================================================
# Gamma/entropy + perf stats utilities
# ============================================================
def entropy_from_gamma(gamma: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    gamma: (T, K) rows sum to 1
    entropy_t = -sum_k gamma_tk log(gamma_tk)
    """
    g = np.asarray(gamma, float)
    g = np.clip(g, eps, 1.0)
    g = g / (g.sum(axis=1, keepdims=True) + eps)
    return -np.sum(g * np.log(g), axis=1)


def max_drawdown_from_wealth(W: np.ndarray) -> float:
    W = np.asarray(W, float).reshape(-1)
    if W.size < 2:
        return 0.0
    peak = np.maximum.accumulate(W)
    dd = W / (peak + 1e-12) - 1.0
    return float(np.min(dd))  # negative


def perf_stats_from_rsimple(rsimple: np.ndarray, *, dt: float, rf_ann: float = 0.0, mar: float = 0.0) -> Dict[str, float]:
    """
    rsimple: simple returns per step (daily)
    dt:      step size in years (e.g., 1/252)
    mar:     Minimum Acceptable Return (annual) for Sortino threshold.
             Here default 0.0 is fine (excess-return view).
    """
    r = np.asarray(rsimple, float).reshape(-1)
    if r.size == 0:
        return dict(ret=np.nan, ann_mean=np.nan, ann_vol=np.nan, sharpe=np.nan, sortino=np.nan, maxdd=np.nan)

    # wealth & total return
    W = wealth_from_rsimple(r, w0=100.0)
    tot_ret = float(W[-1] / W[0] - 1.0)

    # annualized mean/vol (arith)
    ann_mean = float(r.mean() / dt)
    ann_vol  = float(r.std(ddof=1) / math.sqrt(dt)) if r.size >= 2 else 0.0
    sharpe   = float((ann_mean - rf_ann) / (ann_vol + 1e-12))

    # sortino (downside deviation vs MAR)
    mar_step = float(mar * dt)
    downside = np.minimum(0.0, r - mar_step)
    ddv = float(np.sqrt(np.mean(downside**2)) / math.sqrt(dt))  # annualized downside vol
    sortino = float((ann_mean - rf_ann) / (ddv + 1e-12))

    # max drawdown
    maxdd = max_drawdown_from_wealth(W)

    return dict(
        ret=tot_ret,
        ann_mean=ann_mean,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        maxdd=maxdd,
    )


def _default_crash_windows() -> List[Dict[str, str]]:
    """
    Fixed representative crisis windows (US-centric; adjust freely).
    Dates are inclusive bounds, will be clipped to window date index.
    """
    return [
        dict(name="GFC",   start="2008-09-01", end="2009-03-31"),
        dict(name="COVID", start="2020-02-15", end="2020-04-30"),
        dict(name="Rates2022", start="2022-01-01", end="2022-10-31"),
    ]


def save_gamma_entropy_plots(
    *,
    dates: pd.DatetimeIndex,
    gamma: np.ndarray,
    out_png: str,
    out_npz: str,
    title: str = "HMM filtered regime prob (gamma) + entropy",
):
    """
    Saves:
      - out_npz: {dates, gamma, entropy}
      - out_png: time series plot
    """
    dates = pd.DatetimeIndex(dates)
    gamma = np.asarray(gamma, float)
    ent = entropy_from_gamma(gamma)

    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        dates=dates.astype("datetime64[ns]").values,
        gamma=gamma,
        entropy=ent,
    )

    K = gamma.shape[1]
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    for k in range(K):
        ax1.plot(dates, gamma[:, k], label=f"gamma{k}")
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend(ncol=min(K, 4), fontsize=9)

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(dates, ent)
    ax2.set_ylabel("entropy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=170)
    plt.close(fig)

# ============================================================
# 0) Regime related helpers
# ============================================================
def _stationary_dist(P: np.ndarray) -> np.ndarray:
    """Compute stationary dist of a Markov chain (fallback to uniform)."""
    P = np.asarray(P, float)
    K = P.shape[0]
    try:
        w, V = np.linalg.eig(P.T)
        j = int(np.argmin(np.abs(w - 1.0)))
        v = np.real(V[:, j])
        v = np.maximum(v, 0.0)
        s = float(v.sum())
        if np.isfinite(s) and s > 0:
            return v / s
    except Exception:
        pass
    return np.full(K, 1.0 / K, dtype=float)

def _regime_emission_params_from_regimes(regimes: list[dict], dt: float):
    """
    Convert regime dicts (beta/sigmas/R) -> Gaussian emission params for log-returns:
      x_t ~ N(m_k, Cov_k)
    Under your env: log return increment is
      (sig^2*beta - 0.5*sig^2)*dt + sig*sqrt(dt)*z
    If sigmas are annualized, Cov_k = Sigma_ann * dt.
    """
    mus = []
    covs = []
    for reg in regimes:
        beta = np.asarray(reg["beta"], float).reshape(-1)
        sig  = np.asarray(reg["sigmas"], float).reshape(-1)   # assume annualized
        R    = np.asarray(reg["R"], float)
        Sigma_ann = np.diag(sig) @ R @ np.diag(sig)
        Cov_step = Sigma_ann * float(dt)
        m_step = (sig**2 * beta - 0.5 * sig**2) * float(dt)
        mus.append(m_step)
        covs.append(Cov_step)
    return np.asarray(mus, float), np.asarray(covs, float)

def hmm_forward_filter_gaussian(X: np.ndarray, P: np.ndarray, pi: np.ndarray, mus: np.ndarray, covs: np.ndarray):
    """
    Forward filtering for Gaussian HMM.
    Returns gamma[t,k] = P(z_t=k | x_1:t).
    X: [T,N], mus:[K,N], covs:[K,N,N], P:[K,K], pi:[K]
    """
    X = np.asarray(X, float)
    P = np.asarray(P, float)
    pi = np.asarray(pi, float).reshape(-1)
    K = P.shape[0]
    T = X.shape[0]
    N = X.shape[1]

    # precompute log-likelihoods
    logB = np.zeros((T, K), float)
    const = -0.5 * N * np.log(2.0 * np.pi)
    for k in range(K):
        C = covs[k]
        # stabilize
        C = 0.5 * (C + C.T)
        C = C + 1e-9 * np.eye(N)
        L = np.linalg.cholesky(C)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        invC = np.linalg.inv(C)
        d = X - mus[k]
        quad = np.einsum("ti,ij,tj->t", d, invC, d)
        logB[:, k] = const - 0.5 * (logdet + quad)

    # forward in log-space
    logP = np.log(np.clip(P, 1e-300, None))
    loga = np.log(np.clip(pi, 1e-300, None)) + logB[0]
    # normalize
    m = np.max(loga)
    loga = loga - (m + np.log(np.sum(np.exp(loga - m))))
    gamma = np.zeros((T, K), float)
    gamma[0] = np.exp(loga)

    for t in range(1, T):
        # log alpha_t(j) = logB(t,j) + logsum_i( alpha_{t-1}(i) * P(i,j) )
        prev = loga.reshape(K, 1) + logP  # [K,K]
        m = np.max(prev, axis=0)
        lse = m + np.log(np.sum(np.exp(prev - m), axis=0))
        loga = logB[t] + lse
        m2 = np.max(loga)
        loga = loga - (m2 + np.log(np.sum(np.exp(loga - m2))))
        gamma[t] = np.exp(loga)

    return gamma

def augment_obs_with_gamma(obs: np.ndarray, gamma_t: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, float).reshape(-1)
    g = np.asarray(gamma_t, float).reshape(-1)
    s = float(g.sum())
    if np.isfinite(s) and s > 0:
        g = g / s
    return np.concatenate([obs, g], axis=0)

# ============================================================
# 1) Data loading + column mapping to globalcfg.N_ASSETS
# ============================================================
def choose_asset_columns(df: pd.DataFrame, N: int, prefer: list[str] | None = None) -> list[str]:
    """
    Choose N columns from df to match your N_ASSETS.

    Strategy:
      - if prefer is provided and valid, use it
      - else use a sensible default basket in a fixed order
      - fill remaining from available columns
    """
    available = [c for c in df.columns if c in df.columns]

    if prefer is not None:
        prefer = [c for c in prefer if c in df.columns]
        if len(prefer) >= N:
            return prefer[:N]

    # Default "core" order (common multi-asset set)
    default_order = [
        "LargeCap",   # US equity
        "EAFE",       # DM ex-US equity
        "EM",         # EM equity
        "HighYield",  # credit
        "Treasury",   # rates
        "Commodity",
        "Gold",
        "REIT",
        "AggBond",
        "Corporate",
        "SmallCap",
        "MidCap",
        "TBill",
    ]
    cols = [c for c in default_order if c in df.columns]

    # Fill from any remaining columns if needed
    if len(cols) < N:
        rest = [c for c in available if c not in cols]
        cols = cols + rest

    if len(cols) < N:
        raise ValueError(f"Not enough columns in returns data: need {N}, have {len(cols)}.")

    return cols[:N]


def load_historical_log_returns(pkl_path: str, N: int, prefer_cols: list[str] | None = None):
    """
    Returns:
      df_sel: DataFrame (T, N) log returns, NaNs dropped
      Rlog:   np.ndarray (T, N) log returns
      cols:   selected column names
    """
    df = pd.read_pickle(pkl_path)
    cols = choose_asset_columns(df, N, prefer=prefer_cols)

    df_sel = df[cols].copy()
    df_sel = df_sel.dropna(axis=0, how="any")  # critical: ensure contiguous window

    Rlog = df_sel.to_numpy(dtype=float)
    return df_sel, Rlog, cols

def apply_eval_date_range(df_sel: pd.DataFrame, eval_start: str | None, eval_end: str | None) -> pd.DataFrame:
    """
    Robust slice by date range.
    - Forces index to DatetimeIndex (handles datetime.date index)
    - Allows eval_start/end to be non-trading days (uses searchsorted)
    """
    if df_sel is None or len(df_sel) == 0:
        return df_sel

    out = df_sel.copy()

    # 1) normalize index -> DatetimeIndex, sorted
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    if eval_start is None and eval_end is None:
        return out

    # 2) searchsorted slice so missing dates are OK
    i0 = 0
    i1 = len(out)
    if eval_start is not None:
        ts0 = pd.Timestamp(eval_start)
        i0 = int(out.index.searchsorted(ts0, side="left"))
    if eval_end is not None:
        ts1 = pd.Timestamp(eval_end)
        i1 = int(out.index.searchsorted(ts1, side="right"))

    out = out.iloc[i0:i1]
    return out
# ============================================================
# 2) Annualization + Sharpe
# ============================================================
def ann_arith_mean_vol_from_rsimple(rsimple, dt=1 / 252):
    r = np.asarray(rsimple, float)
    ann_mean = r.mean() / dt
    ann_vol = r.std(ddof=1) / np.sqrt(dt)
    return float(ann_mean), float(ann_vol)


def ann_geom_mean_vol_from_rsimple(rsimple, dt=1 / 252):
    r = np.asarray(rsimple, float)
    logret = np.log1p(r)
    ann_mean = logret.mean() / dt
    ann_vol = logret.std(ddof=1) / np.sqrt(dt)
    return float(ann_mean), float(ann_vol)


def sharpe_from_ann(ann_mean, ann_vol, rf_ann=0.0):
    return (float(ann_mean) - float(rf_ann)) / (float(ann_vol) + 1e-12)

def wealth_from_rsimple(rsimple, w0=100.0):
    rs = np.asarray(rsimple, float).reshape(-1)
    W = np.empty(rs.size + 1, float)
    W[0] = float(w0)
    for t in range(rs.size):
        W[t + 1] = W[t] * (1.0 + rs[t])
    return W

def safe_log_wealth(W, eps=1e-12):
    W = np.asarray(W, float)
    return np.log(np.maximum(eps, W))

# ============================================================
# 3) MV QP solver
# ============================================================
def mv_weights_target_return(
    Cov, mu_eff, target_ann, allow_cash=True, solver="OSQP", infeasible_policy="skip"
):
    Cov = np.asarray(Cov, float)
    if target_ann is None:
        n = Cov.shape[0]
        mu_eff = None
    else:
        mu_eff = np.asarray(mu_eff, float).reshape(-1)
        n = len(mu_eff)
        mu_max = float(mu_eff.max())

    if target_ann is not None and target_ann > mu_max + 1e-12:
        if infeasible_policy == "skip":
            return None
        ww = np.zeros(n)
        ww[int(np.argmax(mu_eff))] = 1.0
        return ww

    w = cp.Variable(n)
    obj = cp.Minimize(cp.quad_form(w, Cov))
    cons = [w >= 0]
    if target_ann is not None:
        cons += [mu_eff @ w >= float(target_ann)]
    #cons = [w >= 0, mu_eff @ w >= float(target_ann)]
    cons += [cp.sum(w) <= 1.0] if allow_cash else [cp.sum(w) == 1.0]
    prob = cp.Problem(obj, cons)

    try:
        prob.solve(solver=getattr(cp, solver), verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if w.value is None:
        if infeasible_policy == "skip":
            return None
        ww = np.zeros(n)
        ww[int(np.argmax(mu_eff))] = 1.0
        return ww

    ww = np.maximum(0.0, np.array(w.value).reshape(-1))
    s = float(ww.sum())
    if not allow_cash:
        ww /= (s + 1e-12)
    else:
        if s > 1.0:
            ww /= (s + 1e-12)
    return ww


def estimate_mu_cov_ann_from_log_window(window_log: np.ndarray, dt: float):
    """
    window_log: (T,N) log returns
    Returns annualized (mu_ann, Cov_ann) in simple-return units.
    """
    Rsim = np.expm1(window_log)  # simple returns
    mu_daily = Rsim.mean(axis=0)
    Cov_daily = np.cov(Rsim, rowvar=False, ddof=1)
    mu_ann = mu_daily / dt
    Cov_ann = Cov_daily / dt
    return mu_ann, Cov_ann


# ============================================================
# 4) Episode runners (Historical)
# ============================================================
def run_episode_MV_daily_frictionless_hist(
    cfg,
    R_corr,
    returns_log,
    target_ann,
    *,
    start_idx: int,
    T_days: int,
    seed: int = 2025,
    mv_solver: str = "OSQP",
    infeasible_policy: str = "skip",
):
    cfg.seed = int(seed)

    env = HistoricalBandEnvMulti(
        cfg=cfg,
        R=R_corr,
        returns_log=returns_log,
        start_idx=int(start_idx),
        T_days=int(T_days),
        returns_are_excess=False,  # mode B: rf subtraction inside env
    )
    N = cfg.N_ASSETS
    w0 = np.full(N, 1.0 / N) * 0.8
    _ = env.reset(beta=np.zeros(N), lam=1.0, target_ret=None, w0=w0)

    window = np.asarray(returns_log[start_idx : start_idx + T_days], float)
    #target_ann_eff = float(env.target_ret_ann)

    mu_ann, Cov_ann = estimate_mu_cov_ann_from_log_window(window, dt=float(cfg.dt_day))
    w_star = mv_weights_target_return(
        Cov_ann, mu_ann, target_ann = None, allow_cash=True, solver=mv_solver, infeasible_policy=infeasible_policy
    )
    if w_star is None:
        return None

    rs = []
    for _t in range(env.T):
        obs, r_step, done, r_simple = env.step(w_star, w_star, use_trade_penalty=False)
        rs.append(float(r_simple))
        if done:
            break
    return np.array(rs, float)


def run_episode_MV_monthly_cost_hist(
    cfg,
    R_corr,
    returns_log,
    lam_cost,
    target_ann,
    *,
    start_idx: int,
    T_days: int,
    rebalance_every: int = 21,
    seed: int = 2025,
    mv_solver: str = "OSQP",
    infeasible_policy: str = "skip",
):
    cfg.seed = int(seed)

    env = HistoricalBandEnvMulti(
        cfg=cfg,
        R=R_corr,
        returns_log=returns_log,
        start_idx=int(start_idx),
        T_days=int(T_days),
        returns_are_excess=False,
    )
    N = cfg.N_ASSETS
    w0 = np.full(N, 1.0 / N) * 0.8
    _ = env.reset(beta=np.zeros(N), lam=lam_cost, target_ret=None, w0=w0)

    window = np.asarray(returns_log[start_idx : start_idx + T_days], float)
    #target_ann_eff = float(env.target_ret_ann)

    mu_ann, Cov_ann = estimate_mu_cov_ann_from_log_window(window, dt=float(cfg.dt_day))
    w_star = mv_weights_target_return(
        Cov_ann, mu_ann, target_ann = None, allow_cash=True, solver=mv_solver, infeasible_policy=infeasible_policy
    )
    if w_star is None:
        return None

    rs = []
    for t in range(env.T):
        if (t % rebalance_every) == 0:
            A = w_star
            B = w_star
        else:
            A = np.zeros(N)
            B = np.ones(N)
        obs, r_step, done, r_simple = env.step(A, B, use_trade_penalty=True)
        rs.append(float(r_simple))
        if done:
            break
    return np.array(rs, float)


def run_episode_RL_band_A2_hist(
    cfg,
    R_corr,
    returns_log,
    policy,
    lam_cost,
    target_ann,
    *,
    start_idx: int,
    T_days: int,
    seed: int = 2025,
    device: str = "cpu",
    mv_solver: str = "OSQP",
    infeasible_policy: str = "skip",
    force_s_one: bool = False,
    gamma_seq: np.ndarray | None = None,
):
    cfg.seed = int(seed)

    env = HistoricalBandEnvMulti(
        cfg=cfg,
        R=R_corr,
        returns_log=returns_log,
        start_idx=int(start_idx),
        T_days=int(T_days),
        returns_are_excess=False,
    )
    N = cfg.N_ASSETS
    w0 = np.full(N, 1.0 / N) * 0.8
    obs = env.reset(beta=np.zeros(N), lam=lam_cost, target_ret=None, w0=w0)

    window = np.asarray(returns_log[start_idx : start_idx + T_days], float)
    #target_ann_eff = float(env.target_ret_ann)

    mu_ann, Cov_ann = estimate_mu_cov_ann_from_log_window(window, dt=float(cfg.dt_day))
    m_star = mv_weights_target_return(
        Cov_ann, mu_ann, target_ann = None, allow_cash=True, solver=mv_solver, infeasible_policy=infeasible_policy
    )
    if m_star is None:
        return None

    rs = []
    for _t in range(env.T):
        o = np.array(obs, dtype=np.float32)
        if gamma_seq is not None:
            o = augment_obs_with_gamma(o, gamma_seq[_t])
        with torch.no_grad():
            if hasattr(policy, "sample_s_only"):
                s_t, _, _ = policy.sample_s_only(torch.tensor(o, device=device).unsqueeze(0))
                s = s_t.squeeze(0).detach().cpu().numpy()
            else:
                _, s_t, _, _, _ = policy.sample_stage2(torch.tensor(o, device=device).unsqueeze(0))
                s = s_t.squeeze(0).detach().cpu().numpy()

        if force_s_one:
            s = np.ones_like(s, dtype=float)

        m = m_star
        minm = np.minimum(m, 1.0 - m)
        lam_scalar = float(env.lam.mean()) if isinstance(env.lam, np.ndarray) else float(env.lam)

        delta = compute_delta_box(
            w_star=m_star,
            Cov=Cov_ann,  # keep consistent with historical center
            gamma=float(getattr(cfg, "RISK_GAMMA", 0.0)),
            lam=lam_scalar,
            scale=1.0,
            clip=(0.0, 1.0),
        )

        b = 0.95 * s * delta * minm
        A = clamp01_vec(m - b)
        B = np.maximum(A + 1e-6, m + b)

        obs, r_step, done, r_simple = env.step(A, B, use_trade_penalty=True)
        rs.append(float(r_simple))
        if done:
            break

    return np.array(rs, float)


def compute_levelB_basis_and_width(m_star, Cov, gamma, lam_scalar, eps=1e-12, scale=1.0):
    m_star = np.asarray(m_star, float).reshape(-1)
    Cov = np.asarray(Cov, float)
    kappa = max(0.0, 1.0 - float(lam_scalar))

    Q = np.diag(m_star) - np.outer(m_star, m_star)
    a = Q @ Cov @ Q
    a = 0.5 * (a + a.T)

    eigvals, U = np.linalg.eigh(a)
    eigvals = np.maximum(eigvals, 0.0)

    Cov_rot = U.T @ Cov @ U
    Gamma_kk = np.maximum(eps, float(gamma) * np.maximum(eps, np.diag(Cov_rot)))

    delta_z = scale * np.power((kappa * eigvals) / Gamma_kk, 1.0 / 3.0)
    delta_z = np.maximum(delta_z, 0.0)
    return U, delta_z


def run_episode_RL_band_B2_hist(
    cfg,
    R_corr,
    returns_log,
    policy,
    lam_cost,
    target_ann,
    *,
    start_idx: int,
    T_days: int,
    seed: int = 2025,
    device: str = "cpu",
    mv_solver: str = "OSQP",
    qp_solver: str = "OSQP",
    infeasible_policy: str = "skip",
    mv_allow_cash: bool = True,
    force_s_one: bool = False,
    topk: int | None = None,
    gamma_seq: np.ndarray | None = None,
):
    cfg.seed = int(seed)

    env = HistoricalBandEnvMulti(
        cfg=cfg,
        R=R_corr,
        returns_log=returns_log,
        start_idx=int(start_idx),
        T_days=int(T_days),
        returns_are_excess=False,
    )
    N = cfg.N_ASSETS
    w0 = np.full(N, 1.0 / N) * 0.8
    obs = env.reset(beta=np.zeros(N), lam=lam_cost, target_ret=None, w0=w0)

    window = np.asarray(returns_log[start_idx : start_idx + T_days], float)
    #target_ann_eff = float(env.target_ret_ann)

    mu_ann, Cov_ann = estimate_mu_cov_ann_from_log_window(window, dt=float(cfg.dt_day))
    m_star = mv_weights_target_return(
        Cov_ann, mu_ann, target_ann=None, allow_cash=mv_allow_cash, solver=mv_solver, infeasible_policy=infeasible_policy
    )
    if m_star is None:
        return None

    lam_scalar = float(env.lam.mean()) if isinstance(env.lam, np.ndarray) else float(env.lam)
    U, delta_z = compute_levelB_basis_and_width(
        m_star=m_star,
        Cov=Cov_ann,
        gamma=float(getattr(cfg, "RISK_GAMMA", 0.0)),
        lam_scalar=lam_scalar,
        scale=1.0,
    )

    rs = []
    for _t in range(env.T):
        o = np.array(obs, dtype=np.float32)
        if gamma_seq is not None:
            o = augment_obs_with_gamma(o, gamma_seq[_t])
        if force_s_one:
            s = np.ones(N, dtype=float)
        else:
            with torch.no_grad():
                if hasattr(policy, "sample_s_only"):
                    s_t, _, _ = policy.sample_s_only(torch.tensor(o, device=device).unsqueeze(0))
                    s = s_t.squeeze(0).detach().cpu().numpy()
                else:
                    _, s_t, _, _, _ = policy.sample_stage2(torch.tensor(o, device=device).unsqueeze(0))
                    s = s_t.squeeze(0).detach().cpu().numpy()

        s_eff = apply_topk_s(s, topk=topk)
        b_z = 0.95 * s_eff * delta_z

        obs, r_step, done, r_simple = env.step_rotated_box(
            m=m_star,
            U=U,
            b_z=b_z,
            allow_cash=mv_allow_cash,
            solver=qp_solver,
            use_trade_penalty=True,
        )
        rs.append(float(r_simple))
        if done:
            break

    return np.array(rs, float)


# ============================================================
# 5) Frontier evaluation on historical windows
# ============================================================
def eval_frontier_historical(
    cfg,
    returns_log,
    targets,
    *,
    policy_A2,
    policy_B2,
    n_eps: int = 30,
    lam_cost: float = 0.99,
    rebalance_every: int = 21,
    T_days: int = 252 * 5,
    base_seed: int = 2025,
    mv_solver: str = "OSQP",
    qp_solver: str = "OSQP",
    infeasible_policy: str = "skip",
    gamma_all: np.ndarray | None = None,
):
    N = cfg.N_ASSETS
    # R_corr is unused for historical center (we use Cov_ann), but env signature requires it.
    R_corr = np.eye(N, dtype=float)

    strategies = ["MV_daily_frictionless", "MV_monthly_cost", "RL_band_A2", "RL_band_B2"]
    raw = {s: {t: [] for t in targets} for s in strategies}
    skipped = {s: {t: 0 for t in targets} for s in strategies}

    T_max_start = int(returns_log.shape[0] - T_days - 1)
    if T_max_start <= 0:
        raise ValueError(f"Not enough data: returns_log T={returns_log.shape[0]} < T_days={T_days}.")

    for j in range(n_eps):
        seed = base_seed + j
        rng = np.random.default_rng(seed)

        # random contiguous window start
        start_idx = int(rng.integers(0, T_max_start))
        gamma_seq = None
        if gamma_all is not None:
            gamma_seq = gamma_all[start_idx:start_idx + T_days]

        for t_ann in targets:
            # MV daily
            rs0 = run_episode_MV_daily_frictionless_hist(
                cfg, R_corr, returns_log, t_ann,
                start_idx=start_idx, T_days=T_days, seed=seed,
                mv_solver=mv_solver, infeasible_policy=infeasible_policy
            )
            if rs0 is None:
                skipped["MV_daily_frictionless"][t_ann] += 1
            else:
                ar = ann_arith_mean_vol_from_rsimple(rs0, dt=float(cfg.dt_day))
                ge = ann_geom_mean_vol_from_rsimple(rs0, dt=float(cfg.dt_day))
                raw["MV_daily_frictionless"][t_ann].append(
                    {"arith": ar, "geom": ge, "sh_arith": sharpe_from_ann(*ar), "sh_geom": sharpe_from_ann(*ge)}
                )

            # MV monthly
            rsm = run_episode_MV_monthly_cost_hist(
                cfg, R_corr, returns_log, lam_cost, t_ann,
                start_idx=start_idx, T_days=T_days, rebalance_every=rebalance_every, seed=seed,
                mv_solver=mv_solver, infeasible_policy=infeasible_policy
            )
            if rsm is None:
                skipped["MV_monthly_cost"][t_ann] += 1
            else:
                ar = ann_arith_mean_vol_from_rsimple(rsm, dt=float(cfg.dt_day))
                ge = ann_geom_mean_vol_from_rsimple(rsm, dt=float(cfg.dt_day))
                raw["MV_monthly_cost"][t_ann].append(
                    {"arith": ar, "geom": ge, "sh_arith": sharpe_from_ann(*ar), "sh_geom": sharpe_from_ann(*ge)}
                )

            # RL A2
            rsA2 = run_episode_RL_band_A2_hist(
                cfg, R_corr, returns_log, policy_A2, lam_cost, t_ann,
                start_idx=start_idx, T_days=T_days, seed=seed, device=cfg.device,
                mv_solver=mv_solver, infeasible_policy=infeasible_policy,
                gamma_seq=gamma_seq
            )
            if rsA2 is None:
                skipped["RL_band_A2"][t_ann] += 1
            else:
                ar = ann_arith_mean_vol_from_rsimple(rsA2, dt=float(cfg.dt_day))
                ge = ann_geom_mean_vol_from_rsimple(rsA2, dt=float(cfg.dt_day))
                raw["RL_band_A2"][t_ann].append(
                    {"arith": ar, "geom": ge, "sh_arith": sharpe_from_ann(*ar), "sh_geom": sharpe_from_ann(*ge)}
                )

            # RL B2
            rsB2 = run_episode_RL_band_B2_hist(
                cfg, R_corr, returns_log, policy_B2, lam_cost, t_ann,
                start_idx=start_idx, T_days=T_days, seed=seed, device=cfg.device,
                mv_solver=mv_solver, qp_solver=qp_solver, infeasible_policy=infeasible_policy,
                gamma_seq=gamma_seq
            )
            if rsB2 is None:
                skipped["RL_band_B2"][t_ann] += 1
            else:
                ar = ann_arith_mean_vol_from_rsimple(rsB2, dt=float(cfg.dt_day))
                ge = ann_geom_mean_vol_from_rsimple(rsB2, dt=float(cfg.dt_day))
                raw["RL_band_B2"][t_ann].append(
                    {"arith": ar, "geom": ge, "sh_arith": sharpe_from_ann(*ar), "sh_geom": sharpe_from_ann(*ge)}
                )

    # aggregate
    res_arith = {s: {} for s in strategies}
    res_geom = {s: {} for s in strategies}
    res_sh_arith = {s: {} for s in strategies}
    res_sh_geom = {s: {} for s in strategies}

    for s in strategies:
        for t in targets:
            ptsA = np.array([d["arith"] for d in raw[s][t]], float)  # (k,2)
            ptsG = np.array([d["geom"] for d in raw[s][t]], float)
            shA = np.array([d["sh_arith"] for d in raw[s][t]], float)
            shG = np.array([d["sh_geom"] for d in raw[s][t]], float)

            if ptsA.size == 0:
                res_arith[s][t] = (np.nan, np.nan)
                res_geom[s][t] = (np.nan, np.nan)
                res_sh_arith[s][t] = np.nan
                res_sh_geom[s][t] = np.nan
            else:
                res_arith[s][t] = (float(ptsA[:, 0].mean()), float(ptsA[:, 1].mean()))
                res_geom[s][t] = (float(ptsG[:, 0].mean()), float(ptsG[:, 1].mean()))
                res_sh_arith[s][t] = float(shA.mean())
                res_sh_geom[s][t] = float(shG.mean())

    info = {"skipped": skipped}
    return res_arith, res_geom, res_sh_arith, res_sh_geom, raw, info

def make_start_indices(T_total: int, T_days: int, n_windows: int, seed: int, start_idx_list=None):
    max_start = int(T_total - T_days - 1)
    if max_start <= 0:
        raise ValueError(f"Not enough data for chosen T_days. T_total={T_total}, T_days={T_days}")
    if start_idx_list is not None:
        out = []
        for s in start_idx_list:
            s = int(s)
            if s < 0 or s > max_start:
                raise ValueError(f"start_idx {s} out of range [0, {max_start}]")
            out.append(s)
        return out[:n_windows]
    rng = np.random.default_rng(int(seed))
    return [int(rng.integers(0, max_start)) for _ in range(int(n_windows))]

def plot_wealth_overlay_grid(
    df_sel: pd.DataFrame,
    returns_log: np.ndarray,
    cfg,
    *,
    policy_A2,
    policy_B2,
    target_ann: float,
    T_days: int,
    lam_cost: float,
    rebalance_every: int,
    n_windows: int = 4,
    seed: int = 2025,
    start_idx_list=None,
    log_scale: bool = True,
    mv_solver: str = "OSQP",
    qp_solver: str = "OSQP",
    out_png: str | None = None,
):
    starts = make_start_indices(
        T_total=returns_log.shape[0],
        T_days=T_days,
        n_windows=n_windows,
        seed=seed,
        start_idx_list=start_idx_list,
    )
    n = len(starts)
    ncols = 1 if n == 1 else 2
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 4.7 * nrows), squeeze=False)
    for k, start_idx in enumerate(starts):
        r = k // ncols
        c = k % ncols
        ax = axes[r][c]

        dates = pd.DatetimeIndex(df_sel.index[start_idx : start_idx + T_days])

        rs_mv_d = run_episode_MV_daily_frictionless_hist(
            cfg, np.eye(cfg.N_ASSETS), returns_log, target_ann,
            start_idx=start_idx, T_days=T_days, seed=int(seed) + k,
            mv_solver=mv_solver, infeasible_policy="skip",
        )
        rs_mv_m = run_episode_MV_monthly_cost_hist(
            cfg, np.eye(cfg.N_ASSETS), returns_log, lam_cost, target_ann,
            start_idx=start_idx, T_days=T_days, rebalance_every=rebalance_every, seed=int(seed) + k,
            mv_solver=mv_solver, infeasible_policy="skip",
        )
        rs_a2 = run_episode_RL_band_A2_hist(
            cfg, np.eye(cfg.N_ASSETS), returns_log, policy_A2, lam_cost, target_ann,
            start_idx=start_idx, T_days=T_days, seed=int(seed) + k, device=cfg.device,
            mv_solver=mv_solver, infeasible_policy="skip",
        )
        rs_b2 = run_episode_RL_band_B2_hist(
            cfg, np.eye(cfg.N_ASSETS), returns_log, policy_B2, lam_cost, target_ann,
            start_idx=start_idx, T_days=T_days, seed=int(seed) + k, device=cfg.device,
            mv_solver=mv_solver, qp_solver=qp_solver, infeasible_policy="skip",
        )

        curves = []
        if rs_mv_d is not None: curves.append(("MV daily (frictionless)", rs_mv_d))
        if rs_mv_m is not None: curves.append(("MV monthly (cost)", rs_mv_m))
        if rs_a2 is not None:   curves.append(("RL A2 (band)", rs_a2))
        if rs_b2 is not None:   curves.append(("RL B2 (rot-box)", rs_b2))

        if not curves:
            ax.set_title(f"start_idx={start_idx} (no feasible runs)")
            ax.axis("off")
            continue

        for label, rs in curves:
            W = wealth_from_rsimple(rs, 100.0)
            dates_r = dates[:len(rs)]
            dates_w = dates_r.insert(0, dates_r[0])
            y = safe_log_wealth(W) if log_scale else W
            ax.plot(dates_w, y, label=label)

        ax.grid(True, alpha=0.3)
        ax.set_title(f"Window {k+1}/{n} | start_idx={start_idx} | target={target_ann:.3f}")
        ax.set_ylabel("log(wealth)" if log_scale else "wealth (base=100)")
        ax.legend(fontsize=8)

    # disable unused
    for kk in range(n, nrows * ncols):
        rr = kk // ncols
        cc = kk % ncols
        axes[rr][cc].axis("off")

    fig.suptitle(
        f"Wealth Overlay | target={target_ann:.3f} | {'LOG' if log_scale else 'LEVEL'} | n_windows={n}",
        y=0.995,
    )
    plt.tight_layout()
    if out_png is not None:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=160)
    plt.show()

# ============================================================
# 6) Plotting
# ============================================================
def plot_frontier(res, targets, title):
    strategies = [
        ("MV_daily_frictionless", "o"),
        ("MV_monthly_cost", "s"),
        ("RL_band_A2", "^"),
        ("RL_band_B2", "X"),
    ]
    plt.figure(figsize=(8, 6))
    for s, mk in strategies:
        xs = [res[s][t][1] for t in targets]  # vol
        ys = [res[s][t][0] for t in targets]  # mean
        plt.plot(xs, ys, marker=mk, linestyle="-", label=s)
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Mean Excess Return")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_frontier_save(res, targets, title, out_png: str):
    strategies = [
        ("MV_daily_frictionless", "o"),
        ("MV_monthly_cost", "s"),
        ("RL_band_A2", "^"),
        ("RL_band_B2", "X"),
    ]
    plt.figure(figsize=(8, 6))
    for s, mk in strategies:
        xs = [res[s][t][1] for t in targets]  # vol
        ys = [res[s][t][0] for t in targets]  # mean
        plt.plot(xs, ys, marker=mk, linestyle="-", label=s)
    plt.xlabel("Annualized Volatility")
    plt.ylabel("Annualized Mean Excess Return")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()

# ============================================================
# 6.5) Wealth overlay with stats box + crash windows + gamma plots
# ============================================================
def _fmt_stats_box(stats: Dict[str, float]) -> str:
    """
    Compact multi-line text for matplotlib.
    """
    if stats is None or any(np.isnan(stats.get(k, np.nan)) for k in ["ann_mean", "ann_vol"]):
        return "N/A"
    return (
        f"TotRet:  {stats['ret']*100:6.2f}%\n"
        f"AnnRet:  {stats['ann_mean']*100:6.2f}%\n"
        f"AnnVol:  {stats['ann_vol']*100:6.2f}%\n"
        f"Sharpe:  {stats['sharpe']:6.2f}\n"
        f"Sortino: {stats['sortino']:6.2f}\n"
        f"MaxDD:   {stats['maxdd']*100:6.2f}%"
    )


def _slice_by_dates(
    dates: pd.DatetimeIndex,
    rsimple: np.ndarray,
    start: str,
    end: str,
) -> np.ndarray:
    """
    Slice rsimple using date bounds; handles missing dates via searchsorted.
    """
    dates = pd.DatetimeIndex(dates)
    r = np.asarray(rsimple, float).reshape(-1)
    if r.size == 0:
        return r
    # wealth has length r+1; but for stats we slice rsimple aligned with dates[0:len(r)]
    n = min(len(dates), len(r))
    dates2 = dates[:n]
    i0 = int(dates2.searchsorted(pd.Timestamp(start), side="left"))
    i1 = int(dates2.searchsorted(pd.Timestamp(end), side="right"))
    return r[i0:i1]


def plot_wealth_overlay_grid_with_stats(
    *,
    df_sel: pd.DataFrame,
    returns_log: np.ndarray,
    cfg,
    policy_A2,
    policy_B2,
    target_ann: float,
    lam_cost: float,
    rebalance_every: int,
    T_days: int,
    n_windows: int,
    base_seed: int,
    out_png: str,
    out_dir: str,
    log_scale: bool = True,
    crash_windows: Optional[List[Dict[str, str]]] = None,
    # gamma: pass precomputed (T_window, K) for the *whole eval slice* if you have it
    gamma_full: Optional[np.ndarray] = None,
    gamma_dates: Optional[pd.DatetimeIndex] = None,
):
    """
    Makes a grid of wealth overlays (n_windows windows), and:
      - stats box per window per strategy
      - highlight crash windows
      - output crash-window stats CSV for each strategy
    """
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if crash_windows is None:
        crash_windows = _default_crash_windows()

    # choose random contiguous windows
    rng = np.random.default_rng(int(base_seed))
    T_max_start = int(returns_log.shape[0] - T_days - 1)
    if T_max_start <= 0:
        raise ValueError("Not enough data for wealth windows.")
    starts = [int(rng.integers(0, T_max_start)) for _ in range(int(n_windows))]

    # prepare crash stats rows
    crash_rows: List[Dict[str, Any]] = []

    # subplot grid
    nW = int(n_windows)
    ncols = int(math.ceil(math.sqrt(nW)))
    nrows = int(math.ceil(nW / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5*ncols, 3.6*nrows), squeeze=False)

    for i, start_idx in enumerate(starts):
        ax = axes[i // ncols][i % ncols]
        # window dates
        win_df = df_sel.iloc[start_idx:start_idx + T_days]
        dates = pd.DatetimeIndex(win_df.index)
        # --- run strategies (rsimple) ---
        rs_mv_daily = run_episode_MV_daily_frictionless_hist(
            cfg, np.eye(cfg.N_ASSETS), returns_log, target_ann,
            start_idx=start_idx, T_days=T_days, seed=base_seed + i,
            mv_solver="OSQP", infeasible_policy="fallback",
        )
        rs_mv_monthly = run_episode_MV_monthly_cost_hist(
            cfg, np.eye(cfg.N_ASSETS), returns_log, lam_cost, target_ann,
            start_idx=start_idx, T_days=T_days, rebalance_every=rebalance_every, seed=base_seed + i,
            mv_solver="OSQP", infeasible_policy="fallback",
        )
        rs_a2 = run_episode_RL_band_A2_hist(
            cfg, np.eye(cfg.N_ASSETS), returns_log, policy_A2, lam_cost, target_ann,
            start_idx=start_idx, T_days=T_days, seed=base_seed + i,
            device=cfg.device, mv_solver="OSQP", infeasible_policy="fallback",
        )
        rs_b2 = run_episode_RL_band_B2_hist(
            cfg, np.eye(cfg.N_ASSETS), returns_log, policy_B2, lam_cost, target_ann,
            start_idx=start_idx, T_days=T_days, seed=base_seed + i,
            device=cfg.device, mv_solver="OSQP", qp_solver="OSQP", infeasible_policy="fallback",
        )

        series = [
            ("MV_daily", rs_mv_daily),
            ("MV_monthly", rs_mv_monthly),
            ("RL_A2", rs_a2),
            ("RL_B2", rs_b2),
        ]

        # plot wealth & stats box (aggregate for full window)
        y_max = None
        box_lines = []
        for name, rs in series:
            if rs is None:
                continue
            W = wealth_from_rsimple(rs, 100.0)
            y = safe_log_wealth(W) if log_scale else W
            ax.plot(dates.insert(0, dates[0]), y, label=name)  # align W length = T+1
            y_max = float(np.max(y)) if y_max is None else float(max(y_max, np.max(y)))
            st = perf_stats_from_rsimple(rs, dt=float(cfg.dt_day))
            box_lines.append(f"[{name}]\n{_fmt_stats_box(st)}")

            # crash-window stats
            for cw in crash_windows:
                rsub = _slice_by_dates(dates, rs, cw["start"], cw["end"])
                st_cw = perf_stats_from_rsimple(rsub, dt=float(cfg.dt_day))
                crash_rows.append(dict(
                    window=i,
                    start=str(dates[0].date()) if len(dates) else "",
                    end=str(dates[-1].date()) if len(dates) else "",
                    strategy=name,
                    crash_name=cw["name"],
                    crash_start=cw["start"],
                    crash_end=cw["end"],
                    **st_cw,
                ))
        # crash highlights
        for cw in crash_windows:
            s = pd.Timestamp(cw["start"])
            e = pd.Timestamp(cw["end"])
            # clip into [dates[0], dates[-1]]
            if len(dates) == 0:
                continue
            s2 = max(s, dates[0])
            e2 = min(e, dates[-1])
            if s2 <= e2:
                ax.axvspan(s2, e2, alpha=0.10)

        ax.set_title(f"window {i}  ({dates[0].date()} → {dates[-1].date()})")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("log(wealth)" if log_scale else "wealth (base=100)")
        ax.legend(fontsize=8, loc="upper left")

        # stats box on the right
        if box_lines:
            txt = "\n\n".join(box_lines)
            ax.text(
                1.01, 0.98, txt,
                transform=ax.transAxes,
                va="top", ha="left",
                fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )

    # hide empty axes
    for j in range(nW, nrows*ncols):
        axes[j // ncols][j % ncols].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close(fig)

    # crash stats CSV
    if crash_rows:
        df_c = pd.DataFrame(crash_rows)
        df_c.to_csv(Path(out_dir) / "crash_window_stats.csv", index=False)

def summaries_to_frames(resA, resG, shA, shG, targets):
    rowsA, rowsG = [], []
    for s in resA.keys():
        for t in targets:
            m, v = resA[s][t]
            rowsA.append(dict(strategy=s, target=float(t), mean=float(m), vol=float(v), sharpe=float(shA[s][t])))
            m2, v2 = resG[s][t]
            rowsG.append(dict(strategy=s, target=float(t), mean=float(m2), vol=float(v2), sharpe=float(shG[s][t])))
    return pd.DataFrame(rowsA), pd.DataFrame(rowsG)

def parse_crash_windows(s: str):
    """
    Parse crash windows string into list of dicts.

    Format:
      "NAME:YYYY-MM-DD:YYYY-MM-DD;NAME2:YYYY-MM-DD:YYYY-MM-DD"

    Example:
      "GFC:2008-09-01:2009-03-31;COVID:2020-02-15:2020-04-30"
    """
    if s is None or s.strip() == "":
        return None

    windows = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        try:
            name, start, end = part.split(":")
        except ValueError:
            raise ValueError(
                f"Invalid crash window spec '{part}'. "
                "Expected format NAME:YYYY-MM-DD:YYYY-MM-DD"
            )
        windows.append(
            dict(
                name=name,
                start=start,
                end=end,
            )
        )
    return windows


# ============================================================
# 7) MAIN
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--cols", type=str, required=True)
    ap.add_argument("--rf-col", type=str, default=None)
    ap.add_argument("--no-subtract-rf", action="store_true")
    ap.add_argument("--eval_start", type=str, default=None)
    ap.add_argument("--eval_end", type=str, default=None)
    ap.add_argument("--T_days", type=int, default=1260)
    ap.add_argument("--n_eps", type=int, default=30)
    ap.add_argument("--dt", type=float, default=1.0/252)

    # regime json for filtering gamma_t
    ap.add_argument("--regime_json", type=str, default=None,
                    help="Regime JSON containing {P, regimes:[{beta,sigmas,R},...]}. "
                         "If provided, forward-filter gamma_t and append to obs.")

    # policy/value paths
    ap.add_argument("--policy_A2", type=str, default=None)
    ap.add_argument("--policy_B2", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # --- gamma plots (historical HMM filtering outputs) ---
    ap.add_argument("--plot_gamma", action="store_true", help="Save gamma/entropy plots (requires regime_json usage in this script).")
    ap.add_argument("--gamma_prefix", type=str, default="gamma", help="prefix for gamma outputs (png/npz).")

    # --- wealth overlay extras ---
    ap.add_argument("--crash_windows", type=str, default="", help="Optional crash windows. Format: name:start:end;name:start:end ... (ISO dates)")

    ap.add_argument("--plot_wealth", action="store_true")
    ap.add_argument("--wealth_target", type=float, default=0.06)
    ap.add_argument("--wealth_windows", type=int, default=4)
    ap.add_argument("--wealth_log", action="store_true")

    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default="outputs/historical")


    args = ap.parse_args()

    df = pd.read_pickle(args.data)
    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    df_sel = df[cols].copy()
    if args.rf_col is not None and (not args.no_subtract_rf):
        rf = df[args.rf_col].astype(float)
        df_sel = df_sel.sub(rf, axis=0)    
    df_sel = apply_eval_date_range(df_sel, args.eval_start, args.eval_end)

    returns_log = df_sel.to_numpy(dtype=float)  # [T, N]

    # load regimes -> precompute gamma for full sample (cheap)
    gamma_all = None
    K = 0
    if args.regime_json is not None:
        with open(args.regime_json, "r") as f:
            data = json.load(f)
        regimes = data["regimes"]
        P = np.asarray(data["P"], float)
        mus, covs = _regime_emission_params_from_regimes(regimes, dt=float(args.dt))
        pi = _stationary_dist(P)
        gamma_all = hmm_forward_filter_gaussian(returns_log, P, pi, mus, covs)  # [T,K]
        K = gamma_all.shape[1]
        print(f"[HMM] computed gamma_all: T={gamma_all.shape[0]} K={K}")

    # ----- load policies (if provided) -----
    device = torch.device(args.device)
    N = len(cols)
    # IMPORTANT: if you use gamma, policy/value must be instantiated with global_dim=4+K
    global_dim = 4 + (K if gamma_all is not None else 0)
    policyA2 = None
    policyB2 = None
    if args.policy_A2 is not None:
        policyA2 = JointBandPolicy(N, d_model=128, nlayers=2, nhead=4, use_cash_softmax=True, global_dim=global_dim).to(device)
        policyA2.load_state_dict(torch.load(args.policy_A2, map_location=device))
        policyA2.eval()
    if args.policy_B2 is not None:
        policyB2 = JointBandPolicy(N, d_model=128, nlayers=2, nhead=4, use_cash_softmax=True, global_dim=global_dim).to(device)
        policyB2.load_state_dict(torch.load(args.policy_B2, map_location=device))
        policyB2.eval()
    
    # ============================================================
    # 8) Frontier evaluation (HISTORICAL)
    # ============================================================
    #targets = np.linspace(0.02, 0.10, 9)   # 2% ～ 10% 年率ターゲット
    targets = np.array([float(args.wealth_target)], dtype=float)
    lam_cost = 0.99
    rebalance_every = 21

    print("[Frontier] evaluating historical frontier ...")
    resA, resG, shA, shG, raw, info = eval_frontier_historical(
        globalcfg,
        returns_log,
        targets,
        policy_A2=policyA2,
        policy_B2=policyB2,
        n_eps=args.n_eps,
        lam_cost=lam_cost,
        rebalance_every=rebalance_every,
        T_days=args.T_days,
        base_seed=2025,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"hist_{ts}_lam{lam_cost:.3f}"
    outdir = Path(args.out_dir) / run_name
    outdir.mkdir(parents=True, exist_ok=True)

    #outdir = Path("outputs/historical")
    #outdir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 9) Save frontier plots
    # ============================================================
    #plot_frontier_save(resA,targets,title="Historical Frontier (Arithmetic)",out_png=str(outdir / "frontier_arithmetic.png"),)

    #plot_frontier_save(resG,targets,title="Historical Frontier (Geometric)",out_png=str(outdir / "frontier_geometric.png"),)

    # 9) Save realized mean-vol scatter (arith & geom)
    def plot_realized_scatter(raw, targets, title, out_png):
        strategies = ["MV_daily_frictionless", "MV_monthly_cost", "RL_band_A2", "RL_band_B2"]
        plt.figure(figsize=(8, 6))
        for s in strategies:
            pts = []
            for t in targets:
                pts += [d["arith"] for d in raw[s][t]]  # (mean, vol)
            if len(pts) == 0:
                continue
            P = np.array(pts, float)
            plt.scatter(P[:, 1], P[:, 0], s=18, alpha=0.45, label=s)  # x=vol, y=mean
        plt.xlabel("Annualized Volatility (realized)")
        plt.ylabel("Annualized Mean (realized)")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9)
        plt.tight_layout()
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=160)
        plt.close()

    plot_realized_scatter(raw, targets, "Realized Mean-Vol Scatter (Arithmetic)", str(outdir / "scatter_arithmetic.png"))

    # also save tables
    dfA, dfG = summaries_to_frames(resA, resG, shA, shG, targets)
    dfA.to_csv(outdir / "frontier_arithmetic.csv", index=False)
    dfG.to_csv(outdir / "frontier_geometric.csv", index=False)

    print("[Frontier] saved plots + csv")

    # ============================================================
    # 10) Wealth overlay with stats + crash windows
    # ============================================================
    print("[Wealth] plotting wealth overlay ...")

    if args.plot_wealth:
        plot_wealth_overlay_grid_with_stats(
            df_sel=df_sel,
            returns_log=returns_log,
            cfg=globalcfg,
            policy_A2=policyA2,
            policy_B2=policyB2,
            target_ann=args.wealth_target,
            lam_cost=0.99,
            rebalance_every=21,
            T_days=args.T_days,
            n_windows=args.wealth_windows,
            base_seed=2025,
            out_png="outputs/wealth_overlay.png",
            out_dir="outputs",
            log_scale=args.wealth_log,
            crash_windows=parse_crash_windows(args.crash_windows),
        )

    # ============================================================
    # 11) Gamma / entropy plots (optional)
    # ============================================================
    if args.plot_gamma and gamma_all is not None:
        save_gamma_entropy_plots(
            dates=df_sel.index,
            gamma=gamma_all,
            out_png=str(outdir / f"{args.gamma_prefix}.png"),
            out_npz=str(outdir / f"{args.gamma_prefix}.npz"),
            title="Filtered Regime Probabilities (Historical)",
        )
        print("[Gamma] saved gamma / entropy plots")

    print("=== DONE ===")


