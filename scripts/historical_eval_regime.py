"""
historical_eval_full.py

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

from src.historical import env
from src.historical.env import HistoricalBandEnvMulti
from src.utils.rlopt_helpers import clamp01_vec
from src.ppo.agent import JointBandPolicy
from src.ppo.rollout import compute_delta_box, compute_delta_rotated, apply_topk_s
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
    """
    min w^T Cov w
    s.t. mu_eff^T w >= target_ann, w>=0, sum(w)<=1 (cash allowed)

    infeasible_policy:
      - "skip": return None if target_ann > max(mu_eff)
      - "fallback": invest 100% in argmax(mu_eff)
    """
    mu_eff = np.asarray(mu_eff, float).reshape(-1)
    n = len(mu_eff)
    mu_max = float(mu_eff.max())

    if target_ann > mu_max + 1e-12:
        if infeasible_policy == "skip":
            return None
        ww = np.zeros(n)
        ww[int(np.argmax(mu_eff))] = 1.0
        return ww

    w = cp.Variable(n)
    obj = cp.Minimize(cp.quad_form(w, Cov))
    cons = [w >= 0, mu_eff @ w >= float(target_ann)]
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

def rolling_mu_cov(
    returns_log: np.ndarray,
    *,
    end_idx: int,
    lookback: int,
    dt: float,
):
    """
    Use past lookback window [end_idx-lookback, end_idx)
    """
    i0 = max(0, end_idx - lookback)
    window = returns_log[i0:end_idx]
    if window.shape[0] < 2:
        return None, None
    return estimate_mu_cov_ann_from_log_window(window, dt=dt)

def _burnin_slice_rs(rsimple: np.ndarray, burnin_steps: int) -> np.ndarray:
    """Return rsimple after dropping the first burnin_steps (safe for burnin_steps=0)."""
    r = np.asarray(rsimple, float).reshape(-1)
    b = int(max(0, burnin_steps))
    if b <= 0:
        return r
    if b >= r.size:
        return r[0:0]
    return r[b:]


def _burnin_slice_for_plot(
    dates: pd.DatetimeIndex,
    rsimple: np.ndarray,
    burnin_steps: int,
    *,
    w0: float = 100.0,
    renormalize: bool = True,
):
    """
    Slice (dates, wealth) to exclude burn-in from plots while keeping paths comparable.

    - rsimple length = T
    - wealth length  = T+1
    - dates length   >= T (we use the first T entries as step dates)
    Returns:
        dates_plot: length (T-b)
        W_plot:     length (T-b)+1
    """
    dates = pd.DatetimeIndex(dates)
    r = np.asarray(rsimple, float).reshape(-1)
    T = min(len(dates), len(r))
    dates_step = dates[:T]
    r = r[:T]
    b = int(max(0, burnin_steps))
    b = min(b, T)

    W_full = wealth_from_rsimple(r, w0=w0)  # length T+1
    W_plot = W_full[b:]                     # length (T-b)+1
    dates_plot = dates_step[b:]             # length (T-b)
    if renormalize and W_plot.size > 0:
        base = float(W_plot[0])
        if abs(base) > 1e-12:
            W_plot = W_plot / base * w0

    return dates_plot, W_plot

def _center_from_policy_deterministic(policy, o_np: np.ndarray, device: str, N: int) -> np.ndarray:
    """
    policy から deterministic center m を取得して、[0,1]^N かつ sum<=1 に正規化。
    JointBandPolicy.sample_stage2() を優先（m決定論）。
    """
    o_t = torch.tensor(o_np, device=device).unsqueeze(0)  # [1, obs_dim]
    with torch.no_grad():
        if hasattr(policy, "sample_stage2"):
            m_t, _, _, _, _ = policy.sample_stage2(o_t)    # agent.pyの仕様 :contentReference[oaicite:2]{index=2}
        else:
            # fallback: deterministic sample
            m_t, _, _, _, _, _ = policy.sample(o_t, deterministic=True)

    m = m_t.squeeze(0).detach().cpu().numpy().astype(float).reshape(-1)
    # sanitize (simplex/boxの安定化)
    m = np.clip(m, 0.0, 1.0)
    ssum = float(m.sum())
    if ssum > 1.0 + 1e-12:
        m = m / (ssum + 1e-30)
    return m

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
    _ = env.reset(beta=np.zeros(N), lam=1.0, target_ret=target_ann, w0=w0)

    window = np.asarray(returns_log[start_idx : start_idx + T_days], float)
    target_ann_eff = float(env.target_ret_ann)

    mu_ann, Cov_ann = estimate_mu_cov_ann_from_log_window(window, dt=float(cfg.dt_day))
    w_star = mv_weights_target_return(
        Cov_ann, mu_ann, target_ann_eff, allow_cash=False, solver=mv_solver, infeasible_policy=infeasible_policy
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
    _ = env.reset(beta=np.zeros(N), lam=lam_cost, target_ret=target_ann, w0=w0)
    
    target_ann_eff = float(env.target_ret_ann)
    #window = np.asarray(returns_log[start_idx : start_idx + T_days], float)
    #mu_ann, Cov_ann = estimate_mu_cov_ann_from_log_window(window, dt=float(cfg.dt_day))
    #w_star = mv_weights_target_return(
    #    Cov_ann, mu_ann, target_ann_eff, allow_cash=False, solver=mv_solver, infeasible_policy=infeasible_policy
    #)
    #if w_star is None:
    #    return None

    rs = []
    inside_cnt = outside_cnt = 0
    trigger_cnt = 0
    turnover_sum = sold_sum = cost_drag_sum = 0.0
    dist_sum = 0.0
    action_sum = 0.0
    width_all = []

    lookback = 252
    w_star = None
    mu_ann = Cov_ann = None
    
    for t in range(env.T):
        # --- recompute MV center on rebalance days (or keep previous if not enough history) ---
        if (t % rebalance_every) == 0:
            mu_ann, Cov_ann = rolling_mu_cov(
                returns_log,
                end_idx=start_idx + t,
                lookback=lookback,
                dt=float(cfg.dt_day),
            )
            if mu_ann is not None:
                w_new = mv_weights_target_return(
                    Cov_ann, mu_ann, target_ann_eff,
                    allow_cash=True,
                    solver=mv_solver,
                    infeasible_policy=infeasible_policy
                )
                if w_new is not None:
                    w_star = w_new

        # --- choose band: trade only on rebalance days if w_star exists; otherwise no-trade (wide band) ---
        if w_star is None:
            A = np.zeros(N)
            B = np.ones(N)
        elif (t % rebalance_every) == 0:
            A = w_star.copy()
            B = w_star.copy()
        else:
            A = np.zeros(N)
            B = np.ones(N)
        #obs, r_step, done, r_simple = env.step(A, B, use_trade_penalty=True)
        # logging: intended trade only on rebalance days
        w_pre, Y_prev = _get_weights_and_wealth(env)
        if w_pre is not None:
            inside = bool(np.all(w_pre >= A - 1e-12) and np.all(w_pre <= B + 1e-12))
            if inside: inside_cnt += 1
            else:      outside_cnt += 1
            dist = 0.0 if inside else _distance_to_band_axis(w_pre, A, B)
        else:
            dist = np.nan

        # action/band width (axis-aligned)
        width = float(np.mean(np.abs(B - A)))
        action_norm = float(np.linalg.norm(0.5 * (B - A)))
 
        obs, r, done, r_simple = env.step(A, B)
 
        if w_pre is not None and Y_prev is not None:
            lam_scalar = float(env.lam.mean()) if isinstance(env.lam, np.ndarray) else float(env.lam)
            st = _trade_stats_step(env, w_pre, Y_prev, lam_scalar)
            if st.get("did_trade", False):
                trigger_cnt += 1
            if np.isfinite(st["turnover"]):   turnover_sum += st["turnover"]
            if np.isfinite(st["sold_total"]): sold_sum     += st["sold_total"]
            if np.isfinite(st["cost_drag"]):  cost_drag_sum+= st["cost_drag"]
        if np.isfinite(dist): dist_sum += dist
        action_sum += action_norm
        width_all.append(width)
        rs.append(float(r_simple))
        if done:
            break
    stats = _finalize_stats(
        steps=len(rs),
        inside_cnt=inside_cnt, outside_cnt=outside_cnt,
        trigger_cnt=trigger_cnt,
        turnover_sum=turnover_sum, sold_sum=sold_sum, cost_drag_sum=cost_drag_sum,
        dist_sum=dist_sum, action_sum=action_sum,
        width_all=np.asarray(width_all),
        mstar_shift=0.0,
    )
    return dict(rs=np.asarray(rs, float), stats=stats)

def run_episode_EW_monthly_cost_hist(
    cfg,
    R_corr,
    returns_log,
    lam_cost,
    *,
    start_idx: int,
    T_days: int,
    rebalance_every: int = 21,
    seed: int = 2025,
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
    _ = env.reset(beta=np.zeros(N), lam=lam_cost, target_ret=0.0, w0=w0)

    w_eq = np.full(N, 1.0 / N, dtype=float)

    rs = []
    inside_cnt = outside_cnt = 0
    trigger_cnt = 0
    turnover_sum = sold_sum = cost_drag_sum = 0.0
    dist_sum = 0.0
    action_sum = 0.0
    width_all = []

    for t in range(env.T):
        # rebalance day: force trade to equal-weight
        if (t % rebalance_every) == 0:
            A = w_eq.copy()
            B = w_eq.copy()
        else:
            A = np.zeros(N)
            B = np.ones(N)

        w_pre, Y_prev = _get_weights_and_wealth(env)
        if w_pre is not None:
            inside = bool(np.all(w_pre >= A - 1e-12) and np.all(w_pre <= B + 1e-12))
            if inside: inside_cnt += 1
            else:      outside_cnt += 1
            dist = 0.0 if inside else _distance_to_band_axis(w_pre, A, B)
        else:
            dist = np.nan

        width = float(np.mean(np.abs(B - A)))
        action_norm = float(np.linalg.norm(0.5 * (B - A)))

        obs, r, done, r_simple = env.step(A, B)

        if w_pre is not None and Y_prev is not None:
            lam_scalar = float(env.lam.mean()) if isinstance(env.lam, np.ndarray) else float(env.lam)
            st = _trade_stats_step(env, w_pre, Y_prev, lam_scalar)
            if st.get("did_trade", False):
                trigger_cnt += 1
            if np.isfinite(st["turnover"]):   turnover_sum += st["turnover"]
            if np.isfinite(st["sold_total"]): sold_sum     += st["sold_total"]
            if np.isfinite(st["cost_drag"]):  cost_drag_sum+= st["cost_drag"]

        if np.isfinite(dist): dist_sum += dist
        action_sum += action_norm
        width_all.append(width)

        rs.append(float(r_simple))
        if done:
            break

    stats = _finalize_stats(
        steps=len(rs),
        inside_cnt=inside_cnt, outside_cnt=outside_cnt,
        trigger_cnt=trigger_cnt,
        turnover_sum=turnover_sum, sold_sum=sold_sum, cost_drag_sum=cost_drag_sum,
        dist_sum=dist_sum, action_sum=action_sum,
        width_all=np.asarray(width_all),
        mstar_shift=0.0,
    )
    return dict(rs=np.asarray(rs, float), stats=stats)


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
    obs = env.reset(beta=np.zeros(N), lam=lam_cost, target_ret=target_ann, w0=w0)

    target_ann_eff = float(env.target_ret_ann)

    #window = np.asarray(returns_log[start_idx : start_idx + T_days], float)
    #mu_ann, Cov_ann = estimate_mu_cov_ann_from_log_window(window, dt=float(cfg.dt_day))
    #m_star = mv_weights_target_return(
    #    Cov_ann, mu_ann, target_ann_eff, allow_cash=False, solver=mv_solver, infeasible_policy=infeasible_policy
    #)
    #if m_star is None:
    #    return None

    rs = []
    inside_cnt = outside_cnt = 0
    trigger_cnt = 0
    turnover_sum = sold_sum = cost_drag_sum = 0.0
    dist_sum = 0.0
    action_sum = 0.0
    width_all = []

    lookback = 252
    #m_star = None
    mu_ann = Cov_ann = None
    rebalance_every = 21

    for _t in range(env.T):
        b = np.zeros(N, dtype=float)
        if (_t % rebalance_every) == 0:
            mu_ann, Cov_ann = rolling_mu_cov(
                returns_log,
                end_idx=start_idx + _t,
                lookback=lookback,
                dt=float(cfg.dt_day),
            )
            #if mu_ann is not None:
                #m_new = mv_weights_target_return(
                #    Cov_ann, mu_ann, target_ann_eff,
                #    allow_cash=False,
                #    solver=mv_solver,
                #    infeasible_policy=infeasible_policy,
                #)
                #if m_new is not None:
                #    m_star = m_new
        #if (m_star is None) or (Cov_ann is None):
        #    A = np.zeros(N)
        #    B = np.ones(N)
        #    width = float(np.mean(np.abs(B - A)))
        #    action_norm = float(np.linalg.norm(b))  # = 0
        #    obs, r, done, r_simple = env.step(A, B)
        #    rs.append(float(r_simple))
        #    if done:
        #        break
        #    continue
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
        
        m = _center_from_policy_deterministic(policy, o, device=device, N=N)

        # If MV center is not available yet, fall back to no-trade wide band
        lam_scalar = float(env.lam.mean()) if isinstance(env.lam, np.ndarray) else float(env.lam)
        if Cov_ann is None:
            A = np.zeros(N); B = np.ones(N)
        #if (m_star is None) or (Cov_ann is None):
        #    A = np.zeros(N)
        #    B = np.ones(N)
        else:
            #m = np.asarray(m_star, float).reshape(-1)
            minm = np.minimum(m, 1.0 - m)

            delta = compute_delta_box(
                w_star=m,
                Cov=Cov_ann,
                gamma=float(getattr(cfg, "RISK_GAMMA", 0.0)),
                lam=lam_scalar,
                scale=1.0,
                clip=(0.0, 1.0),
            )

            b = 0.95 * s * delta * minm
            A = clamp01_vec(m - b)
            B = np.maximum(A + 1e-6, m + b)

        #obs, r_step, done, r_simple = env.step(A, B, use_trade_penalty=True)
        # inside check (axis box)
        w_pre, Y_prev = _get_weights_and_wealth(env)
        if w_pre is not None:
            inside = bool(np.all(w_pre >= A - 1e-12) and np.all(w_pre <= B + 1e-12))
            if inside: inside_cnt += 1
            else:      outside_cnt += 1
            dist = 0.0 if inside else _distance_to_band_axis(w_pre, A, B)
        else:
            dist = np.nan
        width = float(np.mean(np.abs(B - A)))
        action_norm = float(np.linalg.norm(b))

        obs, r, done, r_simple = env.step(A, B)
 
        if w_pre is not None and Y_prev is not None:
            st = _trade_stats_step(env, w_pre, Y_prev, lam_scalar)
            if st.get("did_trade", False):
                trigger_cnt += 1
            if np.isfinite(st["turnover"]):   turnover_sum += st["turnover"]
            if np.isfinite(st["sold_total"]): sold_sum     += st["sold_total"]
            if np.isfinite(st["cost_drag"]):  cost_drag_sum+= st["cost_drag"]
        if np.isfinite(dist): dist_sum += dist
        action_sum += action_norm
        width_all.append(width)
        rs.append(float(r_simple))
        if done:
            break

    stats = _finalize_stats(
        steps=len(rs),
        inside_cnt=inside_cnt, outside_cnt=outside_cnt,
        trigger_cnt=trigger_cnt,
        turnover_sum=turnover_sum, sold_sum=sold_sum, cost_drag_sum=cost_drag_sum,
        dist_sum=dist_sum, action_sum=action_sum,
        width_all=np.asarray(width_all),
        mstar_shift=0.0,
    )
    return dict(rs=np.asarray(rs, float), stats=stats)

def run_episode_RL_band_A2_hist_MPC(
    cfg,
    R_corr,
    returns_log,
    policy,
    lam_cost,
    target_ann,
    *,
    start_idx: int,
    T_total_days: int,
    seed: int = 2025,
    horizon_days: int = 252,
    rebalance_every: int = 21,
    device="cpu",
    mv_solver: str = "OSQP",
    infeasible_policy: str = "skip",
    force_s_one: bool = False,
    gamma_seq: np.ndarray | None = None,
):
    cfg.seed = int(seed)
    N = cfg.N_ASSETS
    t = 0
    w_current = np.full(N, 1.0 / N) * 0.8
    rs = []

    # ---- stats (cumulative over MPC horizon) ----
    inside_cnt = outside_cnt = 0
    trigger_cnt = 0
    turnover_sum = sold_sum = cost_drag_sum = 0.0
    dist_sum = 0.0
    action_sum = 0.0
    width_all = []
    #m_prev = None

    while t < T_total_days:
        # --- ① rolling windowで Cov 推定 ---
        window = returns_log[start_idx + t - horizon_days : start_idx + t]
        if window.shape[0] < horizon_days:
            break

        mu_ann, Cov_ann = estimate_mu_cov_ann_from_log_window(
            window, dt=float(cfg.dt_day)
        )

        #m_star = mv_weights_target_return(
        #    Cov_ann, mu_ann, target_ann,
        #    allow_cash=False,
        #    solver=mv_solver,
        #    infeasible_policy=infeasible_policy,
        #)
        #if m_star is None:
        #    break
        

        #if m_prev is not None:
        #    print("t=", t, "start=", start_idx+t, "dm1=", np.sum(np.abs(m_star-m_prev)))
        #else:
        #    print("t=", t, "start=", start_idx+t, "dm1=NA")
        #m_prev = m_star.copy()

        # --- ② env を 1年 horizon で reset ---
        env = HistoricalBandEnvMulti(
            cfg=cfg,
            R=R_corr,
            returns_log=returns_log,
            start_idx=start_idx + t,
            T_days=horizon_days,
            returns_are_excess=False,
        )
        obs = env.reset(
            beta=np.zeros(N),
            lam=lam_cost,
            target_ret=target_ann,
            w0=w_current,
        )

        lam_scalar = float(env.lam.mean()) if isinstance(env.lam, np.ndarray) else float(env.lam)

        # --- ③ 最初の 1か月だけ実行 ---
        for _ in range(min(rebalance_every, horizon_days)):
            o = np.array(obs, dtype=np.float32)

            with torch.no_grad():
                s = policy.sample_s_only(
                    torch.tensor(o, device=device).unsqueeze(0)
                )[0].squeeze(0).cpu().numpy()

            if force_s_one:
                s = np.ones_like(s, dtype=float)

            m = _center_from_policy_deterministic(policy, o, device=device, N=N)

            delta = compute_delta_box(
                w_star=m,
                Cov=Cov_ann,
                gamma=float(getattr(cfg, "RISK_GAMMA", 0.0)),
                lam=lam_scalar,
                scale=1.0,
                clip=(0.0, 1.0),
            )

            b = 0.95 * s * delta #* np.minimum(m, 1.0 - m)
            A = clamp01_vec(m - b)
            B = np.maximum(A + 1e-6, m + b)

            # ---- stats BEFORE step ----
            w_pre, Y_prev = _get_weights_and_wealth(env)
            if w_pre is not None:
                inside = bool(np.all(w_pre >= A - 1e-12) and np.all(w_pre <= B + 1e-12))
                if inside:
                    inside_cnt += 1
                else:
                    outside_cnt += 1
                dist = 0.0 if inside else _distance_to_band_axis(w_pre, A, B)
            else:
                dist = np.nan

            width = float(np.mean(np.abs(B - A)))
            action_norm = float(np.linalg.norm(b))

            # ---- step ----
            obs, _, done, r_simple = env.step(A, B)
            rs.append(float(r_simple))

            # ---- stats AFTER step ----
            if w_pre is not None and Y_prev is not None:
                st = _trade_stats_step(env, w_pre, Y_prev, lam_scalar)
                if st.get("did_trade", False):
                    trigger_cnt += 1
                if np.isfinite(st["turnover"]):   turnover_sum += st["turnover"]
                if np.isfinite(st["sold_total"]): sold_sum     += st["sold_total"]
                if np.isfinite(st["cost_drag"]):  cost_drag_sum+= st["cost_drag"]

            if np.isfinite(dist):
                dist_sum += dist
            action_sum += action_norm
            width_all.append(width)

            if done:
                break

        # --- ④ 次月へ（state 引き継ぎ） ---
        w_current, _ = _get_weights_and_wealth(env)
        t += rebalance_every

    stats = _finalize_stats(
        steps=len(rs),
        inside_cnt=inside_cnt,
        outside_cnt=outside_cnt,
        trigger_cnt=trigger_cnt,
        turnover_sum=turnover_sum,
        sold_sum=sold_sum,
        cost_drag_sum=cost_drag_sum,
        dist_sum=dist_sum,
        action_sum=action_sum,
        width_all=np.asarray(width_all),
        mstar_shift=0.0,  # MPCでは定義しにくいので0でOK
    )

    return dict(rs=np.asarray(rs, float), stats=stats)


def compute_levelB_basis_and_width(m_star, Cov, gamma, lam_scalar, eps=1e-12, scale=1.0):
    m_star = np.asarray(m_star, float).reshape(-1)
    Cov = np.asarray(Cov, float)
    kappa = max(0.0, 1.0 - float(lam_scalar))

    Q = np.diag(m_star) - np.outer(m_star, m_star)
    a = Q @ Cov @ Q
    a = 0.5 * (a + a.T)

    eigvals, U = np.linalg.eigh(a)
    eigvals = np.maximum(eigvals, 0.0)

    gamma_eff = 1 + float(gamma)
    Cov_rot = U.T @ Cov @ U
    Gamma_kk = np.maximum(eps, gamma_eff * np.maximum(eps, np.diag(Cov_rot)))

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
    mv_allow_cash: bool = False,
    force_s_one: bool = False,
    topk: int | None = None,
    gamma_seq: np.ndarray | None = None,
    center_mode: str = "policy",
    center_policy: JointBandPolicy | None = None,
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
    obs = env.reset(beta=np.zeros(N), lam=lam_cost, target_ret=target_ann, w0=w0)

    target_ann_eff = float(env.target_ret_ann)

    rs = []
    inside_cnt = outside_cnt = 0
    trigger_cnt = 0
    turnover_sum = sold_sum = cost_drag_sum = 0.0
    dist_sum = 0.0
    action_sum = 0.0
    width_all = []

    lookback = 252
    m_star = None
    mu_ann = Cov_ann = None
    rebalance_every = 21

    lam_scalar = float(env.lam.mean()) if isinstance(env.lam, np.ndarray) else float(env.lam)
    gamma_use = float(getattr(cfg, "RISK_GAMMA", getattr(env.cfg, "RISK_GAMMA", 0.0)))

    for _t in range(env.T):
        b = np.zeros(N, dtype=float)
        if (_t % rebalance_every) == 0:
            mu_ann, Cov_ann = rolling_mu_cov(
                returns_log,
                end_idx=start_idx + _t,
                lookback=lookback,
                dt=float(cfg.dt_day),
            )
            if mu_ann is not None:
                cm = str(center_mode).lower().strip()
                if cm == "mv":
                    sigmas_for_mv = np.asarray(getattr(env, "sigmas", cfg.sigmas), float).reshape(-1)
                    beta_for_mv   = np.asarray(getattr(env, "beta", cfg.beta), float).reshape(-1)
                    m_star = mv_weights_target_return(Cov_ann, mu_ann, target_ann_eff,allow_cash=False,solver=mv_solver,infeasible_policy=infeasible_policy,)
                elif cm in ("policy", "a2", "center"):
                    pol_c = center_policy if center_policy is not None else policy
                    o0 = torch.tensor(obs, dtype=torch.float32, device=cfg.device).unsqueeze(0)
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
                #m_new = mv_weights_target_return(
                #    Cov_ann, mu_ann, target_ann_eff,
                #    allow_cash=False,
                #    solver=mv_solver,
                #    infeasible_policy=infeasible_policy,
                #)
                #if m_new is not None:
                    #m_star = m_new
                U, delta_z = compute_delta_rotated(
                    m_star=m_star,
                    Cov=Cov_ann,
                    gamma=gamma_use,
                    lam=lam_scalar,
                    scale=1.0,
                    debug=False,
                    )
        if (m_star is None) or (Cov_ann is None):
            A = np.zeros(N)
            B = np.ones(N)
            width = float(np.mean(np.abs(B - A)))
            action_norm = float(np.linalg.norm(b))  # = 0
            obs, r, done, r_simple = env.step(A, B)
            rs.append(float(r_simple))
            if done:
                break
            continue
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
        # If rotated box params not available yet, fall back to no-trade wide band
        if (m_star is None) or (Cov_ann is None):
            A = np.zeros(N)
            B = np.ones(N)
            # logging/inside check uses A,B form for this step
            w_pre, Y_prev = _get_weights_and_wealth(env)
            if w_pre is not None:
                inside = bool(np.all(w_pre >= A - 1e-12) and np.all(w_pre <= B + 1e-12))
                if inside: inside_cnt += 1
                else:      outside_cnt += 1
                dist = 0.0 if inside else _distance_to_band_axis(w_pre, A, B)
            else:
                dist = np.nan
            width = float(np.mean(np.abs(B - A)))
            action_norm = 0.0
            obs, r, done, r_simple = env.step(A, B)
        else:
            b_z = 0.95 * s_eff * delta_z
            # inside check (rotated box)
            w_pre, Y_prev = _get_weights_and_wealth(env)
            if w_pre is not None:
                z = U.T @ (w_pre - m_star)
                inside = bool(np.all(np.abs(z) <= b_z + 1e-12))
                if inside: inside_cnt += 1
                else:      outside_cnt += 1
                dist = 0.0 if inside else _distance_to_band_rotated(w_pre, m_star, U, b_z)
            else:
                dist = np.nan
            width = float(np.mean(2.0 * b_z))
            action_norm = float(np.linalg.norm(b_z))
            obs, r, done, r_simple = env.step_rotated_box(m=m_star, U=U, b_z=b_z, allow_cash=mv_allow_cash)

        if w_pre is not None and Y_prev is not None:
            st = _trade_stats_step(env, w_pre, Y_prev, lam_scalar)
            if st.get("did_trade", False):
                trigger_cnt += 1
            if np.isfinite(st["turnover"]):   turnover_sum += st["turnover"]
            if np.isfinite(st["sold_total"]): sold_sum     += st["sold_total"]
            if np.isfinite(st["cost_drag"]):  cost_drag_sum+= st["cost_drag"]
        if np.isfinite(dist): dist_sum += dist
        action_sum += action_norm
        width_all.append(width)
        rs.append(float(r_simple))
        if done:
            break

    stats = _finalize_stats(
        steps=len(rs),
        inside_cnt=inside_cnt, outside_cnt=outside_cnt,
        trigger_cnt=trigger_cnt,
        turnover_sum=turnover_sum, sold_sum=sold_sum, cost_drag_sum=cost_drag_sum,
        dist_sum=dist_sum, action_sum=action_sum,
        width_all=np.asarray(width_all),
        mstar_shift=0.0,
    )
    return dict(rs=np.asarray(rs, float), stats=stats)


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
    outdir: str | None = None,
    burnin_steps: int = 0,
    center_mode: str = "policy",
    center_policy: JointBandPolicy | None = None,
):
    N = cfg.N_ASSETS
    # R_corr is unused for historical center (we use Cov_ann), but env signature requires it.
    R_corr = np.eye(N, dtype=float)

    #strategies = ["MV_daily_frictionless", "MV_monthly_cost", "RL_band_A2", "RL_band_B2"]
    #strategies = ["MV_daily_frictionless", "MV_monthly_cost", "EW_monthly_cost", "RL_band_A2", "RL_band_B2"]
    strategies = ["MV_monthly_cost", "EW_monthly_cost", "RL_band_A2", "RL_band_B2"]

    raw = {s: {t: [] for t in targets} for s in strategies}
    skipped = {s: {t: 0 for t in targets} for s in strategies}
    trade_rows = []  # per-run aggregated stats -> CSV

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
            #rs0 = run_episode_MV_daily_frictionless_hist(
            #    cfg, R_corr, returns_log, t_ann,
            #    start_idx=start_idx, T_days=T_days, seed=seed,
            #    mv_solver=mv_solver, infeasible_policy=infeasible_policy
            #)
            #if rs0 is None:
            #    skipped["MV_daily_frictionless"][t_ann] += 1
            #else:
            #    rs0_eval = _burnin_slice_rs(rs0, burnin_steps)
            #    ar = ann_arith_mean_vol_from_rsimple(rs0_eval, dt=float(cfg.dt_day))
            #    ge = ann_geom_mean_vol_from_rsimple(rs0_eval, dt=float(cfg.dt_day))
            #    raw["MV_daily_frictionless"][t_ann].append(
            #        {"arith": ar, "geom": ge, "sh_arith": sharpe_from_ann(*ar), "sh_geom": sharpe_from_ann(*ge)}
            #    )

            # MV monthly
            out = run_episode_MV_monthly_cost_hist(
                cfg, R_corr, returns_log, lam_cost, t_ann,
                start_idx=start_idx, T_days=T_days, rebalance_every=rebalance_every, seed=seed,
                mv_solver=mv_solver, infeasible_policy=infeasible_policy
            )
            if out is None:
                skipped["MV_monthly_cost"][t_ann] += 1
            else:
                rsm = out["rs"]; stats = out.get("stats", {})
                rs_eval = _burnin_slice_rs(rsm, burnin_steps)
                ar = ann_arith_mean_vol_from_rsimple(rs_eval, dt=float(cfg.dt_day))
                ge = ann_geom_mean_vol_from_rsimple(rs_eval, dt=float(cfg.dt_day))
                raw["MV_monthly_cost"][t_ann].append(
                    {"arith": ar, "geom": ge, "sh_arith": sharpe_from_ann(*ar), "sh_geom": sharpe_from_ann(*ge)}
                )
                trade_rows.append(dict(name="MV_monthly_cost", target=float(t_ann), seed=int(seed), start_idx=int(start_idx), **stats))
            
            # EW monthly
            out = run_episode_EW_monthly_cost_hist(
                cfg, R_corr, returns_log, lam_cost,
                start_idx=start_idx, T_days=T_days, rebalance_every=rebalance_every, seed=seed,
            )
            if out is None:
                skipped["EW_monthly_cost"][t_ann] += 1
            else:
                rsew = out["rs"]; stats = out.get("stats", {})
                rs_eval = _burnin_slice_rs(rsew, burnin_steps)
                ar = ann_arith_mean_vol_from_rsimple(rs_eval, dt=float(cfg.dt_day))
                ge = ann_geom_mean_vol_from_rsimple(rs_eval, dt=float(cfg.dt_day))
                raw["EW_monthly_cost"][t_ann].append(
                    {"arith": ar, "geom": ge, "sh_arith": sharpe_from_ann(*ar), "sh_geom": sharpe_from_ann(*ge)}
                )
                trade_rows.append(dict(name="EW_monthly_cost", target=float(t_ann), seed=int(seed), start_idx=int(start_idx), **stats))

            # RL A2
            out = run_episode_RL_band_A2_hist_MPC(
                cfg, R_corr, returns_log, policy_A2, lam_cost, t_ann,
                start_idx=start_idx, T_total_days=T_days, horizon_days = T_days,seed=seed, device=cfg.device,
                mv_solver=mv_solver, infeasible_policy=infeasible_policy,
                gamma_seq=gamma_seq
            )
            if out is None:
                skipped["RL_band_A2"][t_ann] += 1
            else:
                rsA2 = out["rs"]; stats = out.get("stats", {})
                rsA2_eval = _burnin_slice_rs(rsA2, burnin_steps)
                ar = ann_arith_mean_vol_from_rsimple(rsA2_eval, dt=float(cfg.dt_day))
                ge = ann_geom_mean_vol_from_rsimple(rsA2_eval, dt=float(cfg.dt_day))
                raw["RL_band_A2"][t_ann].append(
                    {"arith": ar, "geom": ge, "sh_arith": sharpe_from_ann(*ar), "sh_geom": sharpe_from_ann(*ge)}
                )
                trade_rows.append(dict(name="RL_band_A2", target=float(t_ann), seed=int(seed), start_idx=int(start_idx), **stats))

            # RL B2
            out = run_episode_RL_band_B2_hist(
                cfg, R_corr, returns_log, policy_B2, lam_cost, t_ann,
                start_idx=start_idx, T_days=T_days, seed=seed, device=cfg.device,
                mv_solver=mv_solver, qp_solver=qp_solver, infeasible_policy=infeasible_policy,
                gamma_seq=gamma_seq,center_mode=center_mode, center_policy=center_policy,
            )
            if out is None:
                skipped["RL_band_B2"][t_ann] += 1
            else:
                rsB2 = out["rs"]; stats = out.get("stats", {})
                rsB2_eval = _burnin_slice_rs(rsB2, burnin_steps)
                ar = ann_arith_mean_vol_from_rsimple(rsB2_eval, dt=float(cfg.dt_day))
                ge = ann_geom_mean_vol_from_rsimple(rsB2_eval, dt=float(cfg.dt_day))
                raw["RL_band_B2"][t_ann].append(
                    {"arith": ar, "geom": ge, "sh_arith": sharpe_from_ann(*ar), "sh_geom": sharpe_from_ann(*ge)}
                )
                trade_rows.append(dict(name="RL_band_B2", target=t_ann, seed=seed, **stats))

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
    # ---- always dump trade stats if outdir is set (keeps format stable) ----
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        df_trade = pd.DataFrame(trade_rows)
        # stable column order (subset ok if some fields missing)
        col_order = [
            "name","target","seed","start_idx",
            "steps","inside_rate","trigger_rate",
            "turnover_mean","sold_total_sum","cost_drag_sum",
            "avg_distance_to_band","avg_action_norm",
            "width_mean","width_median","width_min","width_max",
            "mstar_shift",
        ]
        df_trade = df_trade[[c for c in col_order if c in df_trade.columns]]
        df_trade.to_csv(os.path.join(outdir, "turnover_cost_stats.csv"), index=False)
  
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

# ============================================================
# Crash-aware window selection (ensure at least 1 overlaps)
# ============================================================
def _pick_start_idx_to_cover_crash(df_index: pd.DatetimeIndex, T_days: int, crash_windows: list[dict], seed: int):
    """
    Try to choose a start_idx so that [start_idx, start_idx+T_days) overlaps at least
    one crash window. Returns None if impossible.
    """
    if crash_windows is None or len(crash_windows) == 0:
        return None
    idx = pd.DatetimeIndex(df_index).sort_values()
    T_total = len(idx)
    max_start = int(T_total - T_days - 1)
    if max_start <= 0:
        return None

    candidates = []
    for cw in crash_windows:
        s = pd.Timestamp(cw["start"])
        e = pd.Timestamp(cw["end"])
        j0 = int(idx.searchsorted(s, side="left"))
        j1 = int(idx.searchsorted(e, side="right")) - 1
        if j1 < 0 or j0 >= T_total:
            continue
        j0 = max(0, min(j0, T_total - 1))
        j1 = max(0, min(j1, T_total - 1))
        # overlap condition: start <= j1 AND start+T_days-1 >= j0
        lo = max(0, j0 - (T_days - 1))
        hi = min(max_start, j1)
        if lo <= hi:
            candidates.append((lo, hi))

    if not candidates:
        return None

    rng = np.random.default_rng(int(seed))
    lo, hi = candidates[int(rng.integers(0, len(candidates)))]
    return int(rng.integers(lo, hi + 1))

def _normalize_episode_output(out):
    """
    Normalize episode runner outputs to:
      rs: np.ndarray (T,)
      trade_stats: dict or None
    Accepts:
      - None
      - np.ndarray / list (rs)
      - dict with keys {"rs", "stats"} (your current convention)
    """
    if out is None:
        return None, None
    if isinstance(out, dict):
        rs = out.get("rs", None)
        st = out.get("stats", None)
        return rs, st
    # assume array-like returns
    return out, None

def _pick_starts_with_at_least_one_crash(
     idx: pd.DatetimeIndex,
     *,
     T_days: int,
     n_windows: int,
     rng: np.random.Generator,
     crash_windows: list[dict] | None,
 ):
     """
     Pick start indices so that at least one window overlaps at least one crash window (if possible).
     Overlap condition: [start, start+T_days-1] intersects [crash_start, crash_end]
     """
     idx = pd.DatetimeIndex(idx)
     T_total = len(idx)
     Tmax = int(T_total - T_days - 1)
     if Tmax <= 0:
         raise ValueError("Not enough data for wealth windows.")

     starts = []
     if crash_windows:
         for cw in crash_windows:
             s = pd.Timestamp(cw["start"])
             e = pd.Timestamp(cw["end"])
             if s > idx[-1] or e < idx[0]:
                 continue
             # indices (clip into [0, T_total-1])
             s_i = int(idx.searchsorted(s, side="left"))
             e_i = int(idx.searchsorted(e, side="right")) - 1
             s_i = max(0, min(s_i, T_total - 1))
             e_i = max(0, min(e_i, T_total - 1))
             # overlap window start range:
             # start <= e_i and start + T_days - 1 >= s_i
             lo = max(0, s_i - (T_days - 1))
             hi = min(Tmax, e_i)
             if lo <= hi:
                 starts.append(int(rng.integers(lo, hi + 1)))
                 break  # "at least one crash-including window" requirement

     # fill remaining randomly (unique-ness is optional; keep simple)
     while len(starts) < int(n_windows):
         starts.append(int(rng.integers(0, Tmax + 1)))
     return starts[: int(n_windows)]

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
        #("MV_daily_frictionless", "o"),
        ("MV_monthly_cost", "s"),
        ("EW_monthly_cost", "D"),
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
        #f"TotRet:  {stats['ret']*100:6.2f}%\n"
        f"AnnRet:  {stats['ann_mean']*100:6.2f}%\n"
        #f"AnnVol:  {stats['ann_vol']*100:6.2f}%\n"
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
    burnin_steps: int = 0,
    center_mode: str = "policy",
    center_policy: JointBandPolicy | None = None,
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
    # force at least one window to overlap a crash window (if feasible)
    forced = _pick_start_idx_to_cover_crash(df_sel.index, T_days, crash_windows, seed=base_seed)
    if forced is not None and len(starts) > 0:
        starts[0] = int(forced)
    #starts = [int(rng.integers(0, T_max_start)) for _ in range(int(n_windows))]
    # choose windows (ensure at least one overlaps a crash window if possible)
    #rng = np.random.default_rng(int(base_seed))
    #starts = _pick_starts_with_at_least_one_crash(
    #    df_sel.index,
    #    T_days=int(T_days),
    #    n_windows=int(n_windows),
    #    rng=rng,
    #    crash_windows=crash_windows,
    #)

    # prepare crash stats rows
    crash_rows: List[Dict[str, Any]] = []
    wealth_turn_rows: List[Dict[str, Any]] = []

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
        out_mv_daily = run_episode_MV_daily_frictionless_hist(
            cfg, np.eye(cfg.N_ASSETS), returns_log, target_ann,
            start_idx=start_idx, T_days=T_days, seed=base_seed + i,
            mv_solver="OSQP", infeasible_policy="fallback",
        )
        out_mv_monthly = run_episode_MV_monthly_cost_hist(
            cfg, np.eye(cfg.N_ASSETS), returns_log, lam_cost, target_ann,
            start_idx=start_idx, T_days=T_days, rebalance_every=rebalance_every, seed=base_seed + i,
            mv_solver="OSQP", infeasible_policy="fallback",
        )
        out_ew_monthly = run_episode_EW_monthly_cost_hist(
            cfg, np.eye(cfg.N_ASSETS), returns_log, lam_cost,
            start_idx=start_idx, T_days=T_days, rebalance_every=rebalance_every, seed=base_seed + i,
        )
        out_a2 = run_episode_RL_band_A2_hist_MPC(
            cfg, np.eye(cfg.N_ASSETS), returns_log, policy_A2, lam_cost, target_ann,
            start_idx=start_idx, T_total_days=T_days, horizon_days=T_days, seed=base_seed + i,
            device=cfg.device, mv_solver="OSQP", infeasible_policy="fallback",
        )
        out_b2 = run_episode_RL_band_B2_hist(
            cfg, np.eye(cfg.N_ASSETS), returns_log, policy_B2, lam_cost, target_ann,
            start_idx=start_idx, T_days=T_days, seed=base_seed + i,
            device=cfg.device, mv_solver="OSQP", qp_solver="OSQP", infeasible_policy="fallback",
            center_mode=center_mode, center_policy=center_policy,
        )
        rs_mv_daily, st_mv_daily   = _normalize_episode_output(out_mv_daily)
        rs_mv_monthly, st_mv_monthly = _normalize_episode_output(out_mv_monthly)
        rs_ew_monthly, st_ew_monthly = _normalize_episode_output(out_ew_monthly)
        rs_a2, st_a2 = _normalize_episode_output(out_a2)
        rs_b2, st_b2 = _normalize_episode_output(out_b2)

        series = [
            #("MV_daily",   rs_mv_daily,   st_mv_daily),
            ("MV_monthly", rs_mv_monthly, st_mv_monthly),
            ("EW_monthly", rs_ew_monthly, st_ew_monthly),
            ("RL_A2",      rs_a2,         st_a2),
            ("RL_B2",      rs_b2,         st_b2),
        ]

        # plot wealth & stats box (aggregate for full window)
        y_max = None
        box_lines = []
        for name, rs, trade_st in series:
            if rs is None or len(rs) == 0:
                continue
            # rs can be np.ndarray (MV_daily) or dict (others)
            if isinstance(rs, dict):
                rs_use = rs["rs"]
                stats_use = rs.get("stats", {})
            else:
                rs_use = rs
                stats_use = {}
            dates_p, W_p = _burnin_slice_for_plot(dates, rs, burnin_steps, w0=100.0, renormalize=True)
            y = safe_log_wealth(W_p) if log_scale else W_p
            ax.plot(dates_p.insert(0, dates_p[0]), y, label=name)
            #W = wealth_from_rsimple(rs, 100.0)
            #y = safe_log_wealth(W) if log_scale else W
            #ax.plot(dates.insert(0, dates[0]), y, label=name)  # align W length = T+1
            y_max = float(np.max(y)) if y_max is None else float(max(y_max, np.max(y)))
            rs_use_eval = _burnin_slice_rs(rs_use, burnin_steps)
            st = perf_stats_from_rsimple(rs_use_eval, dt=float(cfg.dt_day))           
            #st = perf_stats_from_rsimple(rs_use, dt=float(cfg.dt_day))
            box_lines.append(f"[{name}]\n{_fmt_stats_box(st)}")
            # wealth-mode turnover/cost logger (match turnover_cost_stats.csv columns)
            if isinstance(trade_st, dict) and len(trade_st) > 0:
                 wealth_turn_rows.append(dict(
                     name=name,
                     target=float(target_ann),
                     seed=int(base_seed + i),
                     inside_rate=float(trade_st.get("inside_rate", np.nan)),
                     turnover_mean=float(trade_st.get("turnover_mean", np.nan)),
                     sold_total_sum=float(trade_st.get("sold_total_sum", np.nan)),
                     cost_drag_sum=float(trade_st.get("cost_drag_sum", np.nan)),
                     steps=int(trade_st.get("steps", len(rs))),
                     # optional but helpful for debugging:
                     window=int(i),
                     start=str(dates[0].date()) if len(dates) else "",
                     end=str(dates[-1].date()) if len(dates) else "",
                 ))

            # crash-window stats
            for cw in crash_windows:
                # only add if overlap is non-empty
                rsub = _slice_by_dates(dates, rs_use, cw["start"], cw["end"])
                if rsub.size == 0:
                    continue
                if rsub is None or len(rsub) == 0:
                     continue
                st_cw = perf_stats_from_rsimple(rsub, dt=float(cfg.dt_day))
                crash_rows.append(dict(
                    window=int(i),
                    start=str(dates[0].date()) if len(dates) else "",
                    end=str(dates[-1].date()) if len(dates) else "",
                    strategy=name,
                    crash_name=cw["name"],
                    crash_start=cw["start"],
                    crash_end=cw["end"],
                    # attach trade/band stats if available (same format as turnover_cost_stats.csv)
                    **{k: stats_use.get(k, np.nan) for k in [
                        "steps","inside_rate","trigger_rate",
                        "turnover_mean","sold_total_sum","cost_drag_sum",
                        "avg_distance_to_band","avg_action_norm",
                        "width_mean","width_median","width_min","width_max",
                        "mstar_shift",
                    ]},
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

    # crash stats CSV (same file name/columns convention)
    if crash_rows:
        df_c = pd.DataFrame(crash_rows)
        # stable column order (match turnover_cost_stats + perf stats)
        col_order = [
            "window","start","end",
            "strategy","crash_name","crash_start","crash_end",
            "steps","inside_rate","trigger_rate",
            "turnover_mean","sold_total_sum","cost_drag_sum",
            "avg_distance_to_band","avg_action_norm",
            "width_mean","width_median","width_min","width_max",
            "mstar_shift",
            "ret","ann_mean","ann_vol","sharpe","sortino","maxdd",
        ]
        df_c = df_c[[c for c in col_order if c in df_c.columns]]
        df_c.to_csv(Path(out_dir) / "crash_window_stats.csv", index=False)

    # wealth-mode turnover/cost CSV (same columns as turnover_cost_stats.csv + window/start/end)
    if wealth_turn_rows:
        df_t = pd.DataFrame(wealth_turn_rows)
        df_t.to_csv(Path(out_dir) / "turnover_cost_stats_wealth.csv", index=False)

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

# ==========================================================
# Turnover / cost logger (shared across strategies)
# ==========================================================
def _get_weights_and_wealth(env):
    """
    Best-effort extraction of (w, Y) from env.
    Works for envs that expose S (risky holdings) and C (cash).
    Returns (None, None) if unavailable.
    """
    if hasattr(env, "S") and hasattr(env, "C"):
        S = np.asarray(getattr(env, "S"), float).reshape(-1)
        C = float(getattr(env, "C"))
        Y = float(S.sum() + C)
        w = S / (Y + 1e-30)
        return w, Y
    return None, None


def _trade_stats_step(env, w_pre, Y_prev, lam_scalar):
    """
    Compute per-step turnover / sold_total / cost_drag in a solver-agnostic way.

    turnover: 0.5 * ||w_post - w_pre||_1
    sold_total: env.sold_total_last があればそれ、無ければ sell-only を仮定して weights 差分から近似
    cost_drag: (1-lam) * sold_total / Y_prev
    did_trade: turnover > tiny
    """
    w_post = getattr(env, "w_post_trade", None)
    if w_post is None:
        w_post, _ = _get_weights_and_wealth(env)
    if w_post is None:
        return dict(turnover=np.nan, sold_total=np.nan, cost_drag=np.nan, did_trade=False)

    w_post = np.asarray(w_post, float).reshape(-1)
    w_pre  = np.asarray(w_pre,  float).reshape(-1)

    turnover = 0.5 * float(np.sum(np.abs(w_post - w_pre)))

    sold_total = getattr(env, "sold_total_last", None)
    if sold_total is None or (isinstance(sold_total, float) and (not np.isfinite(sold_total))):
         sold_total = float(Y_prev) * float(np.sum(np.maximum(w_pre - w_post, 0.0)))

    kappa = max(0.0, 1.0 - float(lam_scalar))
    cost_drag = kappa * float(sold_total) / max(float(Y_prev), 1e-30)

    return dict(
        turnover=float(turnover),
        sold_total=float(sold_total),
        cost_drag=float(cost_drag),
        did_trade=bool(turnover > 1e-12),
    )

def _distance_to_band_axis(w, A, B):
    """L1 distance to the axis-aligned box [A,B] (0 if inside)."""
    w = np.asarray(w, float).reshape(-1)
    A = np.asarray(A, float).reshape(-1)
    B = np.asarray(B, float).reshape(-1)
    return float(np.sum(np.maximum(A - w, 0.0) + np.maximum(w - B, 0.0)))


def _distance_to_band_rotated(w, m, U, b_z):
    """L1 distance to rotated box {|U^T(w-m)| <= b_z} (0 if inside)."""
    w = np.asarray(w, float).reshape(-1)
    m = np.asarray(m, float).reshape(-1)
    U = np.asarray(U, float)
    b_z = np.asarray(b_z, float).reshape(-1)
    z = U.T @ (w - m)
    return float(np.sum(np.maximum(np.abs(z) - b_z, 0.0)))


def _width_stats(width_vec):
    width_vec = np.asarray(width_vec, float).reshape(-1)
    if width_vec.size == 0:
        return dict(width_mean=np.nan, width_median=np.nan, width_min=np.nan, width_max=np.nan)
    return dict(
        width_mean=float(np.mean(width_vec)),
        width_median=float(np.median(width_vec)),
        width_min=float(np.min(width_vec)),
        width_max=float(np.max(width_vec)),
    )

def _finalize_stats(*, steps, inside_cnt, outside_cnt, trigger_cnt,
                    turnover_sum, sold_sum, cost_drag_sum,
                    dist_sum, action_sum, width_all, mstar_shift=0.0):
    steps = int(steps)
    denom_io = max(1, inside_cnt + outside_cnt)
    denom_s  = max(1, steps)
    out = dict(
        steps=steps,
        inside_rate=float(inside_cnt / denom_io),
        trigger_rate=float(trigger_cnt / denom_s),
        turnover_mean=float(turnover_sum / denom_s),
        sold_total_sum=float(sold_sum),
        cost_drag_sum=float(cost_drag_sum),
        avg_distance_to_band=float(dist_sum / denom_s),
        avg_action_norm=float(action_sum / denom_s),
        mstar_shift=float(mstar_shift),
    )
    out.update(_width_stats(width_all))
    return out
 
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
    ap.add_argument("--n_eps", type=int, default=50)
    ap.add_argument("--dt", type=float, default=1.0/252)
    ap.add_argument("--burnin_days", type=int, default=21,
                    help="Exclude first N steps from performance stats and wealth plots (evaluation only).")

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
    args = ap.parse_args()

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
        OBS_BETA_ZERO = False,

        ROLL_COV_SUMMARY_ON_OBS = True,
        ROLL_OBS_LOOKBACK = 21,
        ROLL_TOP_EIGS = 5,
    )
    # --- episode-level regime randomization ---
    # Each episode samples {beta_k, sigmas_k, R_k} for all regimes and keeps them fixed within the episode.
    globalcfg.REGIME_EPISODE_RANDOMIZE = True
    globalcfg.REGIME_BETA_STD = 0.25        # std for beta perturbation
    globalcfg.REGIME_SIGMA_LOGSTD = 0.4    # log-std for sigma multiplicative noise
    globalcfg.REGIME_CORR_NOISE = 0.08      # additive noise on correlation matrix entries
    globalcfg.REGIME_BETA_CLIP = 0.999
    globalcfg.REGIME_SIGMA_CLIP = (1e-4, 10.0)

    REGIME_GAMMA_ON_OBS: bool = False
    OBS_BETA_ZERO: bool = False

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

    ## ----- load policies (if provided) -----
    #device = torch.device(args.device)
    #N = len(cols)
    ## IMPORTANT: if you use gamma, policy/value must be instantiated with global_dim=4+K
    #global_dim = 4 + (K if gamma_all is not None else 0)
    #policyA2 = None
    #policyB2 = None
    #if args.policy_A2 is not None:
    #    policyA2 = JointBandPolicy(N, d_model=128, nlayers=2, nhead=4, use_cash_softmax=True, global_dim=global_dim).to(device)
    #    policyA2.load_state_dict(torch.load(args.policy_A2, map_location=device))
    #    policyA2.eval()
    #if args.policy_B2 is not None:
    #    policyB2 = JointBandPolicy(N, d_model=128, nlayers=2, nhead=4, use_cash_softmax=True, global_dim=global_dim).to(device)
    #    policyB2.load_state_dict(torch.load(args.policy_B2, map_location=device))
    #    policyB2.eval()
    
    N = len(cols)
    # Default global_dim (used only if no checkpoint is loaded)
    global_dim_default = 4 + (K if gamma_all is not None else 0)

    def _infer_dims_from_ckpt(sd: dict, fallback_global_dim: int):
        """
        Infer per_dim/global_dim from checkpoint weights.
        - per_dim from body.token_enc.weight: [d_model, per_dim]
        - global_dim from body.glob_proj.weight: [d_model, global_dim]
        """
        per_dim = None
        glob_dim = None

        w_tok = sd.get("body.token_enc.weight", None)
        if w_tok is not None and hasattr(w_tok, "shape") and len(w_tok.shape) == 2:
            per_dim = int(w_tok.shape[1])

        w_glob = sd.get("body.glob_proj.weight", None)
        if w_glob is not None and hasattr(w_glob, "shape") and len(w_glob.shape) == 2:
            glob_dim = int(w_glob.shape[1])

        if per_dim is None:
            # fall back to old design (per_asset = [beta,w,sigma,Rw,lam] => 5)
            per_dim = 5
        if glob_dim is None:
            glob_dim = int(fallback_global_dim)
        return per_dim, glob_dim

    def _load_policy_from_path(N:int, path: str, device: str = args.device) -> JointBandPolicy:
        sd = torch.load(path, map_location=device)
        per_dim_ckpt, global_dim_ckpt = _infer_dims_from_ckpt(sd, global_dim_default)
        pol = JointBandPolicy(
            N, d_model=128, nlayers=2, nhead=4,
            use_cash_softmax=True,
            global_dim=global_dim_ckpt,
            per_dim=per_dim_ckpt,
        ).to(device)
        pol.load_state_dict(sd, strict=True)
        pol.eval()
        print(f"[Policy] loaded {path} with per_dim={per_dim_ckpt}, global_dim={global_dim_ckpt}")
        return pol

    policyA2 = None
    policyB2 = None
    if args.policy_A2 is not None:
        policyA2 = _load_policy_from_path(N = N, path=args.policy_A2, device=args.device)
    if args.policy_B2 is not None:
        policyB2 = _load_policy_from_path(N = N, path=args.policy_B2, device=args.device)
    
    # ============================================================
    # 8) Frontier evaluation (HISTORICAL)
    # ============================================================
    targets = np.linspace(0.02, 0.10, 15)   # 2% ～ 10% 年率ターゲット
    lam_cost = 0.95
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
        outdir=str(Path("outputs/historical")),
        burnin_steps=int(args.burnin_days),
        center_mode="policy",
        center_policy=policyA2,
    )

    outdir = Path("outputs/historical")
    outdir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 9) Save frontier plots
    # ============================================================
    plot_frontier_save(
        resA,
        targets,
        title="Historical Frontier (Arithmetic)",
        out_png=str(outdir / "frontier_arithmetic.png"),
    )

    plot_frontier_save(
        resG,
        targets,
        title="Historical Frontier (Geometric)",
        out_png=str(outdir / "frontier_geometric.png"),
    )

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
            out_png="outputs/historical/wealth_overlay.png",
            out_dir="outputs/historical",
            log_scale=args.wealth_log,
            crash_windows=parse_crash_windows(args.crash_windows),
            burnin_steps=int(args.burnin_days),
            center_mode="policy",
            center_policy=policyA2,
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