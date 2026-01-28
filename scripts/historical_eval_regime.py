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

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cvxpy as cp
import torch

from src.historical.env import HistoricalBandEnvMulti            # ←そのまま
from src.utils.rlopt_helpers import clamp01_vec
from src.ppo.agent import JointBandPolicy
from src.ppo.rollout import compute_delta_box, apply_topk_s
from src.regime_gbm.gbm_env import globalsetting

globalcfg = globalsetting()   # <- instantiate


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

    window = np.asarray(returns_log[start_idx : start_idx + T_days], float)
    target_ann_eff = float(env.target_ret_ann)

    mu_ann, Cov_ann = estimate_mu_cov_ann_from_log_window(window, dt=float(cfg.dt_day))
    w_star = mv_weights_target_return(
        Cov_ann, mu_ann, target_ann_eff, allow_cash=False, solver=mv_solver, infeasible_policy=infeasible_policy
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

    window = np.asarray(returns_log[start_idx : start_idx + T_days], float)
    target_ann_eff = float(env.target_ret_ann)

    mu_ann, Cov_ann = estimate_mu_cov_ann_from_log_window(window, dt=float(cfg.dt_day))
    m_star = mv_weights_target_return(
        Cov_ann, mu_ann, target_ann_eff, allow_cash=False, solver=mv_solver, infeasible_policy=infeasible_policy
    )
    if m_star is None:
        return None

    rs = []
    for _t in range(env.T):
        o = np.array(obs, dtype=np.float32)
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
            gamma=float(getattr(cfg, "RISK_GAMMA", 1.0)),
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
    mv_allow_cash: bool = False,
    force_s_one: bool = False,
    topk: int | None = None,
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

    window = np.asarray(returns_log[start_idx : start_idx + T_days], float)
    target_ann_eff = float(env.target_ret_ann)

    mu_ann, Cov_ann = estimate_mu_cov_ann_from_log_window(window, dt=float(cfg.dt_day))
    m_star = mv_weights_target_return(
        Cov_ann, mu_ann, target_ann_eff, allow_cash=mv_allow_cash, solver=mv_solver, infeasible_policy=infeasible_policy
    )
    if m_star is None:
        return None

    lam_scalar = float(env.lam.mean()) if isinstance(env.lam, np.ndarray) else float(env.lam)
    U, delta_z = compute_levelB_basis_and_width(
        m_star=m_star,
        Cov=Cov_ann,
        gamma=float(getattr(cfg, "RISK_GAMMA", 1.0)),
        lam_scalar=lam_scalar,
        scale=1.0,
    )

    rs = []
    for _t in range(env.T):
        o = np.array(obs, dtype=np.float32)
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
                mv_solver=mv_solver, infeasible_policy=infeasible_policy
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
                mv_solver=mv_solver, qp_solver=qp_solver, infeasible_policy=infeasible_policy
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

        dates = df_sel.index[start_idx : start_idx + T_days + 1]

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
            y = safe_log_wealth(W) if log_scale else W
            ax.plot(dates, y, label=label)

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

def summaries_to_frames(resA, resG, shA, shG, targets):
    rowsA, rowsG = [], []
    for s in resA.keys():
        for t in targets:
            m, v = resA[s][t]
            rowsA.append(dict(strategy=s, target=float(t), mean=float(m), vol=float(v), sharpe=float(shA[s][t])))
            m2, v2 = resG[s][t]
            rowsG.append(dict(strategy=s, target=float(t), mean=float(m2), vol=float(v2), sharpe=float(shG[s][t])))
    return pd.DataFrame(rowsA), pd.DataFrame(rowsG)


# ============================================================
# 7) MAIN
# ============================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="asset_data/asset_returns.pkl")
    ap.add_argument("--cols", type=str, default="LargeCap,MidCap,SmallCap,EAFE,EM")
    ap.add_argument("--policy_a2", type=str, default="checkpoints_regime_A2/policy_stage2_A2_regime.pt")
    ap.add_argument("--policy_b2", type=str, default="checkpoints_regime_B2/policy_stage2_B2_finetuned.pt")
    ap.add_argument("--targets", type=str, default="0.02:0.12:0.004")  # start:stop:step
    ap.add_argument("--n_eps", type=int, default=30)
    ap.add_argument("--T_days", type=int, default=252*5)
    ap.add_argument("--lam", type=float, default=0.99)
    ap.add_argument("--rebalance_every", type=int, default=21)
    ap.add_argument("--out_dir", type=str, default="outputs/historical")
    ap.add_argument("--run_name", type=str, default=None)
    ap.add_argument("--eval_start", type=str, default=None, help="YYYY-MM-DD (slice data for evaluation)")
    ap.add_argument("--eval_end", type=str, default=None, help="YYYY-MM-DD (slice data for evaluation)")

    ap.add_argument("--plot_wealth", action="store_true", help="Also plot wealth overlay grids.")
    ap.add_argument("--wealth_target", type=float, default=0.06)
    ap.add_argument("--wealth_windows", type=int, default=4)
    ap.add_argument("--wealth_log", action="store_true", help="Plot log(wealth) (recommended).")

    ap.add_argument("--base_seed", type=int, default=2025)
    ap.add_argument("--mv_solver", type=str, default="OSQP")
    ap.add_argument("--qp_solver", type=str, default="OSQP")
    ap.add_argument("--infeasible_policy", type=str, default="skip")
    ap.add_argument("--no_show", action="store_true", help="Do not show plots (still saves png).")
    args = ap.parse_args()

    DATA_PKL = args.data
    PREFER_COLS = [c.strip() for c in args.cols.split(",") if c.strip()]
    policy_path_A2 = args.policy_a2
    policy_path_B2 = args.policy_b2

    # parse targets "a:b:step"
    if ":" in args.targets:
        a, b, st = args.targets.split(":")
        targets = np.arange(float(a), float(b), float(st))
    else:
        targets = np.array([float(x) for x in args.targets.split(",") if x.strip()], float)
    n_eps = int(args.n_eps)
    T_days = int(args.T_days)
    lam_cost = float(args.lam)
    rebalance_every = int(args.rebalance_every)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or f"hist_{ts}_eps{n_eps}_T{T_days}_lam{lam_cost:.3f}"
    out_root = Path(args.out_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    # ----- load returns -----
    N = int(globalcfg.N_ASSETS)
    df_sel, Rlog, cols = load_historical_log_returns(DATA_PKL, N, prefer_cols=PREFER_COLS)
    # apply eval period slice (BEFORE dropna window sampling)
    df_sel = apply_eval_date_range(df_sel, args.eval_start, args.eval_end)
    df_sel = df_sel.dropna(axis=0, how="any")
    Rlog = df_sel.to_numpy(dtype=float)

    print(f"[data] selected columns (N={N}): {cols}")
    print(f"[data] usable rows after eval-slice+dropna: {len(df_sel)} (from {DATA_PKL})")
    if args.eval_start or args.eval_end:
        print(f"[data] eval range: start={args.eval_start} end={args.eval_end}")

    # Ensure contiguous windows possible
    if Rlog.shape[0] < T_days + 10:
        raise ValueError(f"Not enough history after dropna: T={Rlog.shape[0]} < T_days={T_days}")

    # ----- load policies -----
    policy_A2 = JointBandPolicy(N, d_model=128, nlayers=2, nhead=4, use_cash_softmax=True)
    policy_A2.load_state_dict(torch.load(policy_path_A2, map_location=globalcfg.device))
    policy_A2.to(globalcfg.device).eval()

    policy_B2 = JointBandPolicy(N, d_model=128, nlayers=2, nhead=4, use_cash_softmax=True)
    policy_B2.load_state_dict(torch.load(policy_path_B2, map_location=globalcfg.device))
    policy_B2.to(globalcfg.device).eval()

    # ----- run eval -----
    resA, resG, shA, shG, raw, info = eval_frontier_historical(
        globalcfg,
        returns_log=Rlog,
        targets=targets,
        policy_A2=policy_A2,
        policy_B2=policy_B2,
        n_eps=n_eps,
        lam_cost=lam_cost,
        rebalance_every=rebalance_every,
        T_days=T_days,
        base_seed=int(args.base_seed),
        mv_solver=str(args.mv_solver),
        qp_solver=str(args.qp_solver),
        infeasible_policy=str(args.infeasible_policy),
    )

    dfA, dfG = summaries_to_frames(resA, resG, shA, shG, targets)
    dfA.to_csv(out_root / "summary_arith.csv", index=False)
    dfG.to_csv(out_root / "summary_geom.csv", index=False)
    with open(out_root / "summary.json", "w") as f:
        json.dump(
            dict(
                run_name=run_name,
                data=str(DATA_PKL),
                cols=cols,
                targets=[float(t) for t in targets],
                n_eps=n_eps,
                T_days=T_days,
                lam_cost=lam_cost,
                rebalance_every=rebalance_every,
                base_seed=int(args.base_seed),
                mv_solver=str(args.mv_solver),
                qp_solver=str(args.qp_solver),
                infeasible_policy=str(args.infeasible_policy),
                skipped=info.get("skipped", {}),
            ),
            f,
            indent=2,
        )
    pd.to_pickle(raw, out_root / "raw.pkl")
    plot_frontier_save(resA, targets, "Historical Frontier (Arithmetic)", str(out_root / "frontier_arith.png"))
    plot_frontier_save(resG, targets, "Historical Frontier (Geometric / log1p)", str(out_root / "frontier_geom.png"))

    # ----- optional wealth overlay -----
    if args.plot_wealth:
        out_png = str(out_root / "wealth_overlay.png")
        plot_wealth_overlay_grid(
            df_sel=df_sel, returns_log=Rlog, cfg=globalcfg,
            policy_A2=policy_A2, policy_B2=policy_B2,
            target_ann=float(args.wealth_target),
            T_days=T_days, lam_cost=lam_cost, rebalance_every=rebalance_every,
            n_windows=int(args.wealth_windows),
            seed=int(args.base_seed) + 777,
            log_scale=bool(args.wealth_log),
            mv_solver=str(args.mv_solver), qp_solver=str(args.qp_solver),
            out_png=out_png,
        )

    print(f"[Saved] {out_root}")

    # ----- print summary -----
    print("\n=== Arithmetic (mean, vol, Sharpe) ===")
    for s in resA.keys():
        print(f"\n[{s}]")
        for t in targets:
            m, v = resA[s][t]
            print(f"  target={t: .3f}  mean={m: .4f}  vol={v: .4f}  Sharpe={shA[s][t]: .3f}")

    print("\n=== Geometric (mean, vol, Sharpe) ===")
    for s in resG.keys():
        print(f"\n[{s}]")
        for t in targets:
            m, v = resG[s][t]
            print(f"  target={t: .3f}  mean={m: .4f}  vol={v: .4f}  Sharpe={shG[s][t]: .3f}")

    # ----- plot -----
    if not args.no_show:
        plot_frontier(resA, targets, "Historical Frontier (Arithmetic)")
        plot_frontier(resG, targets, "Historical Frontier (Geometric / log1p)")