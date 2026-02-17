
"""
Evaluate a Stage-2 (width-only) band policy on a regime-switching GBM environment.

Features:
- multiprocessing over episode seeds
- per-(seed,target,strategy) caching of episode paths to .npz
- frontier plots (arithmetic + geometric) and wealth-path plot
- overlay plot: s, delta, reward with regime shading

Typical usage:
python eval_regime_gbm_final.py \
  --policy_path policy_stage2_regime.pt \
  --n_eps 100 --n_workers 8 \
  --lam_cost 0.995 --T_days 1260 \
  --cache_dir episode_cache_regime \
  --outdir eval_outputs_regime \
  --overlay_target 0.03 --overlay_seed 2025
"""

from __future__ import annotations

import os
import json
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import torch

from src.ppo.agent import JointBandPolicy
from src.utils.rlopt_helpers import build_corr_from_pairs, build_cov, clamp01_vec
from src.regime_gbm.gbm_env import globalsetting
from src.ppo.rollout import compute_delta_box, mv_center_qp
from src.regime_gbm.regime_gbm_env import RegimeGBMBandEnvMulti


# -----------------------------
# Annualization helpers
# -----------------------------
def ann_arith_mean_vol_from_rsimple(rsimple: np.ndarray, dt: float = 1/252) -> Tuple[float, float]:
    r = np.asarray(rsimple, float)
    return float(r.mean() / dt), float(r.std(ddof=1) / math.sqrt(dt))


def ann_geom_mean_vol_from_rsimple(rsimple: np.ndarray, dt: float = 1/252) -> Tuple[float, float]:
    r = np.asarray(rsimple, float)
    logret = np.log1p(r)
    return float(logret.mean() / dt), float(logret.std(ddof=1) / math.sqrt(dt))


# -----------------------------
# Default regime specification (same spirit as train_regime_gbm.py)
# -----------------------------
def _corr_from_pairs(N: int, base_rho: float, pair_rhos: Dict[Tuple[int, int], float]) -> np.ndarray:
    R = np.full((N, N), float(base_rho))
    np.fill_diagonal(R, 1.0)
    for (i, j), rho in pair_rhos.items():
        R[i, j] = R[j, i] = float(rho)
    # PSD fix (tiny)
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, 1e-8)
    R = (V * w) @ V.T
    d = np.sqrt(np.diag(R))
    R = R / (d[:, None] * d[None, :])
    return R


def build_default_regimes(N: int) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    pair_rhos = {(0, 1): 0.60, (1, 3): -0.20, (2, 4): 0.05}

    R1 = _corr_from_pairs(N, base_rho=0.20, pair_rhos=pair_rhos)
    R2 = _corr_from_pairs(N, base_rho=0.35, pair_rhos=pair_rhos)
    R3 = _corr_from_pairs(N, base_rho=0.10, pair_rhos=pair_rhos)

    regimes = [
        dict(
            name="calm_pos",
            beta=np.array([0.60, 0.45, 0.25, 0.35, 0.30], float),
            sigmas=np.array([0.30, 0.22, 0.10, 0.16, 0.18], float),
            R=R1,
        ),
        dict(
            name="stress_neg",
            beta=np.array([-0.70, -0.45, -0.20, -0.40, -0.35], float),
            sigmas=np.array([0.55, 0.45, 0.18, 0.30, 0.35], float),
            R=R2,
        ),
        dict(
            name="mid",
            beta=np.array([0.20, 0.10, 0.05, 0.10, 0.08], float),
            sigmas=np.array([0.40, 0.30, 0.12, 0.22, 0.25], float),
            R=R3,
        ),
    ]

    # 3-state sticky transition
    P = np.array(
        [
            [0.985, 0.010, 0.005],
            [0.020, 0.970, 0.010],
            [0.010, 0.015, 0.975],
        ],
        float,
    )
    return regimes, P


def maybe_load_regimes(path: Optional[str], N: int) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    if not path:
        return build_default_regimes(N)
    p = Path(path)
    obj = json.loads(p.read_text())
    regimes = obj["regimes"]
    P = np.asarray(obj["P"], float)
    # to ndarray
    for r in regimes:
        r["beta"] = np.asarray(r["beta"], float)
        r["sigmas"] = np.asarray(r["sigmas"], float)
        r["R"] = np.asarray(r["R"], float)
    return regimes, P


# -----------------------------
# Episode simulation (3 strategies)
# -----------------------------
@dataclass
class EpisodeResult:
    seed: int
    target: float
    strategy: str
    rsimple: np.ndarray
    wealth: np.ndarray
    # diagnostics (optional)
    s_path: Optional[np.ndarray] = None      # (T,N) for RL
    delta_path: Optional[np.ndarray] = None  # (T,N) for RL
    reward_path: Optional[np.ndarray] = None # (T,) realized r_simple
    regime_path: Optional[np.ndarray] = None # (T,) regime index


def _cache_file(cache_dir: Path, seed: int, target: float, strategy: str) -> Path:
    # target as basis points string for stability
    tcode = f"{int(round(target * 10000)):04d}bp"
    return cache_dir / f"ep_seed{seed}_t{tcode}_{strategy}.npz"


def _simulate_one(
    *,
    gcfg: Any,
    regimes: List[Dict[str, Any]],
    P: np.ndarray,
    policy: Optional[JointBandPolicy],
    target_ann: float,
    lam_cost: float,
    seed: int,
    T_days: int,
    strategy: str,
    mv_solver: str = "OSQP",
    infeasible_policy: str = "skip",
    verbose: bool = False,
) -> Optional[EpisodeResult]:
    # Make env with deterministic seed.
    cfg2 = gcfg
    cfg2.seed = int(seed)

    R0 = np.asarray(regimes[0].get('R', np.eye(cfg2.N_ASSETS)), float) if len(regimes)>0 else np.eye(cfg2.N_ASSETS)
    env = RegimeGBMBandEnvMulti(cfg=cfg2, regimes=regimes, P=P, R=R0)

    N = cfg2.N_ASSETS
    w0 = np.full(N, 1.0 / N) * 0.8
    obs = env.reset(beta=None, lam=lam_cost if strategy != "MV_daily_frictionless" else 1.0,
                    target_ret=float(target_ann), w0=w0)

    # Use env-internal effective annual target (discounted-by-bank, etc.)
    target_eff = float(env.target_ret_ann)

    # mu_eff for MV center: sigma^2 * beta (discounted-by-bank world)
    beta = np.asarray(env.beta, float).reshape(N,)
    sigmas = np.asarray(env.sigmas, float).reshape(N,)
    mu_eff = (sigmas ** 2) * beta

    #w_star = mv_weights_target_return(
    #    env.Cov, mu_eff, target_eff,
    #    allow_cash=False,
    #    solver=mv_solver,
    #    infeasible_policy=infeasible_policy,
    #)
    w_star = mv_center_qp(
        Cov=env.Cov,
        sigmas=sigmas,
        beta=beta,
        target_ann=target_eff,
        allow_cash=False,
        solver=mv_solver,
    )
    if w_star is None:
        return None

    # Diagnostics (VERY noisy under multiprocessing) -> gate by verbose
    if bool(verbose):
        print("[diag]",
            "target_eff=", target_eff,
            "mu_eff_max=", float(mu_eff.max()),
            "beta(min/mean/max)=", float(beta.min()), float(beta.mean()), float(beta.max()),
            "sigmas(min/mean/max)=", float(sigmas.min()), float(sigmas.mean()), float(sigmas.max()),
            "w_star(min/max/sum)=", float(w_star.min()), float(w_star.max()), float(w_star.sum()))
        print("[diag2]", "target_eff>mu_max?", target_eff > float(mu_eff.max()),
          "w_star_max", float(w_star.max()), "w_star_min", float(w_star.min()))

    T = int(min(int(env.T), int(T_days)))

    rs = np.zeros(T, float)
    wealth = np.ones(T, float)

    s_path = None
    delta_path = None
    reward_path = None
    regime_path = None

    if strategy == "RL_band":
        assert policy is not None
        s_path = np.zeros((T, N), float)
        delta_path = np.zeros((T, N), float)
        reward_path = np.zeros(T, float)
        regime_path = np.zeros(T, int)

    device = getattr(cfg2, "device", torch.device("cpu"))

    for t in range(T):
        if strategy == "MV_daily_frictionless":
            A = w_star
            B = w_star
            obs, _, done, r_simple = env.step(A, B, use_trade_penalty=False)

        elif strategy == "MV_monthly_cost":
            if (t % 21) == 0:
                A = w_star
                B = w_star
            else:
                A = np.zeros(N)
                B = np.ones(N)
            obs, _, done, r_simple = env.step(A, B, use_trade_penalty=True)

        elif strategy == "RL_band":
            o = np.asarray(obs, np.float32)
            with torch.no_grad():
                if hasattr(policy, "sample_s_only"):
                    s_t, _, _ = policy.sample_s_only(torch.tensor(o, device=device).unsqueeze(0))
                    s = s_t.squeeze(0).detach().cpu().numpy()
                else:
                    # fallback: stage2 sampler (m ignored)
                    m_t, s_t, _, _, _ = policy.sample_stage2(torch.tensor(o, device=device).unsqueeze(0))
                    s = s_t.squeeze(0).detach().cpu().numpy()

            minm = np.minimum(w_star, 1.0 - w_star)
            lam_scalar = float(env.lam.mean()) if isinstance(env.lam, np.ndarray) else float(env.lam)

            delta = compute_delta_box(
                w_star=w_star,
                Cov=env.Cov,
                gamma=float(getattr(cfg2, "RISK_GAMMA", 1.0)),
                lam=lam_scalar,
                scale=1.0,
                clip=(0.0, 1.0),
            )
            b = 0.95 * s * delta * minm
            A = clamp01_vec(w_star - b)
            B = np.maximum(A + 1e-6, w_star + b)

            obs, _, done, r_simple = env.step(A, B, use_trade_penalty=True)

            s_path[t, :] = s
            delta_path[t, :] = delta
            reward_path[t] = float(r_simple)
            regime_path[t] = int(getattr(env, "regime", -1))

            if t < 5:
                print("t", t,
                    "lam_scalar", lam_scalar,
                    "delta_min/mean/max", float(delta.min()), float(delta.mean()), float(delta.max()),
                    "s_min/mean/max", float(s.min()), float(s.mean()), float(s.max()))
                band = (B - A) / 2.0
                print("band_min/mean/max", float(band.min()), float(band.mean()), float(band.max()))

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        rs[t] = float(r_simple)
        wealth[t] = (wealth[t - 1] if t > 0 else 1.0) * (1.0 + rs[t])

        if done:
            rs = rs[: t + 1]
            wealth = wealth[: t + 1]
            if s_path is not None:
                s_path = s_path[: t + 1]
                delta_path = delta_path[: t + 1]
                reward_path = reward_path[: t + 1]
                regime_path = regime_path[: t + 1]
            break

    return EpisodeResult(
        seed=int(seed),
        target=float(target_ann),
        strategy=strategy,
        rsimple=rs,
        wealth=wealth,
        s_path=s_path,
        delta_path=delta_path,
        reward_path=reward_path,
        regime_path=regime_path,
    )


def load_or_run_episode(
    *,
    cache_dir: Path,
    gcfg: Any,
    regimes: List[Dict[str, Any]],
    P: np.ndarray,
    policy: Optional[JointBandPolicy],
    target_ann: float,
    lam_cost: float,
    seed: int,
    T_days: int,
    strategy: str,
    mv_solver: str,
    infeasible_policy: str,
    verbose: bool = False,
) -> Optional[EpisodeResult]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = _cache_file(cache_dir, seed, target_ann, strategy)
    if fp.exists():
        z = np.load(fp, allow_pickle=False)
        rs = z["rsimple"]
        wealth = z["wealth"]
        s_path = z["s_path"] if "s_path" in z.files else None
        delta_path = z["delta_path"] if "delta_path" in z.files else None
        reward_path = z["reward_path"] if "reward_path" in z.files else None
        regime_path = z["regime_path"] if "regime_path" in z.files else None
        return EpisodeResult(
            seed=int(seed), target=float(target_ann), strategy=strategy,
            rsimple=rs, wealth=wealth,
            s_path=s_path, delta_path=delta_path, reward_path=reward_path, regime_path=regime_path
        )

    ep = _simulate_one(
        gcfg=gcfg, regimes=regimes, P=P, policy=policy,
        target_ann=target_ann, lam_cost=lam_cost, seed=seed, T_days=T_days,
        strategy=strategy, mv_solver=mv_solver, infeasible_policy=infeasible_policy,verbose=verbose,
        )
    if ep is None:
        return None

    save_kwargs = dict(rsimple=ep.rsimple, wealth=ep.wealth)
    if ep.s_path is not None:
        save_kwargs.update(
            s_path=ep.s_path,
            delta_path=ep.delta_path,
            reward_path=ep.reward_path,
            regime_path=ep.regime_path,
        )
    np.savez_compressed(fp, **save_kwargs)
    return ep


# -----------------------------
# Parallel frontier evaluation
# -----------------------------
def _worker_one_seed(args):
    # Unpack (needed for multiprocessing pickling)
    (seed, targets, lam_cost, T_days, mv_solver, infeasible_policy,
     cache_dir, gcfg_dict, regimes_obj, P, policy_state, policy_kwargs,verbose,
     base_seed, first_target) = args
 

    # reconstruct config & policy inside worker
    gcfg = globalsetting(**gcfg_dict)
    device = gcfg.device

    policy = None
    if policy_state is not None:
        policy = JointBandPolicy(**policy_kwargs).to(device)
        policy.load_state_dict(policy_state, strict=True)
        policy.eval()

    out = []
    for t in targets:
        for strat in ("MV_daily_frictionless", "MV_monthly_cost", "RL_band"):
            verbose_one = bool(verbose) and (int(seed) == int(base_seed)) and (strat == "RL_band") and (abs(float(t) - float(first_target)) < 1e-12)
            ep = load_or_run_episode(
                cache_dir=Path(cache_dir),
                gcfg=gcfg,
                regimes=regimes_obj,
                P=P,
                policy=policy if strat == "RL_band" else None,
                target_ann=float(t),
                lam_cost=float(lam_cost),
                seed=int(seed),
                T_days=int(T_days),
                strategy=strat,
                mv_solver=mv_solver,
                infeasible_policy=infeasible_policy,
                verbose=verbose_one,
            )
            if ep is None:
                out.append((strat, float(t), None))
            else:
                out.append((strat, float(t), ep.rsimple))
    return int(seed), out


def frontier_compare_regime(
    *,
    gcfg: Any,
    regimes: List[Dict[str, Any]],
    P: np.ndarray,
    policy: JointBandPolicy,
    targets: np.ndarray,
    n_eps: int,
    lam_cost: float,
    T_days: int,
    base_seed: int,
    n_workers: int,
    cache_dir: Path,
    mv_solver: str,
    infeasible_policy: str,
    verbose: bool = False,
    # Policy reconstruction params (needed for multiprocessing).
    # JointBandPolicy does not necessarily store d_model/nlayers/nhead as attributes,
    # so we pass them explicitly.
    policy_d_model: int = 128,
    policy_nlayers: int = 2,
    policy_nhead: int = 4,
    policy_use_cash_softmax: bool = True,
) -> Tuple[Dict[str, Dict[float, Tuple[float, float]]],
           Dict[str, Dict[float, Tuple[float, float]]],
           Dict[str, Dict[float, List[Dict[str, Tuple[float, float]]]]],
           Dict[str, Any]]:

    targets = np.asarray(targets, float)
    strategies = ["MV_daily_frictionless", "MV_monthly_cost", "RL_band"]
    raw: Dict[str, Dict[float, List[Dict[str, Tuple[float, float]]]]] = {s: {float(t): [] for t in targets} for s in strategies}
    skipped: Dict[str, Dict[float, int]] = {s: {float(t): 0 for t in targets} for s in strategies}

    # serialize policy for workers
    policy_state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    # NOTE: JointBandPolicy may not expose architectural params as attributes.
    # We therefore rely on explicit args to reconstruct the model in each worker.
    policy_kwargs = dict(
        N=int(gcfg.N_ASSETS),
        d_model=int(policy_d_model),
        nlayers=int(policy_nlayers),
        nhead=int(policy_nhead),
        use_cash_softmax=bool(policy_use_cash_softmax),
    )

    # serialize gcfg for workers (dataclass-like)
    gcfg_dict = dict(
        seed=int(gcfg.seed),
        device=gcfg.device,
        N_ASSETS=int(gcfg.N_ASSETS),
        sigmas=np.asarray(getattr(gcfg, "sigmas", np.ones(gcfg.N_ASSETS))).astype(float),
        pair_rhos=getattr(gcfg, "pair_rhos", {}),
        DISCOUNT_BY_BANK=bool(getattr(gcfg, "DISCOUNT_BY_BANK", True)),
        INIT_W0_UNIFORM=bool(getattr(gcfg, "INIT_W0_UNIFORM", True)),
        BAND_SMOOTH_COEF=float(getattr(gcfg, "BAND_SMOOTH_COEF", 0.0)),
        TRADE_PEN_COEF=float(getattr(gcfg, "TRADE_PEN_COEF", 0.4)),
        ALPHA=float(getattr(gcfg, "ALPHA", 1/5)),
        STAGE1_WIDTH_COEF=float(getattr(gcfg, "STAGE1_WIDTH_COEF", 0.05)),
    )

    #seeds = [base_seed + i for i in range(int(n_eps))]
    #jobs = [
    #    (seed, targets.tolist(), lam_cost, T_days, mv_solver, infeasible_policy,
    #     str(cache_dir), gcfg_dict, regimes, P, policy_state, policy_kwargs,bool(verbose),)
    #    for seed in seeds
    #]
    
    jobs = []
    # Define the single "verbose" (seed, target) once, and pass to all workers
    base_seed_int = int(base_seed)
    first_target = float(targets[0]) if len(targets) > 0 else float("nan")
    for j in range(int(n_eps)):
        seed = int(base_seed) + j
        jobs.append((
            int(seed), targets.tolist(), float(lam_cost), int(T_days), str(mv_solver), str(infeasible_policy),
            str(cache_dir), gcfg_dict, regimes, np.asarray(P, float), policy_state, policy_kwargs,
            bool(verbose),
            base_seed_int,
            first_target,
            ))

    if n_workers <= 1:
        results = [_worker_one_seed(j) for j in jobs]
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")  # safer on macOS
        with ctx.Pool(processes=int(n_workers)) as pool:
            results = pool.map(_worker_one_seed, jobs)

    # aggregate
    for seed, outs in results:
        for strat, t, rs in outs:
            if rs is None:
                skipped[strat][t] += 1
                continue
            raw[strat][t].append({
                "arith": ann_arith_mean_vol_from_rsimple(rs),
                "geom": ann_geom_mean_vol_from_rsimple(rs),
            })

    resA: Dict[str, Dict[float, Tuple[float, float]]] = {s: {} for s in strategies}
    resG: Dict[str, Dict[float, Tuple[float, float]]] = {s: {} for s in strategies}

    for s in strategies:
        for t in targets:
            t = float(t)
            ptsA = np.array([d["arith"] for d in raw[s][t]], float)
            ptsG = np.array([d["geom"] for d in raw[s][t]], float)
            if ptsA.size == 0:
                resA[s][t] = (np.nan, np.nan)
                resG[s][t] = (np.nan, np.nan)
            else:
                resA[s][t] = (float(ptsA[:, 0].mean()), float(ptsA[:, 1].mean()))
                resG[s][t] = (float(ptsG[:, 0].mean()), float(ptsG[:, 1].mean()))

    info = {"skipped": skipped, "n_eps": int(n_eps)}
    return resA, resG, raw, info


# -----------------------------
# Plotting
# -----------------------------
def plot_frontier_two(res_arith, res_geom, targets, *, title_prefix: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    targets = list(map(float, targets))
    strategies = [("MV_daily_frictionless", "o"),
                  ("MV_monthly_cost", "s"),
                  ("RL_band", "^")]

    # Arithmetic
    plt.figure(figsize=(8, 6))
    for s, marker in strategies:
        xs = [res_arith[s][t][1] for t in targets]
        ys = [res_arith[s][t][0] for t in targets]
        plt.plot(xs, ys, marker=marker, label=s)
    plt.xlabel("Annualized Volatility (arith std of r)")
    plt.ylabel("Annualized Mean (arith mean of r)")
    plt.title(f"{title_prefix} (Arithmetic)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "frontier_arith.png", dpi=180)
    plt.close()

    # Geometric
    plt.figure(figsize=(8, 6))
    for s, marker in strategies:
        xs = [res_geom[s][t][1] for t in targets]
        ys = [res_geom[s][t][0] for t in targets]
        plt.plot(xs, ys, marker=marker, label=s)
    plt.xlabel("Annualized Volatility (std of log(1+r))")
    plt.ylabel("Annualized Mean (mean of log(1+r))")
    plt.title(f"{title_prefix} (Geometric / log-return)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "frontier_geom.png", dpi=180)
    plt.close()


def plot_wealth_paths(
    *,
    gcfg: Any,
    regimes: List[Dict[str, Any]],
    P: np.ndarray,
    policy: JointBandPolicy,
    target: float,
    lam_cost: float,
    seed: int,
    T_days: int,
    cache_dir: Path,
    outdir: Path,
    mv_solver: str,
    infeasible_policy: str,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    eps = {}
    for strat in ("MV_daily_frictionless", "MV_monthly_cost", "RL_band"):
        ep = load_or_run_episode(
            cache_dir=cache_dir,
            gcfg=gcfg,
            regimes=regimes,
            P=P,
            policy=policy if strat == "RL_band" else None,
            target_ann=float(target),
            lam_cost=float(lam_cost),
            seed=int(seed),
            T_days=int(T_days),
            strategy=strat,
            mv_solver=str(mv_solver),
            infeasible_policy=str(infeasible_policy),
        )
        if ep is not None:
            eps[strat] = ep

    plt.figure(figsize=(9, 5))
    for strat, ep in eps.items():
        plt.plot(ep.wealth, label=strat)
    plt.xlabel("day")
    plt.ylabel("wealth (normalized)")
    plt.title(f"Wealth path (seed={seed}) target={target:.2%} lam={lam_cost}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"wealth_seed{seed}_target{int(target*10000):04d}bp.png", dpi=180)
    plt.close()


def plot_overlay_s_delta_reward(
    *,
    cache_dir: Path,
    seed: int,
    target: float,
    outdir: Path,
) -> None:
    """Overlay plot using cached RL episode (must have s/delta/reward/regime)."""
    outdir.mkdir(parents=True, exist_ok=True)
    fp = _cache_file(cache_dir, seed, target, "RL_band")
    if not fp.exists():
        print(f"[overlay] cache missing: {fp}")
        return
    z = np.load(fp, allow_pickle=False)
    if "s_path" not in z.files:
        print(f"[overlay] missing diagnostics in cache: {fp}")
        return

    s_path = z["s_path"]        # (T,N)
    delta_path = z["delta_path"]
    reward_path = z["reward_path"]
    regime_path = z["regime_path"] if "regime_path" in z.files else None

    T = reward_path.shape[0]
    t = np.arange(T)

    # Use averages across assets for readability
    s_mean = s_path.mean(axis=1)
    d_mean = delta_path.mean(axis=1)

    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)

    ax[0].plot(t, reward_path)
    ax[0].set_ylabel("r_simple")
    ax[0].grid(alpha=0.3)

    ax[1].plot(t, s_mean)
    ax[1].set_ylabel("mean(s)")
    ax[1].grid(alpha=0.3)

    ax[2].plot(t, d_mean)
    ax[2].set_ylabel("mean(delta)")
    ax[2].set_xlabel("day")
    ax[2].grid(alpha=0.3)

    # Regime shading
    if regime_path is not None and regime_path.size == T:
        # find segments
        cur = int(regime_path[0])
        start = 0
        for i in range(1, T):
            if int(regime_path[i]) != cur:
                for a in ax:
                    a.axvspan(start, i, alpha=0.08)
                start = i
                cur = int(regime_path[i])
        for a in ax:
            a.axvspan(start, T, alpha=0.08)

    fig.suptitle(f"Overlay (RL): s, delta, reward | seed={seed} target={target:.2%}")
    fig.tight_layout()
    fig.savefig(outdir / f"overlay_seed{seed}_target{int(target*10000):04d}bp.png", dpi=180)
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def build_gcfg(device: torch.device, N: int, seed: int) -> Any:
    # Minimal cfg; sigmas/pair_rhos are overridden by regimes inside RegimeGBM env, but keep them consistent.
    return globalsetting(
        seed=seed,
        device=device,
        N_ASSETS=N,
        sigmas=np.array([0.40, 0.30, 0.12, 0.22, 0.25], dtype=float),
        pair_rhos={(0, 1): 0.60, (1, 3): -0.20, (2, 4): 0.05},
        DISCOUNT_BY_BANK=True,
        INIT_W0_UNIFORM=True,
        BAND_SMOOTH_COEF=0.0,
        TRADE_PEN_COEF=0.4,
        ALPHA=1/5,
        STAGE1_WIDTH_COEF=0.05,
    )


def load_policy(
    policy_path: str,
    N: int,
    device: torch.device,
    *,
    d_model: int = 128,
    nlayers: int = 2,
    nhead: int = 4,
    use_cash_softmax: bool = True,
) -> JointBandPolicy:
    pol = JointBandPolicy(N, d_model=d_model, nlayers=nlayers, nhead=nhead, use_cash_softmax=use_cash_softmax).to(device)
    sd = torch.load(policy_path, map_location=device)
    pol.load_state_dict(sd, strict=True)
    pol.eval()
    return pol

def plot_wealth_compare_two_policies(
    *,
    gcfg: Any,
    regimes: List[Dict[str, Any]],
    P: np.ndarray,
    policy_A: JointBandPolicy,
    policy_B: JointBandPolicy,
    label_A: str,
    label_B: str,
    target: float,
    lam_cost: float,
    seed: int,
    T_days: int,
    cache_dir_A: Path,
    cache_dir_B: Path,
    outdir: Path,
    mv_solver: str,
    infeasible_policy: str,
) -> None:
     """
     Save: wealth_compare.png
       - overlay wealth of A2/B2 for the SAME (seed, target, lam, regime_json).
       - uses RL_band episodes (caching respected).
     """
     outdir.mkdir(parents=True, exist_ok=True)
 
     epA = load_or_run_episode(
         cache_dir=cache_dir_A,
         gcfg=gcfg,
         regimes=regimes,
         P=P,
         policy=policy_A,
         target_ann=float(target),
         lam_cost=float(lam_cost),
         seed=int(seed),
         T_days=int(T_days),
         strategy="RL_band",
         mv_solver=str(mv_solver),
         infeasible_policy=str(infeasible_policy),
         )
     epB = load_or_run_episode(
         cache_dir=cache_dir_B,
         gcfg=gcfg,
         regimes=regimes,
         P=P,
         policy=policy_B,
         target_ann=float(target),
         lam_cost=float(lam_cost),
         seed=int(seed),
         T_days=int(T_days),
         strategy="RL_band",
         mv_solver=str(mv_solver),
         infeasible_policy=str(infeasible_policy),
     )
 
     if epA is None or epB is None:
        print("[wealth_compare] missing episode(s):", "A is None" if epA is None else "", "B is None" if epB is None else "")
        return

     plt.figure(figsize=(9, 5))
     plt.plot(epA.wealth, label=f"{label_A}: RL_band")
     plt.plot(epB.wealth, label=f"{label_B}: RL_band")
     plt.xlabel("day")
     plt.ylabel("wealth (normalized)")
     plt.title(f"Wealth compare (seed={seed}) target={target:.2%} lam={lam_cost}")
     plt.legend()
     plt.grid(alpha=0.3)
     plt.tight_layout()
     plt.savefig(outdir / "wealth_compare.png", dpi=180)
     plt.close()

def plot_frontier_compare_multi(
     *,
     resA_A: Dict[str, Dict[float, Tuple[float, float]]],
     resG_A: Dict[str, Dict[float, Tuple[float, float]]],
     resA_B: Dict[str, Dict[float, Tuple[float, float]]],
     resG_B: Dict[str, Dict[float, Tuple[float, float]]],
     targets: np.ndarray,
     label_A: str,
     label_B: str,
     title_prefix: str,
     outdir: Path,
 ) -> None:
     """
     Save:
       - frontier_arith_compare.png
       - frontier_geom_compare.png
 
     Compare A2 vs B2 on:
       - RL_band
       - MV_monthly_cost
     """
     outdir.mkdir(parents=True, exist_ok=True)
     targets = list(map(float, targets))
 
     def _plot_one(resA_A, resA_B, which: str, fname: str, xlabel: str, ylabel: str):
         plt.figure(figsize=(8, 6))
 
         for (resA, lab, ls) in [
             (resA_A, f"{label_A}", "-"),
             (resA_B, f"{label_B}", "--"),
         ]:
             # RL_band
             xs = [resA["RL_band"][t][1] for t in targets]
             ys = [resA["RL_band"][t][0] for t in targets]
             plt.plot(xs, ys, marker="^", linestyle=ls, label=f"{lab}: RL_band")
 
             # MV_monthly_cost
             xs = [resA["MV_monthly_cost"][t][1] for t in targets]
             ys = [resA["MV_monthly_cost"][t][0] for t in targets]
             plt.plot(xs, ys, marker="s", linestyle=ls, label=f"{lab}: MV_monthly")
 
         plt.xlabel(xlabel)
         plt.ylabel(ylabel)
         plt.title(f"{title_prefix} ({which})")
         plt.grid(True, alpha=0.3)
         plt.legend()
         plt.tight_layout()
         plt.savefig(outdir / fname, dpi=180)
         plt.close()

     # Arithmetic
     _plot_one(
         resA_A, resA_B,
         which="Arithmetic",
         fname="frontier_arith_compare.png",
         xlabel="Annualized Volatility (arith std of r)",
         ylabel="Annualized Mean (arith mean of r)",
     )
 
     # Geometric
     _plot_one(
         resG_A, resG_B,
         which="Geometric (log-return)",
         fname="frontier_geom_compare.png",
         xlabel="Annualized Volatility (std of log(1+r))",
         ylabel="Annualized Mean (mean of log(1+r))",
     )

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_path", type=str, required=True, help="Policy A (e.g., A2)")
    ap.add_argument("--policy_path_B", type=str, default=None, help="Optional Policy B (e.g., B2) for comparison")
    ap.add_argument("--label_A", type=str, default="A", help="Label for policy_path")
    ap.add_argument("--label_B", type=str, default="B", help="Label for policy_path_B")
    # Policy architecture (must match the checkpoint). Defaults align with training scripts.
    ap.add_argument("--policy_d_model", type=int, default=128)
    ap.add_argument("--policy_nlayers", type=int, default=2)
    ap.add_argument("--policy_nhead", type=int, default=4)
    ap.add_argument("--policy_use_cash_softmax", action="store_true", default=True)
    ap.add_argument("--policy_no_cash_softmax", action="store_true", default=False,
                    help="If set, overrides --policy_use_cash_softmax and disables cash-softmax.")
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--targets_start", type=float, default=0.01)
    ap.add_argument("--targets_end", type=float, default=0.041)
    ap.add_argument("--targets_step", type=float, default=0.001)
    ap.add_argument("--n_eps", type=int, default=100)
    ap.add_argument("--base_seed", type=int, default=2025)
    ap.add_argument("--lam_cost", type=float, default=0.995)
    ap.add_argument("--T_days", type=int, default=1260)
    ap.add_argument("--n_workers", type=int, default=8)
    ap.add_argument("--cache_dir", type=str, default="episode_cache_regime")
    ap.add_argument("--outdir", type=str, default="eval_outputs_regime")
    ap.add_argument("--mv_solver", type=str, default="OSQP")
    ap.add_argument("--infeasible_policy", type=str, default="skip", choices=["skip", "fallback"])
    ap.add_argument("--regime_json", type=str, default=None, help="Optional JSON with regimes/P")
    ap.add_argument("--overlay_seed", type=int, default=2025)
    ap.add_argument("--overlay_target", type=float, default=0.03)
    ap.add_argument("--verbose", action="store_true", default=False)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gcfg = build_gcfg(device=device, N=int(args.N), seed=int(args.base_seed))

    regimes, P = maybe_load_regimes(args.regime_json, int(args.N))
    targets = np.arange(args.targets_start, args.targets_end, args.targets_step, dtype=float)
    # cash-softmax flag resolution (no_cash overrides)
    use_cash = bool(args.policy_use_cash_softmax) and (not bool(args.policy_no_cash_softmax))


    # policy A
    policy_A = load_policy(
        args.policy_path,
        N=int(args.N),
        device=device,
        d_model=args.policy_d_model,
        nlayers=args.policy_nlayers,
        nhead=args.policy_nhead,
        use_cash_softmax=bool(use_cash),
    )

    # optional policy B
    policy_B = None
    if args.policy_path_B:
        policy_B = load_policy(
            args.policy_path_B,
            N=int(args.N),
            device=device,
            d_model=args.policy_d_model,
            nlayers=args.policy_nlayers,
            nhead=args.policy_nhead,
            use_cash_softmax=bool(use_cash),
        )

    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    # Always separate caches by label to avoid collisions (same seed/target/strategy)
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    outdir_A = outroot / args.label_A
    cache_dir_A = cache_root / args.label_A
    outdir_A.mkdir(parents=True, exist_ok=True)
    cache_dir_A.mkdir(parents=True, exist_ok=True)

    outdir_B = None
    cache_dir_B = None
    if policy_B is not None:
        outdir_B = outroot / args.label_B
        cache_dir_B = cache_root / args.label_B
        outdir_B.mkdir(parents=True, exist_ok=True)
        cache_dir_B.mkdir(parents=True, exist_ok=True)

    print(f"[eval] starting frontier evaluation: {args.label_A}")
    resA_A, resG_A, raw_A, info_A = frontier_compare_regime(
          gcfg=gcfg,
          regimes=regimes,
          P=P,
          policy=policy_A,
          targets=targets,
          n_eps=int(args.n_eps),
          lam_cost=float(args.lam_cost),
          T_days=int(args.T_days),
          base_seed=int(args.base_seed),
          n_workers=int(args.n_workers),
          cache_dir=cache_dir_A,
          mv_solver=str(args.mv_solver),
          infeasible_policy=str(args.infeasible_policy),
          policy_d_model=int(args.policy_d_model),
          policy_nlayers=int(args.policy_nlayers),
          policy_nhead=int(args.policy_nhead),
          policy_use_cash_softmax=bool(use_cash),
          verbose=bool(args.verbose),
          )
    summary_A = {"args": vars(args), "label": args.label_A, "policy_path": args.policy_path, "skipped": info_A["skipped"]}
    (outdir_A / "summary.json").write_text(json.dumps(summary_A, indent=2))

    titleA = f"Regime-switching GBM: {args.label_A} (MV center + RL band vs MV baselines)"
    plot_frontier_two(resA_A, resG_A, targets, title_prefix=titleA, outdir=outdir_A)

    plot_wealth_paths(
         gcfg=gcfg, regimes=regimes, P=P, policy=policy_A,
         target=float(args.overlay_target),
         lam_cost=float(args.lam_cost),
         seed=int(args.overlay_seed),
         T_days=min(int(args.T_days), 252 * 3),
         cache_dir=cache_dir_A,
         outdir=outdir_A,
         mv_solver=str(args.mv_solver),
         infeasible_policy=str(args.infeasible_policy),
         )
    _ = load_or_run_episode(
         cache_dir=cache_dir_A,
         gcfg=gcfg,
         regimes=regimes,
         P=P,
         policy=policy_A,
         target_ann=float(args.overlay_target),
         lam_cost=float(args.lam_cost),
         seed=int(args.overlay_seed),
         T_days=int(args.T_days),
         strategy="RL_band",
         mv_solver=str(args.mv_solver),
         infeasible_policy="skip",
     )
    plot_overlay_s_delta_reward(
         cache_dir=cache_dir_A,
         seed=int(args.overlay_seed),
         target=float(args.overlay_target),
         outdir=outdir_A,
     )
 
     # Optional B policy run + compare plots
    if policy_B is not None:
         assert outdir_B is not None and cache_dir_B is not None
         print(f"[eval] starting frontier evaluation: {args.label_B}")
         resA_B, resG_B, raw_B, info_B = frontier_compare_regime(
             gcfg=gcfg,
             regimes=regimes,
             P=P,
             policy=policy_B,
             targets=targets,
             n_eps=int(args.n_eps),
             lam_cost=float(args.lam_cost),
             T_days=int(args.T_days),
             base_seed=int(args.base_seed),
             n_workers=int(args.n_workers),
             cache_dir=cache_dir_B,
             mv_solver=str(args.mv_solver),
             infeasible_policy=str(args.infeasible_policy),
             policy_d_model=int(args.policy_d_model),
             policy_nlayers=int(args.policy_nlayers),
             policy_nhead=int(args.policy_nhead),
             policy_use_cash_softmax=bool(use_cash),
             verbose=bool(args.verbose),
         )
 
         summary_B = {"args": vars(args), "label": args.label_B, "policy_path": args.policy_path_B, "skipped": info_B["skipped"]}
         (outdir_B / "summary.json").write_text(json.dumps(summary_B, indent=2))
 
         titleB = f"Regime-switching GBM: {args.label_B} (MV center + RL band vs MV baselines)"
         plot_frontier_two(resA_B, resG_B, targets, title_prefix=titleB, outdir=outdir_B)
 
         plot_wealth_paths(
             gcfg=gcfg, regimes=regimes, P=P, policy=policy_B,
             target=float(args.overlay_target),
             lam_cost=float(args.lam_cost),
             seed=int(args.overlay_seed),
             T_days=min(int(args.T_days), 252 * 3),
             cache_dir=cache_dir_B,
             outdir=outdir_B,
             mv_solver=str(args.mv_solver),
             infeasible_policy=str(args.infeasible_policy),
         )
         _ = load_or_run_episode(
             cache_dir=cache_dir_B,
             gcfg=gcfg,
             regimes=regimes,
             P=P,
             policy=policy_B,
             target_ann=float(args.overlay_target),
             lam_cost=float(args.lam_cost),
             seed=int(args.overlay_seed),
             T_days=int(args.T_days),
             strategy="RL_band",
             mv_solver=str(args.mv_solver),
             infeasible_policy="skip",
         )
         plot_overlay_s_delta_reward(
             cache_dir=cache_dir_B,
             seed=int(args.overlay_seed),
             target=float(args.overlay_target),
             outdir=outdir_B,
         )
 
         # ---- compare plots (saved in outroot) ----
         plot_frontier_compare_multi(
             resA_A=resA_A, resG_A=resG_A,
             resA_B=resA_B, resG_B=resG_B,
             targets=targets,
             label_A=args.label_A,
             label_B=args.label_B,
             title_prefix="Regime-switching GBM: A2 vs B2",
             outdir=outroot,
         )
         plot_wealth_compare_two_policies(
             gcfg=gcfg,
             regimes=regimes,
             P=P,
             policy_A=policy_A,
             policy_B=policy_B,
             label_A=args.label_A,
             label_B=args.label_B,
             target=float(args.overlay_target),
             lam_cost=float(args.lam_cost),
             seed=int(args.overlay_seed),
             T_days=int(args.T_days),
             cache_dir_A=cache_dir_A,
             cache_dir_B=cache_dir_B,
             outdir=outroot,
             mv_solver=str(args.mv_solver),
             infeasible_policy=str(args.infeasible_policy),
         )
 
    print(f"[eval] done. outputs in: {outroot}")

if __name__ == "__main__":
    main()