#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_frontier_v3.py
- Compare MV baselines vs RL (A2 executor/B executor/s=1/finetuned-B)
- Fixed seed cases, save raw npz and png plot
"""

import os
import json
import time
import argparse
import random
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from PPO_agent_v3 import JointBandPolicy, ValueNetCLS
from Rollout_HJB_box_v3 import mv_center_qp, compute_delta_box, compute_delta_rotated
from GBMEnv_LQ_v4 import globalsetting, GBMBandEnvMulti
from RLopt_helpers import build_corr_from_pairs, clamp01_vec


# -----------------------------
# utils
# -----------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def load_checkpoint(path: str, device: torch.device):
    payload = torch.load(path, map_location=device)
    meta = payload.get("meta", {})
    gdim = meta.get("global_dim", None)
    if gdim is not None and int(gdim) != 5:
        raise ValueError(f"checkpoint global_dim mismatch: expected 5, got {gdim}")
    return payload


def ann_arith_mean_vol_from_rsimple(rsimple: np.ndarray, dt: float = 1 / 252) -> Tuple[float, float]:
    r = np.asarray(rsimple, float)
    ann_mean = float(r.mean() / dt)
    ann_vol = float(r.std() / np.sqrt(dt))
    return ann_mean, ann_vol


def ann_geom_mean_vol_from_rsimple(rsimple: np.ndarray, dt: float = 1 / 252) -> Tuple[float, float]:
    r = np.asarray(rsimple, float)
    logret = np.log1p(r)
    ann_mean = float(logret.mean() / dt)
    ann_vol = float(logret.std() / np.sqrt(dt))
    return ann_mean, ann_vol


def make_cases(n_cases: int, *, base_seed: int, n_assets: int, beta_low: float, beta_high: float) -> List[Tuple[int, np.ndarray]]:
    cases = []
    for k in range(n_cases):
        seed = base_seed + k
        rng = np.random.default_rng(seed)
        beta = rng.uniform(beta_low, beta_high, size=n_assets)
        cases.append((seed, beta))
    return cases


# -----------------------------
# episode runners
# -----------------------------

def run_episode_mv_center(env_cfg, R, beta, lam_cost, target_ann, *, seed: int, T: int, allow_cash: bool = True):
    cfg_ep = globalsetting(**env_cfg.__dict__)
    cfg_ep.seed = int(seed)
    env = GBMBandEnvMulti(cfg=cfg_ep, R=R)
    obs = env.reset(beta=beta, lam=lam_cost, target_ret=target_ann, w0=None)

    target_ann_eff = float(env.target_ret_ann)
    m_star = mv_center_qp(env.Cov, cfg_ep.sigmas, beta, target_ann_eff, allow_cash=allow_cash)

    rsimple = []
    A = clamp01_vec(m_star)
    B = np.maximum(A + 1e-6, m_star)
    for _ in range(T):
        obs, _, done, r_simple = env.step(A, B, use_trade_penalty=True)
        rsimple.append(float(r_simple))
        if done:
            break
    return np.array(rsimple, dtype=float)


def run_episode_mv_buyhold(env_cfg, R, beta, lam_cost, target_ann, *, seed: int, T: int, allow_cash: bool = True):
    cfg_ep = globalsetting(**env_cfg.__dict__)
    cfg_ep.seed = int(seed)
    env = GBMBandEnvMulti(cfg=cfg_ep, R=R)

    target_ann_eff = float(target_ann)
    m_star = mv_center_qp(env.Cov, cfg_ep.sigmas, beta, target_ann_eff, allow_cash=allow_cash)
    obs = env.reset(beta=beta, lam=lam_cost, target_ret=target_ann, w0=m_star)

    rsimple = []
    A = np.zeros(cfg_ep.N_ASSETS, dtype=float)
    B = np.ones(cfg_ep.N_ASSETS, dtype=float)
    for _ in range(T):
        obs, _, done, r_simple = env.step(A, B, use_trade_penalty=True)
        rsimple.append(float(r_simple))
        if done:
            break
    return np.array(rsimple, dtype=float)


def run_episode_rl_a2(env_cfg, R, policy, beta, lam_cost, target_ann, *, seed: int, T: int, device, force_s_one: bool = False):
    cfg_ep = globalsetting(**env_cfg.__dict__)
    cfg_ep.seed = int(seed)
    env = GBMBandEnvMulti(cfg=cfg_ep, R=R)
    obs = env.reset(beta=beta, lam=lam_cost, target_ret=target_ann, w0=None)

    target_ann_eff = float(env.target_ret_ann)
    m_star = mv_center_qp(env.Cov, cfg_ep.sigmas, beta, target_ann_eff, allow_cash=True)

    rsimple = []
    for _ in range(T):
        o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if force_s_one:
            s = np.ones(cfg_ep.N_ASSETS, dtype=float)
        else:
            s_t, _, _ = policy.sample_s_only(o)
            s = s_t.squeeze(0).detach().cpu().numpy()

        minm = np.minimum(m_star, 1.0 - m_star)
        lam_scalar = float(obs[cfg_ep.N_ASSETS * 5 + 0])
        delta = compute_delta_box(
            w_star=m_star,
            Cov=env.Cov,
            gamma=float(getattr(cfg_ep, "RISK_GAMMA", 1.0)),
            lam=lam_scalar,
            scale=1.0,
            clip=(0.0, 1.0),
        )
        b = 0.95 * s * delta * minm
        A = clamp01_vec(m_star - b)
        B = np.maximum(A + 1e-6, m_star + b)

        obs, _, done, r_simple = env.step(A, B, use_trade_penalty=True)
        rsimple.append(float(r_simple))
        if done:
            break
    return np.array(rsimple, dtype=float)


def run_episode_rl_b(env_cfg, R, policy, beta, lam_cost, target_ann, *, seed: int, T: int, device, force_s_one: bool = False):
    cfg_ep = globalsetting(**env_cfg.__dict__)
    cfg_ep.seed = int(seed)
    env = GBMBandEnvMulti(cfg=cfg_ep, R=R)
    obs = env.reset(beta=beta, lam=lam_cost, target_ret=target_ann, w0=None)

    target_ann_eff = float(env.target_ret_ann)
    m_star = mv_center_qp(env.Cov, cfg_ep.sigmas, beta, target_ann_eff, allow_cash=True)

    U, delta_z = compute_delta_rotated(
        m_star=m_star,
        Cov=env.Cov,
        gamma=float(getattr(cfg_ep, "RISK_GAMMA", 1.0)),
        lam=lam_cost,
        scale=1.0,
    )

    rsimple = []
    for _ in range(T):
        o = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        if force_s_one:
            s = np.ones(cfg_ep.N_ASSETS, dtype=float)
        else:
            s_t, _, _ = policy.sample_s_only(o)
            s = s_t.squeeze(0).detach().cpu().numpy()

        b_z = 0.95 * s * delta_z
        obs, _, done, r_simple = env.step_rotated_box(
            m=m_star,
            U=U,
            b_z=b_z,
            allow_cash=True,
            solver="OSQP",
            use_trade_penalty=True,
        )
        rsimple.append(float(r_simple))
        if done:
            break
    return np.array(rsimple, dtype=float)


# -----------------------------
# main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--a2_ckpt", type=str, required=True)
    p.add_argument("--ft_b_ckpt", type=str, default="")
    p.add_argument("--save_dir", type=str, default="eval_v3")
    p.add_argument("--run_name", type=str, default="frontier_v3")
    p.add_argument("--targets", type=str, default="0.02,0.04,0.06,0.08")
    p.add_argument("--lam", type=float, default=0.99)
    p.add_argument("--n_eps", type=int, default=32)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    if args.dry_run:
        args.n_eps = 2

    run_id = f"{args.run_name}_{now_str()}_seed{args.seed}"
    out_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    gcfg = globalsetting(seed=args.seed, device=device)
    if args.dry_run:
        gcfg.T_days = min(gcfg.T_days, 8)

    R = build_corr_from_pairs(
        gcfg.N_ASSETS,
        base_rho=0.20,
        pair_rhos=getattr(gcfg, "pair_rhos", {}),
        make_psd=True,
    )

    policy_a2 = JointBandPolicy(gcfg.N_ASSETS, d_model=128, nlayers=2, nhead=4, use_cash_softmax=True).to(device)
    payload = load_checkpoint(args.a2_ckpt, device)
    policy_a2.load_state_dict(payload["state_dict"], strict=True)
    policy_a2.eval()

    policy_ft = None
    if args.ft_b_ckpt:
        policy_ft = JointBandPolicy(gcfg.N_ASSETS, d_model=128, nlayers=2, nhead=4, use_cash_softmax=True).to(device)
        payload_ft = load_checkpoint(args.ft_b_ckpt, device)
        policy_ft.load_state_dict(payload_ft["state_dict"], strict=True)
        policy_ft.eval()

    targets = [float(x) for x in args.targets.split(",") if x.strip() != ""]

    strategies = ["MV_center", "MV_buyhold", "RL_A2", "RL_B", "s=1"]
    if policy_ft is not None:
        strategies.append("FT_B")

    raw: Dict[str, Dict[float, List[Dict[str, Tuple[float, float]]]]] = {s: {} for s in strategies}
    res_arith: Dict[str, Dict[float, Tuple[float, float]]] = {s: {} for s in strategies}
    res_geom: Dict[str, Dict[float, Tuple[float, float]]] = {s: {} for s in strategies}

    for t_ann in targets:
        cases = make_cases(args.n_eps, base_seed=args.seed, n_assets=gcfg.N_ASSETS, beta_low=-0.95, beta_high=0.95)
        for s in strategies:
            raw[s][t_ann] = []

        for (seed, beta) in cases:
            rs_mv = run_episode_mv_center(gcfg, R, beta, args.lam, t_ann, seed=seed, T=gcfg.T_days)
            raw["MV_center"][t_ann].append({
                "arith": ann_arith_mean_vol_from_rsimple(rs_mv),
                "geom": ann_geom_mean_vol_from_rsimple(rs_mv),
            })

            rs_bh = run_episode_mv_buyhold(gcfg, R, beta, args.lam, t_ann, seed=seed, T=gcfg.T_days)
            raw["MV_buyhold"][t_ann].append({
                "arith": ann_arith_mean_vol_from_rsimple(rs_bh),
                "geom": ann_geom_mean_vol_from_rsimple(rs_bh),
            })

            rs_a2 = run_episode_rl_a2(gcfg, R, policy_a2, beta, args.lam, t_ann, seed=seed, T=gcfg.T_days, device=device)
            raw["RL_A2"][t_ann].append({
                "arith": ann_arith_mean_vol_from_rsimple(rs_a2),
                "geom": ann_geom_mean_vol_from_rsimple(rs_a2),
            })

            rs_b = run_episode_rl_b(gcfg, R, policy_a2, beta, args.lam, t_ann, seed=seed, T=gcfg.T_days, device=device)
            raw["RL_B"][t_ann].append({
                "arith": ann_arith_mean_vol_from_rsimple(rs_b),
                "geom": ann_geom_mean_vol_from_rsimple(rs_b),
            })

            rs_s1 = run_episode_rl_a2(gcfg, R, policy_a2, beta, args.lam, t_ann, seed=seed, T=gcfg.T_days, device=device, force_s_one=True)
            raw["s=1"][t_ann].append({
                "arith": ann_arith_mean_vol_from_rsimple(rs_s1),
                "geom": ann_geom_mean_vol_from_rsimple(rs_s1),
            })

            if policy_ft is not None:
                rs_ft = run_episode_rl_b(gcfg, R, policy_ft, beta, args.lam, t_ann, seed=seed, T=gcfg.T_days, device=device)
                raw["FT_B"][t_ann].append({
                    "arith": ann_arith_mean_vol_from_rsimple(rs_ft),
                    "geom": ann_geom_mean_vol_from_rsimple(rs_ft),
                })

        # summarize
        for s in strategies:
            ptsA = np.array([d["arith"] for d in raw[s][t_ann]], float)
            ptsG = np.array([d["geom"] for d in raw[s][t_ann]], float)
            res_arith[s][t_ann] = (float(ptsA[:, 0].mean()), float(ptsA[:, 1].mean()))
            res_geom[s][t_ann] = (float(ptsG[:, 0].mean()), float(ptsG[:, 1].mean()))

    info = {
        "targets": targets,
        "lam": args.lam,
        "n_eps": args.n_eps,
        "seed": args.seed,
    }

    npz_path = os.path.join(out_dir, "frontier_raw.npz")
    np.savez(npz_path, raw=raw, res_arith=res_arith, res_geom=res_geom, info=info)

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for s in strategies:
        xs = [res_arith[s][t][1] for t in targets]
        ys = [res_arith[s][t][0] for t in targets]
        axes[0].plot(xs, ys, marker="o", label=s)
    axes[0].set_xlabel("Annualized Volatility (arith)")
    axes[0].set_ylabel("Annualized Mean (arith)")
    axes[0].set_title("Frontier (Arithmetic)")
    axes[0].legend(fontsize=8)

    for s in strategies:
        xs = [res_geom[s][t][1] for t in targets]
        ys = [res_geom[s][t][0] for t in targets]
        axes[1].plot(xs, ys, marker="o", label=s)
    axes[1].set_xlabel("Annualized Volatility (geom)")
    axes[1].set_ylabel("Annualized Mean (geom)")
    axes[1].set_title("Frontier (Geometric)")
    axes[1].legend(fontsize=8)

    png_path = os.path.join(out_dir, "frontier.png")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)

    print(f"[DONE] saved: {npz_path}")
    print(f"[DONE] saved: {png_path}")


if __name__ == "__main__":
    main()
