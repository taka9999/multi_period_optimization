#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_B_v3.py
- Fine-tune width head under Level-B executor (rotated box)
- Stage-2 style: freeze centers, update widths only
"""

import os
import json
import time
import copy
import random
import argparse
from dataclasses import asdict
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.optim as optim

from PPO_agent_v3 import JointBandPolicy, ValueNetCLS, PPOConfig
from PPO_update_v3 import ppo_update_joint
from Rollout_HJB_box_v3 import (
    rollout_joint_levelB,
    mv_center_qp,
    compute_delta_rotated,
)
from GBMEnv_LQ_v4 import globalsetting, GBMBandEnvMulti
from RLopt_helpers import build_corr_from_pairs
from training_utils_LQ_v3 import freeze_centers_in_stage2


# -----------------------------
# utils
# -----------------------------

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def save_checkpoint(path: str, *, policy, valuef, meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "state_dict": policy.state_dict(),
        "value_state_dict": valuef.state_dict(),
        "meta": meta,
    }
    torch.save(payload, path)


def load_checkpoint(path: str, device: torch.device):
    payload = torch.load(path, map_location=device)
    meta = payload.get("meta", {})
    gdim = meta.get("global_dim", None)
    if gdim is not None and int(gdim) != 5:
        raise ValueError(f"checkpoint global_dim mismatch: expected 5, got {gdim}")
    return payload


def make_meta(gcfg, cfg: PPOConfig, extra: Dict[str, Any]) -> Dict[str, Any]:
    meta = {
        "version": "RLBand-v3-finetuneB",
        "timestamp": now_str(),
        "N_ASSETS": int(gcfg.N_ASSETS),
        "per_asset_dim": 5,
        "global_dim": 5,
        "device": str(gcfg.device),
        "env_cfg": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in gcfg.__dict__.items()
                    if not k.startswith("_")},
        "ppo_cfg": asdict(cfg) if hasattr(cfg, "__dict__") else {},
    }
    meta.update(extra)
    return meta


def _snapshot_trainable_params(model):
    return {name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad}


def _param_update_norm(prev, model) -> float:
    sq = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad:
            d = (p.detach() - prev[name]).float()
            sq += float((d * d).sum().cpu())
    return float(np.sqrt(sq))


def sample_s_stats(policy, obs_batch, device):
    if obs_batch.ndim == 1:
        obs_batch = obs_batch.unsqueeze(0)
    obs_batch = obs_batch.to(device)
    if hasattr(policy, "sample_s_only"):
        s_t, _, _ = policy.sample_s_only(obs_batch)
        s = s_t.detach().cpu().numpy()
    else:
        _, s_t, _, _, _ = policy.sample_stage2(obs_batch)
        s = s_t.detach().cpu().numpy()
    return {
        "s_min": float(np.min(s)),
        "s_mean": float(np.mean(s)),
        "s_max": float(np.max(s)),
        "s_std": float(np.std(s)),
    }


def make_fixed_eval_cases(
    sigmas,
    *,
    base_seed: int = 777,
    n_cases: int = 32,
    beta_low: float = -0.95,
    beta_high: float = 0.95,
    lam_choices: Tuple[float, ...] = (0.99, 0.995),
    frac_choices: Tuple[float, ...] = (0.3, 0.5, 0.7),
) -> List[Tuple[int, np.ndarray, float, float]]:
    sig2 = np.asarray(sigmas, float) ** 2
    cases = []
    for k in range(n_cases):
        seed = base_seed + k
        rng = np.random.default_rng(seed)
        beta = rng.uniform(beta_low, beta_high, size=len(sig2))
        lam = float(rng.choice(lam_choices))
        mu_eff = sig2 * beta
        mu_max = float(mu_eff.max())
        frac = float(rng.choice(frac_choices))
        target = max(0.0, frac * mu_max)
        cases.append((seed, beta, lam, target))
    return cases


def run_episode_levelB(
    env_cfg,
    R,
    policy,
    beta,
    lam_cost,
    target_ann,
    *,
    seed: int,
    T: int,
    device,
    mv_allow_cash: bool,
    mv_solver: str = "OSQP",
    qp_solver: str = "OSQP",
    force_s_one: bool = False,
):
    cfg_ep = copy.deepcopy(env_cfg)
    cfg_ep.seed = int(seed)
    env = GBMBandEnvMulti(cfg=cfg_ep, R=R)
    obs = env.reset(beta=beta, lam=lam_cost, target_ret=target_ann, w0=None)

    target_ann_eff = float(env.target_ret_ann)
    m_star = mv_center_qp(
        env.Cov,
        cfg_ep.sigmas,
        beta,
        target_ann_eff,
        allow_cash=mv_allow_cash,
        round_nd=4,
        solver=mv_solver,
    )

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
            allow_cash=mv_allow_cash,
            solver=qp_solver,
            use_trade_penalty=True,
        )
        rsimple.append(float(r_simple))
        if done:
            break
    return np.array(rsimple, dtype=float)


def eval_levelB_fixed_cases(
    gcfg,
    R,
    policy,
    cases,
    *,
    device,
    mv_allow_cash: bool,
    mv_solver: str = "OSQP",
    qp_solver: str = "OSQP",
    force_s_one: bool = False,
) -> Dict[str, float]:
    vals = []
    T_days = gcfg.T_days
    for (seed, beta, lam, target_ann) in cases:
        rs = run_episode_levelB(
            gcfg,
            R,
            policy,
            beta,
            lam,
            target_ann,
            seed=seed,
            T=T_days,
            device=device,
            mv_allow_cash=mv_allow_cash,
            mv_solver=mv_solver,
            qp_solver=qp_solver,
            force_s_one=force_s_one,
        )
        if rs.size == 0:
            continue
        vals.append(float(np.sum(np.log1p(rs))))

    if not vals:
        return {"n_ok": 0, "mean": np.nan, "std": np.nan}
    return {"n_ok": len(vals), "mean": float(np.mean(vals)), "std": float(np.std(vals))}


# -----------------------------
# main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--ckpt", type=str, default="policy_stage2_A2.pt")
    p.add_argument("--save_dir", type=str, default="runs_v3")
    p.add_argument("--run_name", type=str, default="B_finetune_v3")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch_episodes", type=int, default=48)
    p.add_argument("--minibatch_size", type=int, default=8192)
    p.add_argument("--lr_actor", type=float, default=5e-5)
    p.add_argument("--lr_critic", type=float, default=5e-4)
    p.add_argument("--clip_ratio", type=float, default=0.15)
    p.add_argument("--entropy_coef", type=float, default=0.01)
    p.add_argument("--width_prior_w", type=float, default=0.0)
    p.add_argument("--lam_choices", type=str, default="0.99,0.995")
    p.add_argument("--target_choices", type=str, default="0.02,0.04,0.06,0.08")
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--log_jsonl", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    set_all_seeds(args.seed)

    if args.dry_run:
        args.epochs = 1
        args.batch_episodes = 2
        args.minibatch_size = 256
        args.save_every = 1

    run_id = f"{args.run_name}_{now_str()}_seed{args.seed}"
    out_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    gcfg = globalsetting(
        seed=args.seed,
        device=device,
        N_ASSETS=5,
        DISCOUNT_BY_BANK=True,
        INIT_W0_UNIFORM=True,
        BAND_SMOOTH_COEF=0.0,
        TRADE_PEN_COEF=0.4,
        ALPHA=1 / 5,
        STAGE1_WIDTH_COEF=0.05,
    )
    if args.dry_run:
        gcfg.T_days = min(gcfg.T_days, 8)

    R_default = build_corr_from_pairs(
        gcfg.N_ASSETS,
        base_rho=0.20,
        pair_rhos=getattr(gcfg, "pair_rhos", {}),
        make_psd=True,
    )

    cfg = PPOConfig(
        horizon=gcfg.T_days,
        gamma=1.0,
        gae_lambda=1.0,
        batch_episodes=args.batch_episodes,
        epochs=4,
        minibatch_size=args.minibatch_size,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        lr_actor2=args.lr_actor,
        lr_critic2=args.lr_critic,
        clip_ratio=args.clip_ratio,
        entropy_coef=args.entropy_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
    )

    policy = JointBandPolicy(gcfg.N_ASSETS, d_model=128, nlayers=2, nhead=4, use_cash_softmax=True).to(device)
    valuef = ValueNetCLS(gcfg.N_ASSETS, d_model=128, nlayers=2, nhead=4).to(device)

    payload = load_checkpoint(args.ckpt, device)
    policy.load_state_dict(payload["state_dict"], strict=True)
    valuef.load_state_dict(payload["value_state_dict"], strict=True)

    freeze_centers_in_stage2(policy)

    opt_pi = optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=cfg.lr_actor2)
    opt_v = optim.Adam(valuef.parameters(), lr=cfg.lr_critic2)

    lam_choices = [float(x) for x in args.lam_choices.split(",") if x.strip() != ""]
    target_choices = [float(x) for x in args.target_choices.split(",") if x.strip() != ""]

    fixed_cases = make_fixed_eval_cases(gcfg.sigmas, n_cases=(4 if args.dry_run else 32))

    jsonl_path = os.path.join(out_dir, "finetune_log.jsonl") if args.log_jsonl else None

    def log_row(row: Dict[str, Any]):
        row = dict(row)
        row["t"] = time.time()
        row["seed"] = args.seed
        row["run_id"] = run_id
        print(row)
        if jsonl_path is not None:
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(row) + "\n")

    for ep in range(1, args.epochs + 1):
        prev = _snapshot_trainable_params(policy)

        batch = rollout_joint_levelB(
            policy,
            valuef,
            cfg,
            gcfg=gcfg,
            lam_choices=lam_choices,
            target_choices=target_choices,
            batch_episodes=cfg.batch_episodes,
            R=R_default,
        )

        ppo_update_joint(
            policy,
            valuef,
            opt_pi,
            opt_v,
            cfg,
            batch,
            env_cfg=gcfg,
            stage=2,
            width_prior_w=args.width_prior_w,
        )

        update_norm = _param_update_norm(prev, policy)
        s_stats = sample_s_stats(policy, batch["obs"], device)
        eval_out = eval_levelB_fixed_cases(
            gcfg,
            R_default,
            policy,
            fixed_cases,
            device=device,
            mv_allow_cash=True,
        )

        log_row({
            "epoch": ep,
            "rew_ep_mean": float(batch.get("rew_ep_mean", np.nan)),
            "update_norm": update_norm,
            **s_stats,
            "fixed_eval_mean": float(eval_out.get("mean", np.nan)),
            "fixed_eval_std": float(eval_out.get("std", np.nan)),
            "fixed_eval_n": int(eval_out.get("n_ok", 0)),
        })

        if ep % args.save_every == 0:
            ckpt_path = os.path.join(out_dir, f"policy_stage2_B_v3_ep{ep}.pt")
            meta = make_meta(gcfg, cfg, extra={"ckpt": os.path.basename(ckpt_path)})
            save_checkpoint(ckpt_path, policy=policy, valuef=valuef, meta=meta)

    final_path = os.path.join(out_dir, "policy_stage2_B_v3_final.pt")
    meta = make_meta(gcfg, cfg, extra={"ckpt": os.path.basename(final_path)})
    save_checkpoint(final_path, policy=policy, valuef=valuef, meta=meta)
    log_row({"phase": "done", "final_ckpt": final_path})


if __name__ == "__main__":
    main()
