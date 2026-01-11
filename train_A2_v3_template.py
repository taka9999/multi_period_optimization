#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_A2_v3.py
- A2 training (stage2 main; stage1 optional)
- v3 observation (global_dim=5) + per_asset_dim=5
- R randomized per-episode (policy sees only summary via global feats)
"""

import os
import json
import time
import random
import argparse
from dataclasses import asdict
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.optim as optim

# ===== v3 modules (to be created by Codex) =====
from PPO_agent_v3 import JointBandPolicy, ValueNetCLS, PPOConfig
from PPO_update_v3 import ppo_update_joint
from Rollout_HJB_box_v3 import rollout_joint  # A2 executor rollout
from GBMEnv_LQ_v4 import globalsetting         # env cfg with global_dim=5 obs
from RLopt_helpers import build_corr_from_pairs, build_cov

# ===== optional training utils you already have =====
from training_utils_LQ_v3 import freeze_centers_in_stage2  # keep center frozen in stage2


# -----------------------------
# utils
# -----------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # determinism (optional)
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
    return payload


def make_meta(gcfg, cfg: PPOConfig, extra: Dict[str, Any]) -> Dict[str, Any]:
    # keep it JSON-serializable as much as possible
    meta = {
        "version": "RLBand-v3",
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


# -----------------------------
# main
# -----------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # reproducibility / io
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")  # auto/cpu/cuda
    p.add_argument("--save_dir", type=str, default="runs_v3")
    p.add_argument("--run_name", type=str, default="A2_v3")
    p.add_argument("--save_every", type=int, default=2)

    # training schedule
    p.add_argument("--stage1", action="store_true")  # optional
    p.add_argument("--stage1_updates", type=int, default=0)
    p.add_argument("--stage2_updates", type=int, default=12)   # per lam-stage
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_episodes", type=int, default=48)
    p.add_argument("--minibatch_size", type=int, default=8192)

    # PPO hypers
    p.add_argument("--lr_actor", type=float, default=1e-4)
    p.add_argument("--lr_critic", type=float, default=5e-4)
    p.add_argument("--clip_ratio", type=float, default=0.15)
    p.add_argument("--entropy_coef", type=float, default=0.012)
    p.add_argument("--width_prior_w", type=float, default=0.02)

    # env sampling
    p.add_argument("--lam_choices", type=str, default="0.99,0.995,1.0")
    p.add_argument("--target_choices", type=str, default="0.02,0.04,0.06,0.08")
    p.add_argument("--corr_mode", type=str, default="random", choices=["fixed", "random"])
    p.add_argument("--base_rho_low", type=float, default=-0.2)
    p.add_argument("--base_rho_high", type=float, default=0.5)
    p.add_argument("--n_pair_edges", type=int, default=2)  # random pair_rhos per episode

    # logging
    p.add_argument("--log_jsonl", action="store_true")  # write JSONL logs

    return p.parse_args()


def main():
    args = parse_args()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    set_all_seeds(args.seed)

    # run dir
    run_id = f"{args.run_name}_{now_str()}_seed{args.seed}"
    out_dir = os.path.join(args.save_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    # save args
    with open(os.path.join(out_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # env cfg (v3 obs spec is inside globalsetting / env)
    gcfg = globalsetting(
        seed=args.seed,
        device=device,
        N_ASSETS=5,  # adjust if needed or add arg
        DISCOUNT_BY_BANK=True,
        INIT_W0_UNIFORM=True,
        BAND_SMOOTH_COEF=0.0,
        TRADE_PEN_COEF=0.4,
        ALPHA=1/5,
        STAGE1_WIDTH_COEF=0.05,
        # sigmas/pair_rhos can also be args; keep minimal here
    )

    # Build base Cov once (env will sample R per episode if corr_mode="random")
    # If your env build_cov uses (sigmas, R), you might keep a default R for init only
    R_default = build_corr_from_pairs(
        gcfg.N_ASSETS, base_rho=0.20, pair_rhos=getattr(gcfg, "pair_rhos", {}), make_psd=True
    )
    Cov_default = build_cov(sigmas=gcfg.sigmas, R=R_default, make_psd=True)

    # PPO config
    cfg = PPOConfig(
        horizon=gcfg.T_days,
        gamma=1.0,
        gae_lambda=1.0,
        batch_episodes=args.batch_episodes,
        epochs=args.epochs,
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

    # models
    policy = JointBandPolicy(gcfg.N_ASSETS, d_model=128, nlayers=2, nhead=4, use_cash_softmax=True).to(device)
    valuef = ValueNetCLS(gcfg.N_ASSETS, d_model=128, nlayers=2, nhead=4).to(device)

    # opts
    opt_pi = optim.Adam(policy.parameters(), lr=cfg.lr_actor)
    opt_v  = optim.Adam(valuef.parameters(), lr=cfg.lr_critic)

    # parse choices
    lam_choices = [float(x) for x in args.lam_choices.split(",") if x.strip() != ""]
    target_choices = [float(x) for x in args.target_choices.split(",") if x.strip() != ""]

    # log sink
    jsonl_path = os.path.join(out_dir, "train_log.jsonl") if args.log_jsonl else None

    def log_row(row: Dict[str, Any]):
        row = dict(row)
        row["t"] = time.time()
        row["seed"] = args.seed
        row["run_id"] = run_id
        print(row)
        if jsonl_path is not None:
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(row) + "\n")

    # --------------------------
    # Stage 1 (optional) - usually off for now
    # --------------------------
    if args.stage1 and args.stage1_updates > 0:
        policy.use_cash_softmax = False
        # TODO: add your stage1 routine if you want (frictionless warmup)
        for it in range(args.stage1_updates):
            batch = rollout_joint(
                policy, valuef, cfg,
                gcfg=gcfg,
                lam_choices=[1.0],
                target_choices=target_choices,
                stage=1,
                batch_episodes=cfg.batch_episodes,
                R=R_default,
                corr_mode="fixed",  # usually fixed in stage1
                # if rollout supports: corr sampling params
            )
            ppo_update_joint(policy, valuef, opt_pi, opt_v, cfg, batch, env_cfg=gcfg, stage=1, width_prior_w=0.0)
            log_row({"phase": "stage1", "it": it+1, "rew_ep_mean": float(batch.get("rew_ep_mean", np.nan))})

    # --------------------------
    # Stage 2 (A2) - main
    # --------------------------
    policy.use_cash_softmax = True
    freeze_centers_in_stage2(policy)  # stage2: freeze centers, learn widths

    # re-make actor optimizer for trainable params only
    opt_pi = optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=cfg.lr_actor2)
    opt_v  = optim.Adam(valuef.parameters(), lr=cfg.lr_critic2)

    # lam curriculum (minimal; adjust as desired)
    stages = [
        ([0.90],               6, 32, 0.020),
        ([0.95],               8, 32, 0.015),
        ([0.99, 0.995],       10, 48, 0.012),
        ([0.995, 0.999, 1.0], 12, 48, 0.010),
    ]

    for si, (lam_list, updates, be, ent) in enumerate(stages, start=1):
        cfg.entropy_coef = ent
        cfg.batch_episodes = be

        log_row({"phase": "stage2", "stage_i": si, "lam_list": lam_list, "updates": updates, "batch_episodes": be, "entropy": ent})

        for it in range(updates):
            batch = rollout_joint(
                policy, valuef, cfg,
                gcfg=gcfg,
                lam_choices=lam_list,
                target_choices=target_choices,
                stage=2,
                batch_episodes=cfg.batch_episodes,
                R=R_default,                 # used only if corr_mode="fixed"
                corr_mode=args.corr_mode,     # "random" is v3 key
                base_rho_low=args.base_rho_low,
                base_rho_high=args.base_rho_high,
                n_pair_edges=args.n_pair_edges,
            )

            ppo_update_joint(
                policy, valuef, opt_pi, opt_v, cfg, batch,
                env_cfg=gcfg, stage=2, width_prior_w=args.width_prior_w
            )

            # minimal logging
            log_row({
                "phase": "stage2",
                "stage_i": si,
                "it": it+1,
                "rew_ep_mean": float(batch.get("rew_ep_mean", np.nan)),
                "lret_ep_mean": float(batch.get("lret_ep_mean", np.nan)),
            })

            # save periodically
            if (it + 1) % args.save_every == 0:
                ckpt_path = os.path.join(out_dir, f"policy_stage2_A2_v3_stage{si}_it{it+1}.pt")
                meta = make_meta(gcfg, cfg, extra={"ckpt": os.path.basename(ckpt_path), "corr_mode": args.corr_mode})
                save_checkpoint(ckpt_path, policy=policy, valuef=valuef, meta=meta)

    # final save
    final_path = os.path.join(out_dir, "policy_stage2_A2_v3_final.pt")
    meta = make_meta(gcfg, cfg, extra={"ckpt": os.path.basename(final_path), "corr_mode": args.corr_mode})
    save_checkpoint(final_path, policy=policy, valuef=valuef, meta=meta)
    log_row({"phase": "done", "final_ckpt": final_path})

    print(f"[DONE] saved: {final_path}")


if __name__ == "__main__":
    main()
