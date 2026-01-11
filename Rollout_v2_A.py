from __future__ import annotations
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
from GBMEnv_mv import GBMBandEnvMulti,globalsetting
from RLopt_helpers import build_corr_from_pairs, build_cov, clamp01_vec
from PPO_agent_v2 import JointBandPolicy, ValueNetCLS, PPOConfig, GLOBAL_DIM
from mv_teacher import mv_teacher_batch

def compute_gae(rews, vals, dones, gamma, lam):
    T = len(rews)
    adv = torch.zeros(T)
    lastgaelam = 0.0
    nextvalue = 0.0
    nextnonterm = 0.0
    for t in reversed(range(T)):
        delta = rews[t] + gamma * nextvalue * nextnonterm - vals[t]
        lastgaelam = delta + gamma * lam * nextnonterm * lastgaelam
        adv[t] = lastgaelam
        nextvalue = vals[t].item()
        nextnonterm = 0.0 if bool(dones[t]) else 1.0
    ret = adv + vals
    return adv, ret

def make_env(env_cfg: globalsetting, R: np.ndarray) -> GBMBandEnvMulti:
    return GBMBandEnvMulti(
        env_cfg.T_days,
        env_cfg.sigmas,
        R,
        env_cfg.r,
        env_cfg.dt_day,
        env_cfg.DISCOUNT_BY_BANK,
        rng_seed=np.random.randint(1, 10_000_000),
        cfg=env_cfg,
    )

def _stage1_only_beta_obs(o: torch.Tensor, sigmas_np: np.ndarray) -> torch.Tensor:
    """
    Stage-1 用に観測を β のみに差し替える。
    形式は [N*5 + GLOBAL_DIM] を保ちつつ、w=0, (R@w)=0, lam=1, port_var=0 等のダミーを詰める。
    """
    B, D = o.shape
    N = (D - GLOBAL_DIM) // 5
    device_ = o.device
    # 元の per-asset 部のβだけ抽出
    beta = o[:, :N]                                # [B,N]
    sigma = torch.tensor(sigmas_np, dtype=torch.float32, device=device_).unsqueeze(0).expand(B, -1)
    zeros = torch.zeros_like(beta)                 # w=0, Rw=0
    ones  = torch.ones(B, 1, device=device_)       # lam=1

    # per-asset = [beta_i, w_i(0), sigma_i, (R@w)_i(0), lam(=1)]
    per = torch.stack([beta, zeros, sigma, zeros, ones.expand(-1, N)], dim=-1)  # [B,N,5]
    per_f = per.view(B, -1)

    # global: [lam, mean_sigma, port_var, ||Rw||, target_scaled]
    glob = torch.tensor(
        [1.0, float(np.mean(sigmas_np)), 0.0, 0.0, 0.0], dtype=torch.float32, device=device_
    ).view(1, 5).expand(B, GLOBAL_DIM)

    return torch.cat([per_f, glob], dim=1)         # [B, N*5+GLOBAL_DIM]

@torch.no_grad()
def rollout_joint(policy: JointBandPolicy,
                  valuef: ValueNetCLS,
                  cfg: PPOConfig,
                  *,
                  gcfg:globalsetting = None,
                  lam_choices: List[float],
                  stage: int,
                  batch_episodes: int = None,
                  R: np.ndarray = None,
                  target_choices: List[float] = None,
                  center_mode: str = "policy",  # "policy" or "teacher"
                  ):
    """
    Stage-1: 観測は β のみに差し替え（m 学習を安定化）、m/s とも確率化でもOK（推奨は m だけでも可）
    Stage-2: m は決定論（deterministic_m=True）、s のみ確率化（stochastic_s=True）、
             logp は s 項のみ（policy.logprob_s_only）
    """
    be = cfg.batch_episodes if batch_episodes is None else batch_episodes
    if R is None:
        raise ValueError("rollout_joint: R must be provided (correlation matrix).")

    N_ASSETS = gcfg.N_ASSETS

    obs_buf, m_buf, s_buf, logp_buf, adv_buf, ret_buf = [], [], [], [], [], []
    rew_ep = []

    for _ in range(be):
        # 環境初期化
        beta = np.random.uniform(-0.95, 0.95, size=N_ASSETS)
        lam  = 1.0 if stage == 1 else float(np.random.choice(lam_choices))
        
        # target return（フロンティアを作るならここを複数で回す）
        if target_choices is None:
            target_annual = float(getattr(gcfg, "target_annual_return", 0.0))
        else:
            target_annual = float(np.random.choice(target_choices))
        env  = GBMBandEnvMulti(cfg.horizon, gcfg.sigmas, R, gcfg.r, gcfg.dt_day,
                               gcfg.DISCOUNT_BY_BANK, rng_seed=np.random.randint(1,10_000_000), cfg=gcfg)
        obs  = env.reset(beta=beta, lam=lam, w0=None, target_annual_return=target_annual)
        # ---- 方針A対応: center を teacher から供給（Stage-2のみ推奨） ----
        teacher_m = None
        if (center_mode == "teacher") and (stage == 2):
            Cov = np.outer(gcfg.sigmas, gcfg.sigmas) * R
            # teacher は (beta, target_annual) から中心ウェイトを生成（cash許容・long-only）
            teacher_m = mv_teacher_batch(
                Cov_batch=Cov[None, :, :],
                beta_batch=beta[None, :],
                sigmas=gcfg.sigmas,
                r=gcfg.r,
                target_annual_batch=np.array([target_annual], dtype=float),
                allow_cash=True,
                return_info=False,
            )[0]  # shape (N,)
            teacher_m = clamp01_vec(teacher_m)

        ep_obs, ep_m, ep_s, ep_lp, ep_val, ep_rew, ep_done = [], [], [], [], [], [], []

        for t in range(env.T):
            # ---- Stage-1 / Stage-2 共通ループ内 ----
            o = torch.tensor(obs, dtype=torch.float32, device=gcfg.device).unsqueeze(0)

            # Stage-1 はβのみ
            o_stage = _stage1_only_beta_obs(o, gcfg.sigmas) if stage == 1 else o

            with torch.no_grad():
                v_t = valuef(o_stage)

            # 行動サンプル（両ヘッド出るが、Stage-1 は m だけ使う）
            if stage == 1:
                # これまで通り：m だけ確率化（logp は m-only）
                m_t, s_t, _lp_unused, w_cash, m_pre, s_pre = policy.sample(o_stage, deterministic=False)
                logp_use = policy.logprob_m_only(o_stage, m_pre)
            else:
                # Stage-2: 基本は s のみ確率化（logp は s-only）
                m_pol, s_t, _, m_pre_pol, s_pre = policy.sample_stage2(o_stage)
                if teacher_m is not None:
                    # center は teacher を使用（方針A）
                    m_t = torch.tensor(teacher_m, dtype=torch.float32, device=gcfg.device).unsqueeze(0)
                    m_pre = m_t  # Stage-2 の PPO では m は使わないのでダミーでOK
                else:
                    m_t, m_pre = m_pol, m_pre_pol
                logp_use = policy.logprob_s_only(o_stage, s_pre)

            # numpy へ（環境へ渡すため）
            m = m_t.squeeze(0).detach().cpu().numpy()
            # s は Stage-1 では使わない

            # 幅の決め方
            minm = np.minimum(m, 1.0 - m)
            if stage == 1:
                b = gcfg.STAGE1_WIDTH_COEF * 0.95 * minm                 # ★ 固定極小幅
            else:
                lam_scalar = float(obs[N_ASSETS*5 + 0])
                g = (max(0.0, 1.0 - lam_scalar) + 1e-8)**gcfg.ALPHA
                s = s_t.squeeze(0).detach().cpu().numpy()
                b = 0.95 * s * g * minm

            A = clamp01_vec(m - b)
            B = np.maximum(A + 1e-6, m + b)

            obs, r, done, _ = env.step(A, B, use_trade_penalty=(stage == 2))

            # ---- バッファ投入（detach/CPU）----
            ep_obs.append(o.squeeze(0).detach().cpu())
            ep_m.append(m_pre.squeeze(0).detach().cpu())     # 射影前シグモイド（PPOで使用）
            ep_s.append(s_pre.squeeze(0).detach().cpu())
            ep_lp.append(logp_use.squeeze(0).detach().cpu())
            ep_val.append(v_t.squeeze(0).detach().cpu())
            ep_rew.append(float(r)); ep_done.append(done)

            if done:
                break

        if len(ep_rew) == 0:
            continue

        rew_ep.append(sum(ep_rew))

        # ---- テンソル化 ----
        obs_ep = torch.stack(ep_obs)
        m_ep   = torch.stack(ep_m)
        s_ep   = torch.stack(ep_s)
        lp_ep  = torch.stack(ep_lp)
        val_ep = torch.stack(ep_val)
        rew_t  = torch.tensor(ep_rew, dtype=torch.float32)
        done_t = torch.tensor(ep_done, dtype=torch.bool)

        adv_ep, ret_ep = compute_gae(rew_t, val_ep, done_t, cfg.gamma, cfg.gae_lambda)
        obs_buf.append(obs_ep); m_buf.append(m_ep); s_buf.append(s_ep)
        logp_buf.append(lp_ep); adv_buf.append(adv_ep); ret_buf.append(ret_ep)

    # ---- 連結 & 標準化 ----
    if len(obs_buf) == 0:
        raise RuntimeError("rollout_joint: collected 0 episodes (buffer empty). "
                           "Check env.reset/step and observation layout.")

    obs = torch.cat(obs_buf).to(gcfg.device)
    m   = torch.cat(m_buf).to(gcfg.device)
    s   = torch.cat(s_buf).to(gcfg.device)
    logp= torch.cat(logp_buf).to(gcfg.device)
    adv = torch.cat(adv_buf).to(gcfg.device)
    ret = torch.cat(ret_buf).to(gcfg.device)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    return dict(
        obs=obs, m=m, s=s, logp=logp, adv=adv, ret=ret,
        rew_ep_mean=float(np.mean(rew_ep)), rew_ep_std=float(np.std(rew_ep)),
    )