from __future__ import annotations

import numpy as np
import torch
import cvxpy as cp
from typing import List, Optional, Dict, Tuple

from GBMEnv_LQ_v3 import GBMBandEnvMulti, globalsetting
from RLopt_helpers import clamp01_vec
from PPO_agent import JointBandPolicy, ValueNetCLS, PPOConfig


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


# --- MV center solver (cached) ---
_MV_CACHE: Dict[Tuple[Tuple[float, ...], float, bool], np.ndarray] = {}

def mv_center_qp(Cov: np.ndarray, sigmas: np.ndarray, beta: np.ndarray, target_ann: float, *, allow_cash: bool = True, round_nd: int = 4, solver: str = "OSQP") -> np.ndarray:
    """
    Long-only MV center (discounted-by-bank world):
      mu_eff = (sigmas**2) * beta
      min w^T Cov w
      s.t. mu_eff^T w >= target_ann, w>=0, sum(w)<=1 (cash) or ==1 (no cash).
    Target is clipped to max(mu_eff) to avoid infeasibility.
    """
    beta = np.asarray(beta, float).reshape(-1)
    sigmas = np.asarray(sigmas, float).reshape(-1)
    Cov = np.asarray(Cov, float)
    mu_eff = (sigmas**2) * beta
    max_mu = float(np.max(mu_eff))
    targ = float(min(float(target_ann), max_mu)) if np.isfinite(max_mu) else float(target_ann)
    key = (tuple(np.round(beta, round_nd).tolist()), float(np.round(targ, round_nd)), bool(allow_cash))
    if key in _MV_CACHE:
        return _MV_CACHE[key].copy()
    n = len(mu_eff)
    w = cp.Variable(n)
    obj = cp.Minimize(cp.quad_form(w, Cov))
    cons = [w >= 0, mu_eff @ w >= targ]
    cons += [cp.sum(w) <= 1.0] if allow_cash else [cp.sum(w) == 1.0]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver=getattr(cp, solver), verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)
    if w.value is None:
        ww = np.zeros(n, float)
        ww[int(np.argmax(mu_eff))] = 1.0
    else:
        ww = np.maximum(0.0, np.array(w.value, dtype=float).reshape(-1))
        s = float(ww.sum())
        if allow_cash:
            if s > 1.0 + 1e-10:
                ww /= max(s, 1e-12)
        else:
            if s <= 1e-12:
                ww[:] = 0.0
                ww[int(np.argmax(mu_eff))] = 1.0
            else:
                ww /= s
    ww = clamp01_vec(ww)
    _MV_CACHE[key] = ww.copy()
    return ww


@torch.no_grad()
def rollout_joint(policy: JointBandPolicy, valuef: ValueNetCLS, cfg: PPOConfig, *, gcfg: globalsetting, lam_choices: List[float], target_choices: Optional[List[float]] = None, stage: int = 2, batch_episodes: Optional[int] = None, R: Optional[np.ndarray] = None, mv_allow_cash: bool = True, mv_round_nd: int = 4, mv_solver: str = "OSQP") -> Dict[str, torch.Tensor]:
    """
    MV center is computed outside the policy; policy outputs width only (s).
    Returned batch matches PPO_update expectations: obs, m, s, logp, adv, ret.
    """
    if R is None:
        raise ValueError("rollout_joint: R must be provided")
    be = cfg.batch_episodes if batch_episodes is None else int(batch_episodes)
    N = gcfg.N_ASSETS

    obs_buf, m_buf, s_buf, logp_buf, adv_buf, ret_buf = [], [], [], [], [], []
    rew_ep = []

    for _ in range(be):
        beta = np.random.uniform(-0.95, 0.95, size=N)
        lam = 1.0 if stage == 1 else float(np.random.choice(lam_choices))
        target_ann = (float(np.random.choice(target_choices)) if (target_choices is not None and len(target_choices) > 0) else float(getattr(gcfg, "TARGET_RET_ANN", 0.06)))
        target_ann = max(target_ann, 0.0)

        env = make_env(gcfg, R)
        obs = env.reset(beta=beta, lam=lam, target_ret=target_ann, w0=None)
        # Align to env-internal target (avoids mismatch)
        target_ann_eff = float(env.target_ret_ann)
        m_star = mv_center_qp(env.Cov, gcfg.sigmas, beta, target_ann_eff, allow_cash=mv_allow_cash, round_nd=mv_round_nd, solver=mv_solver)

        ep_obs, ep_m, ep_s, ep_lp, ep_val, ep_rew, ep_done = [], [], [], [], [], [], []
        for t in range(env.T):
            o = torch.tensor(obs, dtype=torch.float32, device=gcfg.device).unsqueeze(0)
            v_t = valuef(o)

            if stage == 1:
                # stage1: fixed tiny width; no PPO on s (logp=0)
                s_np = np.full(N, 0.5, dtype=float)
                s_pre = torch.tensor(s_np, dtype=torch.float32, device=gcfg.device).unsqueeze(0)
                logp_use = torch.zeros(1, device=gcfg.device)
            else:
                s_t, logp_use, s_pre = policy.sample_s_only(o)
                s_np = s_t.squeeze(0).detach().cpu().numpy()

            m = m_star
            minm = np.minimum(m, 1.0 - m)
            if stage == 1:
                b = gcfg.STAGE1_WIDTH_COEF * 0.95 * minm
            else:
                lam_scalar = float(obs[N*5 + 0])
                g = (max(0.0, 1.0 - lam_scalar) + 1e-8) ** gcfg.ALPHA
                b = 0.95 * s_np * g * minm

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