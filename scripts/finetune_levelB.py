import json
import argparse
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import cvxpy as cp
import torch

from src.ppo.agent import JointBandPolicy, ValueNetCLS, PPOConfig
from src.ppo.update import ppo_update_joint
from src.utils.training_utils import freeze_centers_in_stage2
from src.ppo.rollout import rollout_joint_levelB, rollout_eval_levelB

from src.regime_gbm.gbm_env import globalsetting
from src.regime_gbm.regime_gbm_env import RegimeGBMBandEnvMulti
from src.utils.rlopt_helpers import build_corr_from_pairs


# ------------------------------
# Domain randomization utilities
# ------------------------------
def make_market_sampler(
    gcfg: globalsetting,
    *,
    rho_range: tuple[float, float] = (-0.2, 0.8),
    pair_rho_noise: float = 0.15,
    sigma_logn_std: float = 0.20,
    sigma_clip: tuple[float, float] = (0.05, 0.60),
):
    """Return a callable market_sampler(rng, k) -> (R, sigmas).

    - Samples a correlation matrix via your `build_corr_from_pairs(..., make_psd=True)`.
    - Samples per-asset vol as log-normal around the base `gcfg.sigmas`.

    This sampler is designed to be plugged into Rollout_HJB_box_B2.rollout_* via
    the `market_sampler=` keyword.
    """
    N = int(gcfg.N_ASSETS)
    base_sigmas = np.asarray(gcfg.sigmas, float).reshape(-1)
    base_pairs = getattr(gcfg, "pair_rhos", None)

    def _sampler(rng: np.random.Generator, k: int):
        # --- R ---
        base_rho = float(rng.uniform(rho_range[0], rho_range[1]))
        if base_pairs is None:
            pair_rhos = None
        else:
            # perturb each provided pair rho, then clip
            pair_rhos = {}
            for key, val in dict(base_pairs).items():
                pv = float(val) + float(rng.normal(0.0, pair_rho_noise))
                pair_rhos[key] = float(np.clip(pv, -0.95, 0.95))

        R = build_corr_from_pairs(N, base_rho=base_rho, pair_rhos=pair_rhos, make_psd=True)

        # --- sigmas ---
        mult = np.exp(rng.normal(0.0, sigma_logn_std, size=N))
        sigmas = np.clip(base_sigmas * mult, sigma_clip[0], sigma_clip[1])
        return R, sigmas

    return _sampler

def _trainable_param_stats(model):
    n_train = 0
    n_total = 0
    for p in model.parameters():
        n_total += p.numel()
        if p.requires_grad:
            n_train += p.numel()
    return n_train, n_total

def _grad_norm(model):
    # L2 norm over trainable parameters (after backward)
    sq = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            g = p.grad.detach()
            sq += float((g*g).sum().cpu())
    return float(np.sqrt(sq))

def _snapshot_trainable_params(model):
    # clone trainable params only (cheap enough)
    return {name: p.detach().clone()
            for name, p in model.named_parameters()
            if p.requires_grad}

def _param_update_norm(prev, model):
    # L2 norm of (current - prev) over trainable params
    sq = 0.0
    for name, p in model.named_parameters():
        if p.requires_grad:
            d = (p.detach() - prev[name]).float()
            sq += float((d*d).sum().cpu())
    return float(np.sqrt(sq))

@torch.no_grad()
def sample_s_stats(policy, obs_batch, device):
    """
    obs_batch: torch.Tensor [B, obs_dim] on device
    Returns dict of s stats.
    """
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
    sigmas, *, base_seed=777, n_cases=32,
    beta_low=-0.95, beta_high=0.95,
    lam_choices=(0.99, 0.995),
    frac_choices=(0.3, 0.5, 0.7),
):
    sig2 = np.asarray(sigmas, float)**2
    cases = []
    for k in range(n_cases):
        seed = base_seed + k
        rng = np.random.default_rng(seed)
        beta = rng.uniform(beta_low, beta_high, size=len(sig2))
        lam  = float(rng.choice(lam_choices))

        mu_eff = sig2 * beta
        mu_max = float(mu_eff.max())
        frac   = float(rng.choice(frac_choices))
        target = max(0.0, frac * mu_max)  # reachable by construction

        cases.append((seed, beta, lam, target))
    return cases

def maybe_load_regimes(regime_json_path: Optional[str]) -> Tuple[Optional[List[Dict[str, Any]]], Optional[np.ndarray]]:
     """
     regime_json は eval_regime_gbm_final_fix4.py と同じ形式を想定:
       {
         "P": [[...],[...]],
         "regimes": [
            {"beta": [...], "sigmas":[...], "R":[[...],[...]]}, ...
         declare initial_dist optional
       }
     """
     if regime_json_path is None:
         return None, None
     with open(regime_json_path, "r") as f:
         data = json.load(f)
     regimes = data.get("regimes", None)
     if regimes is None:
         raise ValueError("regime_json must contain key 'regimes'")
     P = np.asarray(data.get("P", None), float)
     if P is None or P.ndim != 2:
         raise ValueError("regime_json must contain 2D 'P'")
     return regimes, P
 
def make_regime_env_ctor(regimes: List[Dict[str, Any]], P: np.ndarray, gcfg: globalsetting):
    """
    rollout_joint_levelB に渡す env_ctor を作る。
    R は env 内で regime ごとに切り替わるので rollout 側からはダミーでOK。
    """
    # R は必須引数なので regime[0] の R を使う（なければ I）
    R0 = regimes[0].get("R", None)
    if R0 is None:
        n = len(regimes[0]["sigmas"])
        R0 = np.eye(n)
    else:
         R0 = np.asarray(R0, float)
    def _ctor():
         return RegimeGBMBandEnvMulti(cfg=gcfg, regimes=regimes, P=P, R=R0)
    return _ctor

def mv_weights_target_return(
    Cov, mu_eff, target_ann,
    allow_cash=True,
    solver="OSQP",
    infeasible_policy="fallback",  # "skip" or "fallback"
):
    """
    min w^T Cov w
    s.t. mu_eff^T w >= target_ann, w>=0, sum(w)<=1 (cash allowed)

    infeasible_policy:
      - "skip": return None if target_ann > max(mu_eff) (not achievable under long-only+cash)
      - "fallback": invest 100% in argmax(mu_eff)
    """
    mu_eff = np.asarray(mu_eff, float)
    n = len(mu_eff)
    mu_max = float(mu_eff.max())

    if target_ann > mu_max + 1e-12:
        if infeasible_policy == "skip":
            return None
        else:
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

def compute_levelB_basis_and_width(m_star, Cov, gamma, lam_scalar, eps=1e-12, scale=1.0):
    """
    Returns (U, delta_z) where:
      a* = Q Cov Q,  a* = U diag(eig) U^T
      delta_z_k ∝ (kappa*eig_k / Gamma_kk)^{1/3},  Gamma_kk ≈ gamma*(U^T Cov U)_kk
    """
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

    delta_z = scale * np.power((kappa * eigvals) / Gamma_kk, 1.0/3.0)
    delta_z = np.maximum(delta_z, 0.0)
    return U, delta_z

def run_episode_RL_band_levelB_v2(env_cfg, R, policy, beta, lam_cost, target_ann, seed=2025, T=None, device="cpu",
                               infeasible_policy="skip", mv_solver="OSQP", qp_solver="OSQP",
                               mv_allow_cash=False, force_s_one: bool = False, return_step :bool = True):
    """
    Level B executor evaluation:
      - compute MV center m_star (same as A2)
      - build rotated basis U and z-width prior delta_z (once per episode)
      - per step: get s (or force s≡1), set b_z = 0.95*s*delta_z, then env.step_rotated_box(...)
    Returns: rsimple array or None if infeasible
    """
    cfg2 = env_cfg
    cfg2.seed = int(seed)

    env = GBMBandEnvMulti(cfg=cfg2, R=R)
    N = cfg2.N_ASSETS
    w0 = np.full(N, 1.0 / N) * 0.8
    obs = env.reset(beta=beta, lam=lam_cost, target_ret=target_ann, w0=w0)

    target_ann_eff = float(env.target_ret_ann)
    mu_eff = (cfg2.sigmas**2) * beta
    m_star = mv_weights_target_return(env.Cov, mu_eff, target_ann_eff, allow_cash=mv_allow_cash,
                                      solver=mv_solver, infeasible_policy=infeasible_policy)
    if m_star is None:
        return None

    T = env.T if T is None else int(T)
    rsimple = []
    rstep = []

    lam_scalar = float(env.lam.mean()) if isinstance(env.lam, np.ndarray) else float(env.lam)
    U, delta_z = compute_levelB_basis_and_width(
        m_star=m_star,
        Cov=env.Cov,
        gamma=float(getattr(cfg2, "RISK_GAMMA", 1.0)),
        lam_scalar=lam_scalar,
        scale=1.0,
    )

    for _ in range(T):
        o = np.array(obs, dtype=np.float32)

        # get s from policy (same as A2), or force s≡1
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

        b_z = 0.95 * s * delta_z

        obs, r_step, done, r_simple = env.step_rotated_box(
            m=m_star, U=U, b_z=b_z,
            allow_cash=mv_allow_cash,
            solver=qp_solver,
            use_trade_penalty=True
        )
        rsimple.append(float(r_simple))
        rstep.append(float(r_step))
        if done:
            break
    if return_step:
        return np.array(rsimple), np.array(rstep)
    else:
        return np.array(rsimple)

def eval_levelB_fixed_cases(
    gcfg, R, policy, cases, *, device, mv_solver="OSQP", qp_solver="OSQP",
    infeasible_policy="fallback", mv_allow_cash=False, force_s_one=False,
):
    rews = []
    lrets = []
    T_days = gcfg.T_days
    for (seed, beta, lam, target_ann) in cases:
        rs, rsteps = run_episode_RL_band_levelB_v2(
            gcfg, R, policy, beta, lam, target_ann,
            seed=seed, T=T_days, device=device,
            infeasible_policy=infeasible_policy, mv_solver=mv_solver,
            qp_solver=qp_solver, mv_allow_cash=mv_allow_cash,
            force_s_one=force_s_one,
            return_step=True,
        )
        if rs is None:
            continue
        # ここはあなたの定義に合わせて（例：logret の合計など）でOK
        rews.append(float(np.sum(rsteps)))              # 例：簡易
        lrets.append(float(np.sum(np.log1p(rs))))   # 例：簡易

    out = {
        "n_ok": len(rews),
        "rew_ep_mean": float(np.mean(rews)) if rews else np.nan,
        "rew_ep_std":  float(np.std(rews))  if rews else np.nan,
        "lret_ep_mean": float(np.mean(lrets)) if lrets else np.nan,
        "lret_ep_std":  float(np.std(lrets))  if lrets else np.nan,
    }
    return out

def finetune_stage2_levelB(
    policy: JointBandPolicy,
    valuef: ValueNetCLS,
    cfg: PPOConfig,
    gcfg: globalsetting,
    R: np.ndarray,
    lam_choices: list[float],
    target_choices: list[float] | None = None,
    fine_tune_epochs: int = 8,          # ★ 5〜10 ならここ
    width_prior_w: float = 0.02,
    mv_allow_cash: bool = False,
    mv_solver: str = "OSQP",
    qp_solver: str = "OSQP",
    topk: int | None = None,
    # --- domain randomization ---
    domain_randomize: bool = True,
    market_sampler=None,
    base_seed: int = 1234,
    env_ctor=None,
):
    device = gcfg.device
    policy.to(device); valuef.to(device)

    # stage2: center head freeze (s headだけ動かす)
    freeze_centers_in_stage2(policy)

    opt_pi = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=cfg.lr_actor2)
    opt_v  = torch.optim.Adam(valuef.parameters(), lr=cfg.lr_critic2)

    # --- domain randomization sampler (used during training rollouts) ---
    if domain_randomize:
        if market_sampler is None:
            market_sampler = make_market_sampler(gcfg)
    else:
        market_sampler = None

    # snapshot base sigmas for fixed-eval (and to restore after randomized eval)
    _base_sigmas0 = np.asarray(gcfg.sigmas, float).reshape(-1).copy()

    # default sampler (if requested)
    if domain_randomize and market_sampler is None:
        market_sampler = make_market_sampler(gcfg)

    fixed_cases = make_fixed_eval_cases(
    sigmas = gcfg.sigmas,
    base_seed=999,
    n_cases=32,
    lam_choices=(0.99, 0.995),
    frac_choices=(0.3, 0.5, 0.7),
    )

    # (optional) fine-tune 前のB評価
    #   - fixed: (R fixed, sigmas fixed)
    #   - randomized: sample (R, sigmas) once and evaluate under that market
    gcfg.sigmas = _base_sigmas0.copy()
    eval0_fixed = rollout_eval_levelB(
        policy, cfg,
        gcfg=gcfg,
        lam_choices=lam_choices,
        target_choices=target_choices,
        batch_episodes=16,
        R=R,
        mv_allow_cash=mv_allow_cash,
        mv_solver=mv_solver,
        qp_solver=qp_solver,
        topk=topk,
        env_ctor=env_ctor,
    )

    if market_sampler is None:
        eval0_rand = dict(eval0_fixed)
    else:
        rng_eval0 = np.random.default_rng(int(base_seed) + 424242)
        R0, sig0 = market_sampler(rng_eval0, 0)
        gcfg.sigmas = np.asarray(sig0, float).reshape(-1).copy()
        eval0_rand = rollout_eval_levelB(
            policy, cfg,
            gcfg=gcfg,
            lam_choices=lam_choices,
            target_choices=target_choices,
            batch_episodes=16,
            R=R0,
            mv_allow_cash=mv_allow_cash,
            mv_solver=mv_solver,
            qp_solver=qp_solver,
            topk=topk,
            env_ctor=env_ctor,
        )

    # restore
    gcfg.sigmas = _base_sigmas0.copy()
    print("[Before FT] B-eval fixed:", eval0_fixed)
    print("[Before FT] B-eval randomized:", eval0_rand)

    # --- sanity: trainable params ---
    n_train, n_total = _trainable_param_stats(policy)
    print(f"[Sanity] trainable_params={n_train:,} / total={n_total:,}")

    # --- initial snapshot for update-norm tracking ---
    prev_params = _snapshot_trainable_params(policy)

    for k in range(1, fine_tune_epochs + 1):

        # ---- collect batch ----
        batch = rollout_joint_levelB(
            policy, valuef, cfg,
            gcfg=gcfg,
            lam_choices=lam_choices,
            target_choices=target_choices,
            batch_episodes=cfg.batch_episodes,
            R=None if market_sampler is not None else R,
            market_sampler=market_sampler,
            base_seed=base_seed,
            seed_offset=100000 * k,
            mv_allow_cash=mv_allow_cash,
            mv_solver=mv_solver,
            qp_solver=qp_solver,
            topk=topk,
        )

        # ---- s statistics on the rollout batch obs (cheap) ----
        # batch["obs"] is [T_total, obs_dim] on device already
        obs = batch["obs"]
        # subsample to avoid overhead (e.g. 2048 rows)
        B = min(obs.shape[0], 2048)
        idx = torch.randint(0, obs.shape[0], (B,), device=obs.device)
        s_stats = sample_s_stats(policy, obs[idx], device=obs.device)

        # ---- update policy/value with PPO ----
        # (we also want grad/update norms; we measure PARAM update before/after)
        prev_params_epoch = _snapshot_trainable_params(policy)

        ppo_update_joint(
            policy, valuef, opt_pi, opt_v, cfg, batch,
            env_cfg=gcfg, stage=2, width_prior_w=width_prior_w
        )

        # ---- parameter update magnitude (did we actually change?) ----
        upd_norm = _param_update_norm(prev_params_epoch, policy)

        # ---- optional: quick eval (fixed + randomized) ----
        # fixed
        gcfg.sigmas = _base_sigmas0.copy()
        if env_ctor is None:
             evalk_fixed = rollout_eval_levelB(
                 policy, cfg,
                 gcfg=gcfg,
                 lam_choices=lam_choices,
                 target_choices=target_choices,
                 batch_episodes=16,
                 R=R,
                 mv_allow_cash=mv_allow_cash,
                 mv_solver=mv_solver,
                 qp_solver=qp_solver,
                 topk=topk,
             )
        else:
             # NOTE: rollout_eval_levelB は env_ctor を受け取れない実装なので regime モードではスキップ
             evalk_fixed = rollout_eval_levelB(
                policy, cfg,
                gcfg=gcfg,
                lam_choices=lam_choices,
                target_choices=target_choices,
                batch_episodes=16,
                R=R,  # env_ctor を使うならダミーでもOK
                mv_allow_cash=mv_allow_cash,
                mv_solver=mv_solver,
                qp_solver=qp_solver,
                topk=topk,
                env_ctor=env_ctor,  # ★追加
            )

        # randomized (one sampled market per epoch k)
        if market_sampler is None:
            evalk_rand = dict(evalk_fixed)
        else:
            rng_evalk = np.random.default_rng(int(base_seed) + 900000 + k)
            Rk, sigk = market_sampler(rng_evalk, k)
            gcfg.sigmas = np.asarray(sigk, float).reshape(-1).copy()
            evalk_rand = rollout_eval_levelB(
                policy, cfg,
                gcfg=gcfg,
                lam_choices=lam_choices,
                target_choices=target_choices,
                batch_episodes=16,
                R=Rk,
                mv_allow_cash=mv_allow_cash,
                mv_solver=mv_solver,
                qp_solver=qp_solver,
                topk=topk,
                env_ctor=env_ctor,
            )

        # restore
        gcfg.sigmas = _base_sigmas0.copy()
        print(
            f"[FT {k:02d}/{fine_tune_epochs}] "
            f"rollout_rew_mean={batch['rew_ep_mean']:.4f} "
            f"| upd_norm={upd_norm:.3e} "
            f"| s: (min/mean/max/std)=({s_stats['s_min']:.3f}/{s_stats['s_mean']:.3f}/{s_stats['s_max']:.3f}/{s_stats['s_std']:.3f}) "
            f"| B-eval fixed={evalk_fixed} | randomized={evalk_rand}"
        )

    return policy, valuef

if __name__ == "__main__":
    gcfg = globalsetting()
    gcfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ap = argparse.ArgumentParser()
    ap.add_argument("--regime_json", type=str, default=None, help="path to regime json (K=2 etc). If provided, use RegimeGBM env.")
    ap.add_argument("--policy_in", type=str, default="policy_stage2_A2.pt")
    ap.add_argument("--value_in",  type=str, default="value_stage2_A2.pt")
    ap.add_argument("--policy_out", type=str, default="policy_stage2_B2_2_finetuned.pt")
    ap.add_argument("--value_out",  type=str, default="value_stage2_B2_2_finetuned.pt")
    args = ap.parse_args()

    regimes, P = maybe_load_regimes(args.regime_json)
    if regimes is None:
        # non-regime: Corr matrix (same as your training setup)
        R = build_corr_from_pairs(gcfg.N_ASSETS, base_rho=0.20, pair_rhos=gcfg.pair_rhos, make_psd=True)
        env_ctor = None
    else:
        # regime: env_ctor injects RegimeGBMBandEnvMulti
        env_ctor = make_regime_env_ctor(regimes, P, gcfg)
        # Rollout 側の API 的に R が必要な箇所があるのでダミーを渡す
        R = np.eye(gcfg.N_ASSETS, dtype=float)

    cfg = PPOConfig()
    cfg.batch_episodes = 32          # B rollout重いのでまず小さめ推奨
    cfg.epochs = 4
    cfg.minibatch_size = 4096

    policy = JointBandPolicy(gcfg.N_ASSETS, d_model=128, nlayers=2, nhead=4, use_cash_softmax=True)
    valuef = ValueNetCLS(gcfg.N_ASSETS, d_model=128, nlayers=2, nhead=4)
    policy.load_state_dict(torch.load(args.policy_in, map_location=gcfg.device))
    valuef.load_state_dict(torch.load(args.value_in, map_location=gcfg.device))

    lam_choices = [0.995]  # 例（あなたの実験に合わせて）
    target_choices = None

    finetune_stage2_levelB(
        policy, valuef, cfg, gcfg, R,
        lam_choices=lam_choices,
        target_choices=target_choices,
        fine_tune_epochs=8,      # ★ここを5〜10で
        width_prior_w=0.02,
        mv_allow_cash=False,
        mv_solver="OSQP",
        qp_solver="OSQP",
        topk = 2,
        env_ctor=env_ctor,
        # regime env では market_sampler による相関/σランダム化はまず切って動作確認
        domain_randomize=False if env_ctor is not None else True,
        market_sampler=None,
    )

    # fine-tuned weights save
    torch.save(policy.state_dict(), args.policy_out)
    torch.save(valuef.state_dict(), args.value_out)