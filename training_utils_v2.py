import torch
import torch.nn as nn
from PPO_agent_v2 import JointBandPolicy, GLOBAL_DIM
from GBMEnv_mv import globalsetting
from RLopt_helpers import build_corr_from_pairs
import numpy as np

def warmup_joint_beta_to_m(policy: JointBandPolicy, steps=800, bs=8192, lr=5e-4, lam_value=1.0, cfg:globalsetting=None, R: np.ndarray=None,
                        use_cvxpy_teacher: bool=True,
                        teacher_frac: float=0.25,
                        teacher_every: int=10,device="cpu",log_every: int = 50,):
    if cfg is None:
        raise ValueError("warmup_joint_beta_to_m: cfg(globalsetting) must be provided")
    device = cfg.device
    # ---- ensure float32 everywhere (critical) ----
    f32 = torch.float32
    sig_t = torch.as_tensor(cfg.sigmas, dtype=f32, device=device).view(1, -1)  # [1,N]
    r_t   = torch.tensor(float(cfg.r), dtype=f32, device=device)
    inv_target_scale = torch.tensor(1.0/0.20, dtype=f32, device=device)


    opt = torch.optim.Adam([p for n,p in policy.named_parameters() if "log_std" not in n], lr=lr)
    bce = nn.BCEWithLogitsLoss()  # ロジット vs 目標m（sigmoid内部でlogit化）

    policy.train()

    def _head_m_weight(policy):
        # head_m が Linear or Sequential どちらでも対応
        for m in policy.head_m.modules():
            if isinstance(m, nn.Linear):
                return m.weight
        raise RuntimeError("head_m Linear layer not found")

    prev_head_m_w = _head_m_weight(policy).detach().clone()



    # Lazy import: keeps runtime ok if cvxpy not installed and teacher disabled
    if use_cvxpy_teacher:
        from mv_teacher import mv_teacher_batch

    # cache teacher samples to amortize QP cost
    cached_m_star = None
    cached_steps_left = 0
    cached_teacher_stats = dict(
        fallback_rate=np.nan, maxw_mean=np.nan,
        effN_inv_mean=np.nan, effN_inv_med=np.nan,
        cash_mean=np.nan, cash_rate=np.nan
    )

    for _ in range(steps):
        # 合成バッチ
        beta = torch.empty(bs, cfg.N_ASSETS, device=device, dtype=f32).uniform_(-0.95, 0.95)
        #w    = torch.rand(bs, cfg.N_ASSETS, device=device)
        #w    = (w / (w.sum(dim=1, keepdim=True)+1e-12)) * torch.rand(bs,1, device=device)  # sum<=1
        w    = torch.zeros(bs, cfg.N_ASSETS, device=device)
        w    = w.to(dtype=f32)
        lam  = torch.full((bs,), float(lam_value), device=device, dtype=f32)
        #target_annual = torch.full((bs,), float(cfg.target_annual_return), device=device)
        excess = (sig_t**2) * beta  # [B,N] float32
        rhs_max = torch.clamp(excess.max(dim=1).values, min=0.0)                # [B]
        u = torch.rand(bs, device=device, dtype=f32) * 0.9
        target_annual = r_t + u * rhs_max

        if R is None:
            R_use = build_corr_from_pairs(cfg.N_ASSETS, base_rho=0.20, pair_rhos=cfg.pair_rhos, make_psd=True)
        else:
            R_use = R

        # 観測ベクトル N*5+4 を生成
        with torch.no_grad():
            #sigma = torch.tensor(cfg.sigmas, dtype=torch.float32, device=device).unsqueeze(0).expand(bs, -1)
            sigma = sig_t.expand(bs, -1)
            R_t   = torch.tensor(R_use,      dtype=torch.float32, device=device)                  # [N,N]
            Rw    = torch.matmul(w, R_t.T)                                               # [B,N]
            per   = torch.stack([beta, w, sigma, Rw, lam.view(-1,1).expand(-1,cfg.N_ASSETS)], dim=-1)  # [B,N,5]
            per_f = per.view(bs, -1)                                                     # [B,N*5]
            Dsig  = torch.diag_embed(sigma)                                              # [B,N,N]
            Cov   = torch.matmul(torch.matmul(Dsig, R_t.expand(bs,-1,-1)), Dsig)         # [B,N,N]
            port_var = (w.unsqueeze(1) @ Cov @ w.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # [B]
            mean_sigma = sigma.mean(dim=1)
            rw_norm = Rw.norm(dim=1)
            #target_scaled = target_annual / 0.20
            target_scaled = target_annual * inv_target_scale
            glob = torch.stack([lam, mean_sigma, port_var, rw_norm, target_scaled], dim=1)  # [B,5]

            # sanity check（任意）
            assert glob.shape[1] == GLOBAL_DIM, f"glob dim {glob.shape[1]} != GLOBAL_DIM {GLOBAL_DIM}"

            obs = torch.cat([per_f, glob], dim=1)                                        # [B,N*5+GLOBAL_DIM]


        # 前向き
        cls_out, tok_out = policy.body(per, glob)     # per=[B,N,5], glob=[B,GLOBAL_DIM]
        m_mu = policy.head_m(tok_out).squeeze(-1)     # ロジット

        # ---- NEW: CVX teacher (minimum variance with target return) ----
        if use_cvxpy_teacher:
            if cached_m_star is None or cached_steps_left <= 0:
                # solve QP only for subset to reduce cost
                # fallback init (0,1)
                m_star = ((beta + 0.95)/(2*0.95)).clamp(1e-4, 1-1e-4)
                # --- 重要：教師を feasible(sum<=1) にする ---
                ssum = m_star.sum(dim=1, keepdim=True)
                scale = torch.clamp(ssum, min=1.0)
                m_star = (m_star / scale).clamp(1e-4, 1-1e-4)

                # pick subset indices
                k = max(1, int(bs * float(teacher_frac)))
                idx = torch.randperm(bs, device=device)[:k]
                # move to numpy for cvxpy
                Cov_np = Cov[idx].detach().cpu().numpy()
                beta_np = beta[idx].detach().cpu().numpy()
                sig_np  = np.asarray(cfg.sigmas, dtype=float)
                tgt_np  = target_annual[idx].detach().cpu().numpy()
                #W_star = mv_teacher_batch(
                #    Cov_batch=Cov_np,
                #    beta_batch=beta_np,
                #    sigmas=sig_np,
                #    r=float(cfg.r),
                #    target_annual_batch=tgt_np,
                #    allow_cash=True
                #)  # [k,N]
                W_star, info = mv_teacher_batch(
                    Cov_batch=Cov_np,
                    beta_batch=beta_np,
                    sigmas=sig_np,
                    r=float(cfg.r),
                    target_annual_batch=tgt_np,
                    allow_cash=True
                    , return_info=True)
                fallback_rate = float(info["fallback"].mean())
                maxw_mean     = float(np.mean(info["maxw"]))

                effN_inv = info["effN_inv"]
                effN_ok = effN_inv[~np.isnan(effN_inv)]
                cached_teacher_stats.update(
                    fallback_rate=float(info["fallback"].mean()),
                    maxw_mean=float(np.mean(info["maxw"])),
                    effN_inv_mean=float(np.mean(effN_ok)) if effN_ok.size else np.nan,
                    effN_inv_med=float(np.median(effN_ok)) if effN_ok.size else np.nan,
                    cash_mean=float(np.mean(info["cash"])),
                    cash_rate=float(np.mean(info["cash"] > 1 - 1e-6)),
                )

                W_star_t = torch.as_tensor(W_star, dtype=f32, device=device)
                m_star[idx] = W_star_t.clamp(1e-4, 1-1e-4)

                cached_m_star = m_star
                cached_steps_left = teacher_every
            else:
                m_star = cached_m_star
            cached_steps_left -= 1
        else:
            # fallback: beta monotone map
            m_star = ((beta + 0.95)/(2*0.95)).clamp(1e-4, 1-1e-4)            
            ssum = m_star.sum(dim=1, keepdim=True)
            m_star = (m_star / torch.clamp(ssum, min=1.0)).clamp(1e-4, 1-1e-4)


        # 合計≤1のソフト制約（ロジットをsigmoidしてから）
        m_sig  = torch.sigmoid(m_mu)
        excess = torch.clamp(m_sig.sum(dim=1) - 1.0, min=0.0)
        simplex_pen = 1e-2 * (excess**2).mean()

        #loss = bce(m_mu, m_star) + simplex_pen
        loss = bce(m_mu, m_star)
        opt.zero_grad(); loss.backward()
        # grad norm（head_mのみ）
        head_m_w = _head_m_weight(policy)
        grad_norm = (
            head_m_w.grad.norm().item()
            if head_m_w.grad is not None else 0.0
        )
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        # ---- logging ----
        if (_ % log_every == 0) or (_ == steps-1):
            with torch.no_grad():
                # teacher stats
                m_mean = float(m_star.mean().item())
                m_std  = float(m_star.std().item())
                m_sum_mean = float(m_star.sum(dim=1).mean().item())

                # head_m weight update size
                dw = (_head_m_weight(policy) - prev_head_m_w).abs().mean().item()
                prev_head_m_w.copy_(_head_m_weight(policy))
    
                # beta と m_star の相関（各資産ごと）を平均
                b = beta.detach()
                ms = m_star.detach()
                b0 = b - b.mean(dim=0, keepdim=True)
                m0 = ms - ms.mean(dim=0, keepdim=True)
                corr = (b0*m0).mean(dim=0) / (b0.std(dim=0)*m0.std(dim=0) + 1e-12)
                corr_mean = float(corr.mean().item())
            with torch.no_grad():
                m_pred = torch.sigmoid(m_mu)
                b0 = beta - beta.mean(dim=0, keepdim=True)
                p0 = m_pred - m_pred.mean(dim=0, keepdim=True)
                corr_bp = (b0*p0).mean(dim=0) / (b0.std(dim=0)*p0.std(dim=0) + 1e-12)
                corr_bp_mean = float(corr_bp.mean().item())

            print(
                f"[warmup {_:04d}] "
                f"loss={loss.item():.4e} | "
                f"m* mean/std=({m_mean:.3f},{m_std:.3f}) | "
                f"sum(m*) mean={m_sum_mean:.3f} | "
                f"grad||={grad_norm:.2e} | "
                f"Δhead_m={dw:.2e} | "
                f"fallback={fallback_rate:.2%} | "
                f"maxw={maxw_mean:.2f} | "
                f"corr(β,m*)={corr_mean:.2f}"
                f" | corr(β,ĥm)={corr_bp_mean:.2f}"
            )
            ts = cached_teacher_stats
            print(
            f"... | fallback={ts['fallback_rate']:.2%} | "
            f"maxw={ts['maxw_mean']:.2f} | "
            f"effN_inv(mean/med)=({ts['effN_inv_mean']:.2f},{ts['effN_inv_med']:.2f}) | "
            f"cash_mean={ts['cash_mean']:.2f} | cash_rate={ts['cash_rate']:.2%}"
            )


def freeze_width_head_in_stage1(policy: JointBandPolicy):
    for n,p in policy.named_parameters():
        if 'head_s' in n or 'log_std_s' in n:
            p.requires_grad_(False)

def freeze_centers_in_stage2(policy: JointBandPolicy):
    for n, p in policy.named_parameters():
        if 'head_m' in n or 'log_std_m' in n:
            p.requires_grad_(False)
def freeze_stage1_center_and_body(policy):
    """
    Stage1 PPO中に、warmupで作った beta->m を壊さないため
    body と head_m を凍結する。
    （width head_s は stage1 で学習してもよい/しないは別途制御）
    """
    # Transformer body
    for p in policy.body.parameters():
        p.requires_grad_(False)

    # center head
    for n, p in policy.named_parameters():
        if "head_m" in n or "log_std_m" in n:
            p.requires_grad_(False)


def unfreeze_body(policy):
    """Stage2 に入る前に body を学習可能に戻す（center head は別途 freeze_centers_in_stage2 で管理）"""
    for p in policy.body.parameters():
        p.requires_grad_(True)
def make_optimizer_trainable(policy, lr):
    params = [p for p in policy.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr)
