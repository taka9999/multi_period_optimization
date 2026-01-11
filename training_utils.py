import torch
import torch.nn as nn
from PPO_agent import JointBandPolicy
from GBMEnv import globalsetting
from RLopt_helpers import build_corr_from_pairs
import numpy as np

def warmup_joint_beta_to_m(policy: JointBandPolicy, steps=800, bs=8192, lr=5e-4, lam_value=1.0, cfg:globalsetting=None, R: np.ndarray=None):
    if cfg is None:
        raise ValueError("warmup_joint_beta_to_m: cfg(globalsetting) must be provided")
    device = cfg.device


    opt = torch.optim.Adam([p for n,p in policy.named_parameters() if "log_std" not in n], lr=lr)
    bce = nn.BCEWithLogitsLoss()  # ロジット vs 目標m（sigmoid内部でlogit化）

    policy.train()
    for _ in range(steps):
        # 合成バッチ
        beta = torch.empty(bs, cfg.N_ASSETS, device=device).uniform_(-0.95, 0.95)
        w    = torch.rand(bs, cfg.N_ASSETS, device=device)
        w    = (w / (w.sum(dim=1, keepdim=True)+1e-12)) * torch.rand(bs,1, device=device)  # sum<=1
        lam  = torch.full((bs,), float(lam_value), device=device)

        if R is None:
            R_use = build_corr_from_pairs(cfg.N_ASSETS, base_rho=0.20, pair_rhos=cfg.pair_rhos, make_psd=True)
        else:
            R_use = R

        # 観測ベクトル N*5+4 を生成
        with torch.no_grad():
            sigma = torch.tensor(cfg.sigmas, dtype=torch.float32, device=device).unsqueeze(0).expand(bs, -1)
            R_t   = torch.tensor(R_use,      dtype=torch.float32, device=device)                  # [N,N]
            Rw    = torch.matmul(w, R_t.T)                                               # [B,N]
            per   = torch.stack([beta, w, sigma, Rw, lam.view(-1,1).expand(-1,cfg.N_ASSETS)], dim=-1)  # [B,N,5]
            per_f = per.view(bs, -1)                                                     # [B,N*5]
            Dsig  = torch.diag_embed(sigma)                                              # [B,N,N]
            Cov   = torch.matmul(torch.matmul(Dsig, R_t.expand(bs,-1,-1)), Dsig)         # [B,N,N]
            port_var = (w.unsqueeze(1) @ Cov @ w.unsqueeze(-1)).squeeze(-1).squeeze(-1)  # [B]
            mean_sigma = sigma.mean(dim=1)
            rw_norm = Rw.norm(dim=1)
            glob = torch.stack([lam, mean_sigma, port_var, rw_norm], dim=1)              # [B,4]
            obs = torch.cat([per_f, glob], dim=1)                                        # [B,N*5+4]

        # 前向き
        cls_out, tok_out = policy.body(per, glob)     # per=[B,N,5], glob=[B,4]
        m_mu = policy.head_m(tok_out).squeeze(-1)     # ロジット

        # 目標 m*（フリクションレス目安）
        m_star = ((beta + 0.95)/(2*0.95)).clamp(1e-4, 1-1e-4)

        # 合計≤1のソフト制約（ロジットをsigmoidしてから）
        m_sig  = torch.sigmoid(m_mu)
        excess = torch.clamp(m_sig.sum(dim=1) - 1.0, min=0.0)
        simplex_pen = 1e-2 * (excess**2).mean()

        loss = bce(m_mu, m_star) + simplex_pen
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

def freeze_width_head_in_stage1(policy: JointBandPolicy):
    for n,p in policy.named_parameters():
        if 'head_s' in n or 'log_std_s' in n:
            p.requires_grad_(False)

def freeze_centers_in_stage2(policy: JointBandPolicy):
    for n, p in policy.named_parameters():
        if 'head_m' in n or 'log_std_m' in n:
            p.requires_grad_(False)
