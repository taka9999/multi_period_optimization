from dataclasses import dataclass

import torch
import torch.nn as nn

from src.utils.rlopt_helpers import project_simplex_leq1



@dataclass
class PPOConfig:
    horizon: int = 5*252
    gamma: float = 1.0
    gae_lambda: float = 1.0
    batch_episodes: int = 64
    epochs: int = 4
    minibatch_size: int = 8192
    lr_actor: float = 1e-4
    lr_critic: float = 5e-4
    lr_actor2: float = 1e-4
    lr_critic2: float = 5e-4
    clip_ratio: float = 0.15
    entropy_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

class TokenEncoder(nn.Module):
    def __init__(self, per_dim: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(per_dim, d_model)

    def forward(self, x_tok):  # [B,N,per_dim]
        return self.proj(x_tok)

class JointTransformerBody(nn.Module):
    def __init__(self, N:int, per_dim:int=5, d_model:int=128, nhead:int=4, nlayers:int=2, dropout:float=0.0):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.token_enc = nn.Linear(per_dim, d_model)
        self.glob_proj = nn.Linear(4, d_model, bias=False)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            batch_first=True, dropout=dropout, activation="gelu", norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.cls = nn.Parameter(torch.zeros(1,1,d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, per_tokens: torch.Tensor, global_feats: torch.Tensor=None):
        B = per_tokens.size(0)
        h = self.token_enc(per_tokens)
        cls = self.cls.expand(B,1,self.d_model)
        if global_feats is not None:
            if global_feats.ndim != 2:
                raise RuntimeError(f"global_feats ndim must be 2, got {global_feats.ndim}")
            if global_feats.size(1) != 4:
                global_feats = global_feats[:, -4:]
            g = torch.nn.functional.gelu(self.glob_proj(global_feats))
            cls = cls + g.unsqueeze(1)
        x = torch.cat([cls, h], dim=1)
        x = self.enc(x)
        return x[:,0], x[:,1:]

class JointBandPolicy(nn.Module):
    """
    出力:
      - centers m in [0,1]^N, かつ sum(m) <= 1 になるように射影（又は cash-softmax）
      - widths s in (0,1)^N
    ログ確率は“事前の独立 Sigmoid 変数”に対して計算し、射影は stop-grad で適用（実務で安定）。
    """
    def __init__(self, N:int, d_model:int=128, nlayers:int=2, nhead:int=4, use_cash_softmax:bool=False):
        super().__init__()
        self.N = N
        self.body = JointTransformerBody(N, per_dim=5, d_model=d_model, nhead=nhead, nlayers=nlayers)
        self.head_m = nn.Linear(d_model, 1)   # per-asset
        self.head_s = nn.Linear(d_model, 1)
        self.log_std_m = nn.Parameter(torch.tensor(-0.7))
        self.log_std_s = nn.Parameter(torch.tensor(-0.5))
        self.use_cash_softmax = use_cash_softmax
        # 現金トークン用のヘッド（softmax モードの時のみ使用）
        if use_cash_softmax:
            self.cash_head = nn.Linear(d_model, 1)

    def forward_heads(self, cls_out, tok_out):
        m_mu = self.head_m(tok_out).squeeze(-1)  # [B,N]
        s_mu = self.head_s(tok_out).squeeze(-1)  # [B,N]
        return m_mu, s_mu

    @staticmethod
    def _build_tokens(obs: torch.Tensor, N: int):
        B = obs.size(0)
        per  = obs[:, :N*5].view(B, N, 5).contiguous()
        glob = obs[:, N*5:N*5+4].contiguous()      # ★ここで4次元に固定
        return per, glob

    def sample(self, obs: torch.Tensor, deterministic: bool=False):
        B = obs.size(0); N = self.N
        per, glob = self._build_tokens(obs, N)
        cls_out, tok_out = self.body(per, glob)           # [B,d], [B,N,d]
        m_mu, s_mu = self.forward_heads(cls_out, tok_out)

        std_m = torch.exp(self.log_std_m)
        std_s = torch.exp(self.log_std_s)

        if deterministic:
            raw_m = m_mu
            raw_s = s_mu
            logp = torch.zeros(B, device=obs.device)
        else:
            dist_m = torch.distributions.Normal(m_mu, std_m)
            dist_s = torch.distributions.Normal(s_mu, std_s)
            raw_m = dist_m.rsample()
            raw_s = dist_s.rsample()

        if self.use_cash_softmax:
            # N+1 ロジットから softmax → 現金 w_cash を明示
            cash_logit = self.cash_head(cls_out).squeeze(-1)   # [B]
            logits = torch.cat([raw_m, cash_logit.unsqueeze(1)], dim=1)  # [B,N+1]
            probs = torch.softmax(logits, dim=1)               # sum=1
            m_out = probs[:, :N]                               # ≤1 を自動で満たす
            w_cash = probs[:, N]
            m_pre = raw_m.detach() if not deterministic else raw_m
            if not deterministic:
                logp_m = dist_m.log_prob(raw_m).sum(dim=1)
            else:
                logp_m = torch.zeros(B, device = obs.device)
        else:
            m_i = torch.sigmoid(raw_m).clamp(1e-6, 1-1e-6)
            m_out = project_simplex_leq1(m_i)
            w_cash = 1.0 - m_out.sum(dim=1).clamp(0.0, 1.0)
            m_pre = m_i.detach() if not deterministic else m_i
            if not deterministic:
                logJ_m = -(torch.log(m_i)+torch.log1p(-m_i))
                logp_m = (dist_m.log_prob(raw_m) + logJ_m).sum(dim=1)
            else:
                logp_m = torch.zeros(B, device=obs.device)

        s_i = torch.sigmoid(raw_s).clamp(1e-6, 1-1e-6)
        s_pre = s_i.detach() if not deterministic else s_i
        if not deterministic:
            logJ_s = -(torch.log(s_i)+torch.log1p(-s_i))
            logp_s = (dist_s.log_prob(raw_s) + logJ_s).sum(dim=1)
            logp = logp_m + logp_s
        else:
            logp = torch.zeros(B, device=obs.device)

        return m_out, s_i, logp, w_cash, m_pre, s_pre
    
    @torch.no_grad()
    def sample_s_only(self, obs: torch.Tensor):
        """
        s だけを N(μ_s, σ_s)→sigmoid→(0,1) に写像してサンプル。
        戻り値: s_t(0,1), logp_s(合計), s_pre(0,1)  ※ logprob_s_only と同じ前提
        """
        B = obs.size(0); N = self.N
        per, glob = self._build_tokens(obs, N)
        cls_out, tok_out = self.body(per, glob)

        s_mu = self.head_s(tok_out).squeeze(-1)           # [B,N]
        std_s = torch.exp(self.log_std_s)
        dist_s = torch.distributions.Normal(s_mu, std_s)
        raw_s = dist_s.rsample()                           # [B,N]
        s_i  = torch.sigmoid(raw_s).clamp(1e-6, 1-1e-6)   # (0,1)
        logJ_s = -(torch.log(s_i) + torch.log1p(-s_i))
        logp_s = (dist_s.log_prob(raw_s) + logJ_s).sum(dim=1)  # [B]
        return s_i, logp_s, s_i.detach()
    
    @torch.no_grad()
    def sample_stage2(self, o: torch.Tensor):
        """
        Stage-2 用: m は決定論（平均）で固定、s だけサンプル。
        m の決定論出力は self.sample() と完全に同じ射影規則に従う。
        """
        B = o.size(0); N = self.N
        per, glob = self._build_tokens(o, N)
        cls, tok  = self.body(per, glob)

        # --- m を決定論で ---
        m_mu = self.head_m(tok).squeeze(-1)  # [B,N] ロジット
        if self.use_cash_softmax:
            cash_logit = self.cash_head(cls).squeeze(-1)    # [B]
            logits     = torch.cat([m_mu, cash_logit.unsqueeze(1)], dim=1)  # [B,N+1]
            probs      = torch.softmax(logits, dim=1)       # 合計=1
            m_t        = probs[:, :N]                       # [B,N]
        else:
            m_sig = torch.sigmoid(m_mu).clamp(1e-6, 1-1e-6)
            m_t   = project_simplex_leq1(m_sig)             # sample() と同じ射影

        # --- s を確率化 ---
        s_t, logp_s, s_pre = self.sample_s_only(o)          # 下の変更②参照

        # m_pre は PPO で使わないが、整合性のため返すなら (0,1) に射影後を返す
        m_pre = m_t
        return m_t, s_t, logp_s, m_pre, s_pre
    
    def logprob_m_only(self, obs: torch.Tensor, m_pre: torch.Tensor) -> torch.Tensor:
        """
        Stage-1 用の old-logp 再計算。
        - use_cash_softmax=True: m_pre は『ロジット』を渡す設計にする
         （rollout で m_pre = raw_m.detach() にしている前提）。
        - use_cash_softmax=False: m_pre は (0,1)（sigmoid結果）を渡す。
        """
        B = obs.size(0); N = self.N
        per, glob = self._build_tokens(obs, N)
        cls_out, tok_out = self.body(per, glob)
        m_mu = self.head_m(tok_out).squeeze(-1)           # [B,N]
        std_m = torch.exp(self.log_std_m)
        dist_m = torch.distributions.Normal(m_mu, std_m)

        if self.use_cash_softmax:
            # m_pre は raw logits を想定
            return dist_m.log_prob(m_pre).sum(dim=1)
        else:
            # m_pre は (0,1) を想定
            m_pre = m_pre.clamp(1e-6, 1-1e-6)
            raw_m = torch.log(m_pre) - torch.log1p(-m_pre)     # σ^{-1}
            logJ_m = -(torch.log(m_pre) + torch.log1p(-m_pre))
            return (dist_m.log_prob(raw_m) + logJ_m).sum(dim=1)

    def logprob_s_only(self, obs: torch.Tensor, s_pre: torch.Tensor) -> torch.Tensor:
        B = obs.size(0); N = self.N
        per, glob = self._build_tokens(obs, N)
        cls_out, tok_out = self.body(per, glob)
        s_mu = self.head_s(tok_out).squeeze(-1)          # [B,N]
        std_s = torch.exp(self.log_std_s)

        s_pre = s_pre.clamp(1e-6, 1-1e-6)
        raw_s = torch.log(s_pre) - torch.log1p(-s_pre)
        dist_s = torch.distributions.Normal(s_mu, std_s)
        logJ_s = -(torch.log(s_pre) + torch.log1p(-s_pre))
        return (dist_s.log_prob(raw_s) + logJ_s).sum(dim=1)
    
    def logprob(self, obs: torch.Tensor, m_pre: torch.Tensor, s_pre: torch.Tensor):
        B = obs.size(0); N = self.N
        per, glob = self._build_tokens(obs, N)
        cls_out, tok_out = self.body(per, glob)
        m_mu, s_mu = self.forward_heads(cls_out, tok_out)
        std_m = torch.exp(self.log_std_m); std_s = torch.exp(self.log_std_s)

        dist_m = torch.distributions.Normal(m_mu, std_m)
        dist_s = torch.distributions.Normal(s_mu, std_s)

        if self.use_cash_softmax:
            # m_pre はロジット
            logp_m = dist_m.log_prob(m_pre).sum(dim=1)
        else:
            # m_pre はシグモイド結果
            m_pre = m_pre.clamp(1e-6, 1-1e-6)
            raw_m = torch.log(m_pre) - torch.log1p(-m_pre)
            logJ_m = -(torch.log(m_pre)+torch.log1p(-m_pre))
            logp_m = (dist_m.log_prob(raw_m) + logJ_m).sum(dim=1)

        s_pre = s_pre.clamp(1e-6, 1-1e-6)
        raw_s = torch.log(s_pre) - torch.log1p(-s_pre)
        logJ_s = -(torch.log(s_pre)+torch.log1p(-s_pre))
        logp_s = (dist_s.log_prob(raw_s) + logJ_s).sum(dim=1)

        return logp_m + logp_s

class ValueNetCLS(nn.Module):
    def __init__(self, N:int, d_model:int=128, nlayers:int=2, nhead:int=4):
        super().__init__()
        self.body = JointTransformerBody(N, per_dim=5, d_model=d_model, nhead=nhead, nlayers=nlayers)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, 1)
        )
    def forward(self, obs: torch.Tensor):
        N = (obs.size(1)-4)//5
        per = obs[:, :N*5].view(obs.size(0), N, 5)
        glob = obs[:, N*5:]
        cls_out, _ = self.body(per, glob)
        return self.head(cls_out).squeeze(-1)