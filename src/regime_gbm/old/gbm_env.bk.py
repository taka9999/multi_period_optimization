from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import math
import torch
import cvxpy as cp

from src.utils.rlopt_helpers import clamp01_vec




@dataclass
class globalsetting:
    seed: int = 42
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    N_ASSETS: int = 5
    years: float = 5
    dt_day: float = 1/252
    T_days: int = int(years/dt_day)
    r: float = 0.02
    sigmas: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.18, 0.12, 0.22, 0.19], dtype=float))
    pair_rhos: dict = field(default_factory=lambda: {
    (0,1): 0.60,   # US Eq と ex-US Eq を高相関
    (1,3): -0.20,  # ex-US Eq と EM Eq をやや逆相関に
    (2,4): 0.05,   # HY と SmallCap をほぼ無相関に近づける
    })

    DISCOUNT_BY_BANK: bool = True
    INIT_W0_UNIFORM: bool = True
    BAND_SMOOTH_COEF: float = 0.0
    TRADE_PEN_COEF: float = 0.0
    ALPHA: float = 1/5
    STAGE1_WIDTH_COEF: float = 0.05

    # LQ / MV-style reward parameters
    RISK_GAMMA: float = 1.0        # gamma in (1/2)*gamma*w^T Sigma w
    TARGET_ETA: float = 5.0        # eta in hinge penalty eta*[target - mu^T w]_+
    TARGET_RET_ANN: float = 0.05   # default annual target (discounted-by-bank world)


def reflect_multi(S: np.ndarray,
                  C: float,
                  A: np.ndarray,
                  B: np.ndarray,
                  lam: np.ndarray | float) -> Tuple[np.ndarray, float, float]:
    """
    Multi-asset asymmetric reflection:
      1) Sell assets with w_i > B_i down to B_i (pay cost on proceeds)
      2) Buy assets with w_i < A_i up to A_i (free; limited by cash)
    lam: scalar or per-asset vector in (0,1]
    Returns: (S_new, C_new, sold_value_sum)
    """
    N = len(S)
    lam_vec = lam if isinstance(lam, np.ndarray) else np.full(N, float(lam), dtype=float)

    # Current wealth
    Y = S.sum() + C
    if Y <= 0.0:
        return S.copy(), C, 0.0

    # ---- SELL down to B (cost on proceeds)
    sold_total = 0.0
    w = S / Y
    over = w - B
    sell_idx = np.where(over > 0.0)[0]
    for i in sell_idx:
        # solve S_i_new such that w_i_new = B_i after paying cost
        # Similar algebra to single-asset case; we sell d units:
        # After sale: S_i' = S_i - d, C' = C + lam_i*d, Y' = Y - d + lam_i*d = Y - (1-lam_i)*d
        # Impose w_i' = (S_i - d) / Y' = B_i  ⇒ solve for d
        Si, Bi, lami = S[i], B[i], lam_vec[i]
        num = max(0.0, Si - Bi*Y)
        denom = max(1e-12, 1.0 - Bi*(1.0 - lami))
        d_req = num / denom
        d = min(d_req, Si)
        S[i] -= d
        C    += lami * d
        sold_total += d
        # update wealth for subsequent assets (sequential sell)
        Y = S.sum() + C
        if Y <= 0.0: break

    # ---- BUY up to A (free, limited by cash)
    Y = S.sum() + C
    if Y <= 0.0:
        return S, max(0.0, C), sold_total

    w = S / Y
    # ---- BUY up to A (free, limited by cash)
    Y = S.sum() + C
    w = S / (Y + 1e-30)

    # iterate re-allocation until no progress
    for _ in range(10):  # 10回も回せば十分
        gaps = np.maximum(0.0, A - w)
        need = gaps * Y
        total_need = need.sum()
        if total_need <= 1e-12 or C <= 1e-12:
            break
        spend = min(C, total_need)
        buy_amt = need / (total_need + 1e-30) * spend
        S += buy_amt
        C -= float(buy_amt.sum())
        # update
        Y = S.sum() + C
        w = S / (Y + 1e-30)

    gaps = np.maximum(0.0, A - w)
    need = gaps * Y                       # target dollar needed per asset
    total_need = need.sum()
    if total_need <= 1e-12 or C <= 1e-12:
        return S, C, sold_total

    # allocate cash proportionally to needs
    alloc = need / (total_need + 1e-30)
    buy_amt = np.minimum(need, alloc * C) # ensure we don't exceed cash
    S += buy_amt
    C -= float(buy_amt.sum())
    #w_after = S / (S.sum() + C + 1e-30)
    #print("max A-gap", np.max(A - w_after), "max B-viol", np.max(w_after - B), "cash", C)
    return S, C, sold_total

def project_rotated_box_qp(
    w: np.ndarray,
    m: np.ndarray,
    U: np.ndarray,
    b_z: np.ndarray,
    *,
    allow_cash: bool = True,
    solver: str = "OSQP",
) -> np.ndarray:
    """
    Projection of w onto rotated box:
        |U^T (u - m)| <= b_z, u>=0, sum(u)<=1 (cash) or ==1 (no cash)
    """
    w = np.asarray(w, float).reshape(-1)
    m = np.asarray(m, float).reshape(-1)
    U = np.asarray(U, float)
    b_z = np.asarray(b_z, float).reshape(-1)
    n = w.size

    u = cp.Variable(n)
    z = U.T @ (u - m)

    obj = cp.Minimize(0.5 * cp.sum_squares(u - w))
    cons = [u >= 0, z <= b_z, z >= -b_z]
    cons += [cp.sum(u) <= 1.0] if allow_cash else [cp.sum(u) == 1.0]

    prob = cp.Problem(obj, cons)
    try:
        prob.solve(
            solver=cp.OSQP,
            verbose=False,
            warm_start=True,
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=200000,
            polish=True,
        )
        #prob.solve(solver=getattr(cp, solver), verbose=False, warm_start=True)
    except Exception:
        prob.solve(solver=cp.SCS, verbose=False)

    if u.value is None:
        # fallback: do nothing
        return w.copy()

    out = np.array(u.value, dtype=float).reshape(-1)
    out = clamp01_vec(out)
    # optional: enforce sum constraint softly (avoid numerical glitches)
    s = float(out.sum())
    if allow_cash:
        if s > 1.0 + 1e-10:
            out /= max(s, 1e-12)
    else:
        if s > 1e-12:
            out /= s
    return out

def trade_to_target_sellonly(
    S: np.ndarray,
    C: float,
    w_target: np.ndarray,
    lam: np.ndarray | float,
) -> Tuple[np.ndarray, float, float]:
    """
    Execute trades to move from current weights to target weights (approximately),
    with sell-only proportional wedge via lam in (0,1].

    Strategy:
      1) Sell assets where current dollar > target dollar (pay wedge on proceeds)
      2) Use remaining cash to buy assets where current dollar < target dollar (free buy, cash-limited)

    Returns (S_new, C_new, sold_total_gross)
    """
    S = np.asarray(S, float).copy()
    N = S.size
    lam_vec = lam if isinstance(lam, np.ndarray) else np.full(N, float(lam), dtype=float)
    w_target = np.asarray(w_target, float).reshape(-1)

    Y = float(S.sum() + C)
    if Y <= 0:
        return S, float(C), 0.0

    # --- SELL step ---
    sold_total = 0.0
    x = S.copy()                    # dollars in risky
    x_t = np.maximum(0.0, w_target) * Y  # target dollars (based on pre-trade wealth)

    sell_amt = np.maximum(0.0, x - x_t)  # gross sell dollars
    sell_idx = np.where(sell_amt > 1e-12)[0]
    for i in sell_idx:
        d = min(float(sell_amt[i]), float(S[i]))
        S[i] -= d
        C    += float(lam_vec[i]) * d
        sold_total += d

    # update wealth after sells
    Y = float(S.sum() + C)
    if Y <= 0:
        return S, float(max(0.0, C)), sold_total

    # --- BUY step (free, cash-limited) ---
    x = S.copy()
    # recompute target dollars w.r.t updated wealth
    x_t = np.maximum(0.0, w_target) * Y
    buy_need = np.maximum(0.0, x_t - x)
    total_need = float(buy_need.sum())
    if total_need <= 1e-12 or C <= 1e-12:
        return S, float(C), sold_total

    spend = min(float(C), total_need)
    buy_amt = buy_need / (total_need + 1e-30) * spend
    S += buy_amt
    C -= float(buy_amt.sum())

    return S, float(C), sold_total


# ----------------------------
# Environment (Multi-asset)
# ----------------------------
class GBMBandEnvMulti:
    def __init__(self,
                 cfg: Optional[globalsetting] = None,
                 R: np.ndarray = None,):
        
        self.cfg = cfg if cfg is not None else globalsetting()
        self.T = self.cfg.T_days
        self.r = float(self.cfg.r)
        self.dt = float(self.cfg.dt_day)
        self.discount_by_bank = self.cfg.DISCOUNT_BY_BANK
        self.bank_growth = 1.0 if self.discount_by_bank else math.exp(self.r*self.dt)
        self.rng = np.random.default_rng(self.cfg.seed)
        
        self.sigmas = np.asarray(self.cfg.sigmas, float)
        N = len(self.sigmas)
        self.R = np.asarray(R, float)
        
        assert self.sigmas.ndim == 1 and self.R.shape == (len(self.sigmas), len(self.sigmas))
        self.N = len(self.sigmas)

        # PSD 調整 & コレスキー（相関行列ベース）
        self.R = 0.5*(self.R + self.R.T)
        eig, U = np.linalg.eigh(self.R)
        eig = np.clip(eig, 1e-10, None)
        self.R = U @ np.diag(eig) @ U.T
        self.ChR = np.linalg.cholesky(self.R)           # 相関行列のCholesky
        self.Cov = np.outer(self.sigmas, self.sigmas) * self.R

        # episode vars
        self.beta = None; self.lam = None
        self.t = None; self.S = None; self.C = None
        self.A_prev = None; self.B_prev = None

    def _draw_z(self):
        if getattr(self, "Z_path", None) is not None:
            z = self.Z_path[self.z_ptr]
            self.z_ptr += 1
            return z
        eps = self.rng.standard_normal(self.N)
        return self.ChR @ eps

    def reset(self,
              beta: np.ndarray,
              lam: float | np.ndarray,
              target_ret: Optional[float]=None,
              S0: float=100.0,
              C0: float=100.0,
              w0: Optional[np.ndarray]=None,
              Z: Optional[np.ndarray]=None):

        self.beta = np.asarray(beta, dtype=float)
        assert self.beta.shape == (self.N,), f"beta must be shape ({self.N},)"
        self.lam  = (np.asarray(lam, dtype=float)
                     if isinstance(lam, (np.ndarray, list, tuple))
                     else float(lam))
        self.Z_path = None if Z is None else np.asarray(Z, float)
        self.z_ptr = 0
        # annual target return (discounted-by-bank). If None, use cfg default
        self.target_ret_ann = float(self.cfg.TARGET_RET_ANN if target_ret is None else target_ret)
        self.target_ret_dt  = self.target_ret_ann * self.dt
        self.t = 0
        self.A_prev = None; self.B_prev = None

        if w0 is None:
            if self.cfg.INIT_W0_UNIFORM:
                # random weights that sum to <=1, residual to cash
                raw = self.rng.random(self.N)
                w0  = raw / (raw.sum() + 1e-12) * self.rng.uniform(0.0, 1.0)
            else:
                w0 = np.full(self.N, 1.0/self.N)
        w0 = clamp01_vec(w0)
        Y0 = S0 + C0
        self.S = Y0 * w0
        self.C = Y0 - self.S.sum()
        self.C = float(max(1e-8, self.C))
        self.w_prev = self.S / (self.S.sum() + self.C)

        return self._make_obs()

    def _make_obs(self):
        """
        per-asset token features: [beta_i, w_i, sigma_i, (R@w)_i, lam]
        global features: [lam, target_ret_dt, port_var, ||R w||]
        """
        lam_scalar = float(self.lam.mean()) if isinstance(self.lam, np.ndarray) else float(self.lam)
        Y = self.S.sum() + self.C
        w = self.S / (Y + 1e-30)                       # [N]
        beta = self.beta                                # [N]
        sigma = self.sigmas                             # [N]
        Rw = self.R @ w                                 # [N]

        per_asset = np.stack([beta, w, sigma, Rw, np.full_like(beta, lam_scalar)], axis=0).T  # [N,5]
        per_asset_flat = per_asset.reshape(-1)                                             # [N*5]

        port_var = float(w @ self.Cov @ w)
        rw_norm = float(np.linalg.norm(Rw))
        global_feats = np.array([lam_scalar, float(self.target_ret_dt), port_var, rw_norm], float)
        return np.concatenate([per_asset_flat, global_feats], axis=0)
        

    def step(self, A: np.ndarray, B: np.ndarray, *, use_trade_penalty: bool=True):

        A = clamp01_vec(A)
        B = clamp01_vec(B)
        B = np.maximum(A + 1e-6, B)
        # smoothness penalty
        band_pen = 0.0
        if self.A_prev is not None and self.B_prev is not None:
            dA = A - self.A_prev
            dB = B - self.B_prev
            band_pen = self.cfg.BAND_SMOOTH_COEF * float((dA*dA + dB*dB).sum())

        # reflect
        Y_prev = self.S.sum() + self.C
        self.S, self.C, sold_total = reflect_multi(self.S, self.C, A, B, self.lam)

        # GBM step (vector)
        z = self._draw_z()
        mu = self.r + (self.sigmas**2) * self.beta         # drift with beta tilt (per-asset)
        mu_eff = mu - self.r if self.discount_by_bank else mu
        growth = np.exp((mu_eff - 0.5*self.sigmas**2)*self.dt + self.sigmas*np.sqrt(self.dt)*z)
        self.S *=growth
        self.C *= self.bank_growth
        self.t += 1
        done = (self.t >= self.T)
        Y_next = self.S.sum() + self.C

        # additional trade penalty only for enhancing penalty for transactions, base transaction cost is already reflcted in wealth update
        trade_pen = 0.0
        if use_trade_penalty:
            lam_scalar = float(self.lam.mean()) if isinstance(self.lam, np.ndarray) else float(self.lam)
            trade_pen = self.cfg.TRADE_PEN_COEF * (1.0 - lam_scalar) * float(sold_total / max(Y_prev, 1e-30))

                # discounted-by-bank simple return
        # discounted-by-bank simple return over dt (diagnostic + main reward component)
        r_simple = (Y_next / max(Y_prev, 1e-30)) - 1.0

        # ----- MV / LQ-style shaping (model-based risk + target constraint) -----
        # Evaluate using *held* weights for this step (after reflection / trading).
        Y_mid = self.S.sum() + self.C
        w_mid = self.S / (Y_mid + 1e-30)

        # Model-implied drift used in the GBM step
        mu = self.r + (self.sigmas**2) * self.beta
        mu_eff = mu - self.r if self.discount_by_bank else mu

        # Per-step expected return and variance (small-dt approximation)
        mu_w_dt  = float(mu_eff @ w_mid) * self.dt
        var_w_dt = float(w_mid @ self.Cov @ w_mid) * self.dt

        gamma_risk = float(getattr(self.cfg, "RISK_GAMMA", 1.0))
        eta_target = float(getattr(self.cfg, "TARGET_ETA", 0.0))

        # hinge penalty: only punish *below target* in expected-return space
        shortfall = max(0.0, float(self.target_ret_dt) - mu_w_dt)

        # reward: r_simple - (gamma/2)*Var_dt - eta*shortfall
        u_mv = r_simple - 0.5 * gamma_risk * var_w_dt - eta_target * shortfall

        # total step reward
        r_step = u_mv - band_pen - trade_pen

        # keep r_simple as diagnostic
        lret = r_simple  # returned as diagnostic

        self.A_prev, self.B_prev = A.copy(), B.copy()

        obs = self._make_obs()
        return obs, float(r_step), done, float(lret)
    
    def step_rotated_box(
        self,
        m: np.ndarray,
        U: np.ndarray,
        b_z: np.ndarray,
        *,
        allow_cash: bool = True,
        solver: str = "OSQP",
        use_trade_penalty: bool = True,
        ):
        """
        Level B evaluation step with inside-band no-trade shortcut
        """

        m = np.asarray(m, float).reshape(-1)
        b_z = np.asarray(b_z, float).reshape(-1)
        U = np.asarray(U, float)
        N = self.cfg.N_ASSETS
        assert m.size == N and b_z.size == N and U.shape == (N, N)

        Y_prev = self.S.sum() + self.C
        w = self.S / (Y_prev + 1e-30)

        # ======================================================
        # (A) inside-band check in rotated coordinates
        # ======================================================
        z = U.T @ (w - m)
        inside = np.all(np.abs(z) <= b_z + 1e-12)

        sold_total = 0.0

        if not inside:
            # ==================================================
            # (B) OUTSIDE band: QP + sell-only trades
            # ==================================================
            w_proj = project_rotated_box_qp(
                w, m, U, b_z,
                allow_cash=allow_cash,
                solver=solver
            )
            self.S, self.C, sold_total = trade_to_target_sellonly(
                self.S, self.C, w_proj, self.lam
            )
        # else:
        #   INSIDE band → no trade, keep S,C as is

        # ======================================================
        # (C) GBM evolution (same as before)
        # ======================================================
        z_eps = self._draw_z()
        mu = self.r + (self.sigmas**2) * self.beta
        mu_eff = mu - self.r if self.discount_by_bank else mu
        growth = np.exp(
            (mu_eff - 0.5*self.sigmas**2)*self.dt
            + self.sigmas*np.sqrt(self.dt)*z_eps
        )
        self.S *= growth
        self.C *= self.bank_growth
        self.t += 1
        done = (self.t >= self.T)
        Y_next = self.S.sum() + self.C

        # ======================================================
        # (D) trade penalty (only if traded)
        # ======================================================
        trade_pen = 0.0
        if use_trade_penalty and (sold_total > 0.0):
            lam_scalar = float(self.lam.mean()) if isinstance(self.lam, np.ndarray) else float(self.lam)
            trade_pen = (
                self.cfg.TRADE_PEN_COEF
                * (1.0 - lam_scalar)
                * float(sold_total / max(Y_prev, 1e-30))
            )

        # ======================================================
        # (E) reward (unchanged)
        # ======================================================
        r_simple = (Y_next / max(Y_prev, 1e-30)) - 1.0

        Y_mid = self.S.sum() + self.C
        w_mid = self.S / (Y_mid + 1e-30)

        mu_w_dt  = float(mu_eff @ w_mid) * self.dt
        var_w_dt = float(w_mid @ self.Cov @ w_mid) * self.dt

        gamma_risk = float(getattr(self.cfg, "RISK_GAMMA", 1.0))
        eta_target = float(getattr(self.cfg, "TARGET_ETA", 0.0))
        shortfall = max(0.0, float(self.target_ret_dt) - mu_w_dt)

        u_mv = r_simple - 0.5 * gamma_risk * var_w_dt - eta_target * shortfall
        r_step = u_mv - trade_pen

        obs = self._make_obs()
        return obs, float(r_step), done, float(r_simple)

    def step_rotated_box_old(
            self,
            m: np.ndarray,
            U: np.ndarray,
            b_z: np.ndarray,
            *,
            allow_cash: bool = True,
            solver: str = "OSQP",
            use_trade_penalty: bool = True,
            ):
        """
        Level B evaluation step:
        - Project current w onto rotated box |U^T(w-m)|<=b_z with simplex constraints
        - Execute sell-only trades towards the projected target
        - Then run GBM dynamics and compute reward using existing logic
        """
        Y_prev = self.S.sum() + self.C
        w = self.S / (Y_prev + 1e-30)

        w_proj = project_rotated_box_qp(w, m, U, b_z, allow_cash=allow_cash, solver=solver)
        self.S, self.C, sold_total = trade_to_target_sellonly(self.S, self.C, w_proj, self.lam)

        # ---- GBM evolution (same as step) ----
        z = self._draw_z()
        mu = self.r + (self.sigmas**2) * self.beta
        mu_eff = mu - self.r if self.discount_by_bank else mu
        growth = np.exp((mu_eff - 0.5*self.sigmas**2)*self.dt + self.sigmas*np.sqrt(self.dt)*z)
        self.S *= growth
        self.C *= self.bank_growth
        self.t += 1
        done = (self.t >= self.T)
        Y_next = self.S.sum() + self.C

        # trade penalty (same rule)
        trade_pen = 0.0
        if use_trade_penalty:
            lam_scalar = float(self.lam.mean()) if isinstance(self.lam, np.ndarray) else float(self.lam)
            trade_pen = self.cfg.TRADE_PEN_COEF * (1.0 - lam_scalar) * float(sold_total / max(Y_prev, 1e-30))

        # diagnostic return
        r_simple = (Y_next / max(Y_prev, 1e-30)) - 1.0

        # MV/LQ shaping (same)
        Y_mid = self.S.sum() + self.C
        w_mid = self.S / (Y_mid + 1e-30)

        mu = self.r + (self.sigmas**2) * self.beta
        mu_eff = mu - self.r if self.discount_by_bank else mu

        mu_w_dt  = float(mu_eff @ w_mid) * self.dt
        var_w_dt = float(w_mid @ self.Cov @ w_mid) * self.dt

        gamma_risk = float(getattr(self.cfg, "RISK_GAMMA", 1.0))
        eta_target = float(getattr(self.cfg, "TARGET_ETA", 0.0))
        shortfall = max(0.0, float(self.target_ret_dt) - mu_w_dt)

        u_mv = r_simple - 0.5 * gamma_risk * var_w_dt - eta_target * shortfall
        r_step = u_mv - trade_pen

        obs = self._make_obs()
        return obs, float(r_step), done, float(r_simple)
