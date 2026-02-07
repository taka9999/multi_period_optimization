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
    years: float = 1
    dt_day: float = 1/252
    T_days: int = int(years/dt_day)
    r: float = 0.02
    sigmas: np.ndarray = field(default_factory=lambda: np.array([0.20, 0.18, 0.12, 0.22, 0.19], dtype=float))
    pair_rhos: dict = field(default_factory=lambda: {
    (0,1): 0.60,
    (1,3): -0.20,
    (2,4): 0.05,
    })

    DISCOUNT_BY_BANK: bool = True
    INIT_W0_UNIFORM: bool = True
    BAND_SMOOTH_COEF: float = 0.0
    TRADE_PEN_COEF: float = 0.0
    ALPHA: float = 1/3
    STAGE1_WIDTH_COEF: float = 0.05

    # LQ / MV-style reward parameters
    ALLOW_CASH_IN_MV: bool = False
    MV_USE_TARGET: bool = False   # whether to use target return constraint in MV center
    RISK_GAMMA: float = 0.0
    TARGET_ETA: float = 0.0        # eta in hinge penalty eta*[target - mu^T w]_+
    TARGET_RET_ANN: float = 0.06   # default annual target (discounted-by-bank world)
    # --- regime obs (gamma) ---
    # base global feats are [lam, target_ret_dt, port_var, ||R w||] = 4 dims
    # if env provides self.gamma (len K), obs global dims become 4+K
    REGIME_GAMMA_ON_OBS: bool = False
    OBS_BETA_ZERO: bool = True

    # === rolling obs features (policy1) ===
    ROLL_COV_SUMMARY_ON_OBS = True
    ROLL_OBS_LOOKBACK = 63
    ROLL_TOP_EIGS = 2

    # per-asset token dim: [beta, w, sigma, Rw, lam, sigma_hat]
    PER_ASSET_DIM = 6 if ROLL_COV_SUMMARY_ON_OBS else 5

    # global base: [lam, target_ret_dt, port_var, rw_norm] = 4
    # + (trace, cond, topk) = 2 + ROLL_TOP_EIGS
    ROLL_GLOBAL_DIM = (2 + ROLL_TOP_EIGS) if ROLL_COV_SUMMARY_ON_OBS else 0

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

def buy_to_target_free(S, C, w_tgt, *, max_iter=10):
    """
    Buy towards w_tgt using available cash (no transaction cost).
    """
    S = S.copy()
    C = float(C)

    for _ in range(max_iter):
        Y = S.sum() + C
        if Y <= 0 or C <= 1e-12:
            break
        w = S / (Y + 1e-30)
        gap = np.maximum(0.0, w_tgt - w)
        need = gap * Y
        tot_need = need.sum()
        if tot_need <= 1e-12:
            break
        spend = min(C, tot_need)
        buy = need / (tot_need + 1e-30) * spend
        S += buy
        C -= float(buy.sum())

    return S, C

def project_axis_box_qp(
    w: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    *,
    allow_cash: bool,
    solver: str = "OSQP",
) -> np.ndarray:
    """
    Euclidean projection:
        minimize 0.5||u - w||^2
        s.t. 0 <= u
             A <= u <= B
             sum(u) <= 1 (allow_cash) or == 1 (no cash)
    """
    w = np.asarray(w, float).reshape(-1)
    A = np.asarray(A, float).reshape(-1)
    B = np.asarray(B, float).reshape(-1)
    n = w.size

    u = cp.Variable(n)
    cons = [
        u >= 0,
        u >= A,
        u <= B,
    ]
    cons += [cp.sum(u) <= 1.0] if allow_cash else [cp.sum(u) == 1.0]

    obj = cp.Minimize(0.5 * cp.sum_squares(u - w))
    prob = cp.Problem(obj, cons)

    try:
        prob.solve(solver=getattr(cp, solver), warm_start=True, verbose=False)
    except Exception:
        prob.solve(solver=cp.SCS, warm_start=True, verbose=False)

    if u.value is None:
        # fallback: simple clip (still respects box; sum constraint may violate a bit)
        out = np.clip(w, A, B)
    else:
        out = np.asarray(u.value, float).reshape(-1)

    # numeric safety
    out = np.clip(out, 0.0, 1.0)
    s = float(out.sum())
    if allow_cash:
        if s > 1.0 + 1e-10:
            out /= max(s, 1e-12)
    else:
        if s > 1e-12:
            out /= s
    return out

# =====================================================
# Rotated-box projection QP cache + auto rebuild
# =====================================================
_ROTBOX_PROB_CACHE = {}

def _dense_nozeros(U: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Force a dense-ish matrix to keep OSQP nnz pattern stable across updates."""
    U = np.asarray(U, float)
    U2 = U.copy()
    U2[np.abs(U2) < eps] = eps
    return U2

class _RotBoxQP:
    def __init__(self, n: int, allow_cash: bool):
        import cvxpy as cp
        self.cp = cp
        self.n = int(n)
        self.allow_cash = bool(allow_cash)

        self.u = cp.Variable(self.n)

        # Parameters
        self.w  = cp.Parameter(self.n)
        self.m  = cp.Parameter(self.n)
        self.U  = cp.Parameter((self.n, self.n))
        self.bz = cp.Parameter(self.n, nonneg=True)

        z = self.U.T @ (self.u - self.m)
        cons = [self.u >= 0, z <= self.bz, z >= -self.bz]
        cons += [cp.sum(self.u) <= 1.0] if self.allow_cash else [cp.sum(self.u) == 1.0]

        obj = cp.Minimize(0.5 * cp.sum_squares(self.u - self.w))
        self.prob = cp.Problem(obj, cons)

    def solve_osqp(self, w, m, U, bz, *, warm_start: bool, eps_abs=1e-5, eps_rel=1e-5, max_iter=20000):
        cp = self.cp
        self.w.value  = np.asarray(w, float).reshape(-1)
        self.m.value  = np.asarray(m, float).reshape(-1)
        self.U.value  = _dense_nozeros(U)                 # ★ nnz 安定化
        self.bz.value = np.asarray(bz, float).reshape(-1)

        self.prob.solve(
            solver=cp.OSQP,
            warm_start=bool(warm_start),
            verbose=False,
            eps_abs=float(eps_abs),
            eps_rel=float(eps_rel),
            max_iter=int(max_iter),
            polish=False,
        )
        return self.prob.status, self.u.value

    def solve_scs(self, w, m, U, bz):
        cp = self.cp
        self.w.value  = np.asarray(w, float).reshape(-1)
        self.m.value  = np.asarray(m, float).reshape(-1)
        self.U.value  = _dense_nozeros(U)
        self.bz.value = np.asarray(bz, float).reshape(-1)

        self.prob.solve(solver=cp.SCS, warm_start=True, verbose=False)
        return self.prob.status, self.u.value


def _get_rotbox_qp(n: int, *, allow_cash: bool) -> _RotBoxQP:
    key = (int(n), bool(allow_cash))
    qp = _ROTBOX_PROB_CACHE.get(key)
    if qp is None:
        qp = _RotBoxQP(n, allow_cash)
        _ROTBOX_PROB_CACHE[key] = qp
    return qp


def _rebuild_rotbox_qp(n: int, *, allow_cash: bool) -> _RotBoxQP:
    key = (int(n), bool(allow_cash))
    qp = _RotBoxQP(n, allow_cash)
    _ROTBOX_PROB_CACHE[key] = qp
    return qp

# ----------------------------
# Rotated-box QP stats (minimal)
# ----------------------------
_ROTBOX_QP_STATS = {
    "step_calls": 0,        # step_rotated_box 呼び出し回数
    "inside": 0,            # inside-band 回数（QP未実行）
    "outside": 0,           # outside-band 回数（QP実行）
    "qp_calls": 0,          # project_rotated_box_qp 呼び出し回数（= outside）
    "cached_ok": 0,         # 1) cached solve で _ok
    "rebuild": 0,           # 2) auto rebuild 実行回数
    "rebuild_ok": 0,        # rebuild 後に _ok
    "fallback_scs": 0,      # 3) hard fallback to SCS 実行回数
    "opt_inacc": 0,         # status に optimal_inaccurate が含まれた回数
    "fail": 0,              # val is None で w を返した回数
}

def _rotbox_stats_line(prefix: str = "[RotBoxQP]") -> str:
    s = _ROTBOX_QP_STATS
    return (
        f"{prefix} step={s['step_calls']} inside={s['inside']} outside={s['outside']} | "
        f"qp_calls={s['qp_calls']} cached_ok={s['cached_ok']} rebuild={s['rebuild']} "
        f"rebuild_ok={s['rebuild_ok']} fallback_scs={s['fallback_scs']} "
        f"opt_inacc={s['opt_inacc']} fail={s['fail']}"
    )


def project_rotated_box_qp_old(
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

def project_rotated_box_qp(
    w: np.ndarray,
    m: np.ndarray,
    U: np.ndarray,
    b_z: np.ndarray,
    *,
    allow_cash: bool = True,
    solver: str = "SCS",
) -> np.ndarray:
    """
    Projection of w onto rotated box:
        |U^T (u - m)| <= b_z, u>=0, sum(u)<=1 (cash) or ==1 (no cash)

    Cached CVXPY Problem (via _RotBoxQP) + auto rebuild.
    """

    w = np.asarray(w, float).reshape(-1)
    m = np.asarray(m, float).reshape(-1)
    U = np.asarray(U, float)
    b_z = np.asarray(b_z, float).reshape(-1)
    n = w.size

    qp = _get_rotbox_qp(n, allow_cash=allow_cash)
    _ROTBOX_QP_STATS["qp_calls"] += 1

    def _ok(status: str, val) -> bool:
        if val is None or status is None:
            return False
        s = str(status).lower()
        # SCS: "optimal" / "optimal_inaccurate" があり得る（まずは許容）
        if str(solver).upper() == "SCS":
            return ("optimal" in s)  # optimal / optimal_inaccurate OK
        # OSQP: strict にするなら == "optimal" のみ
        return (s == "optimal")

    status = None
    val = None

    s = str(solver).upper()

    # ---- 1) cached solve ----
    try:
        if s == "SCS":
            status, val = qp.solve_scs(w, m, U, b_z)
        else:
            status, val = qp.solve_osqp(w, m, U, b_z, warm_start=True)
    except Exception:
        status, val = None, None

    if status is not None and ("optimal_inaccurate" in str(status).lower()):
        _ROTBOX_QP_STATS["opt_inacc"] += 1
    if _ok(status, val):
        _ROTBOX_QP_STATS["cached_ok"] += 1

    # ---- 2) auto rebuild + retry ----
    if not _ok(status, val):
        _ROTBOX_QP_STATS["rebuild"] += 1
        qp = _rebuild_rotbox_qp(n, allow_cash=allow_cash)
        try:
            if s == "SCS":
                status, val = qp.solve_scs(w, m, U, b_z)
            else:
                status, val = qp.solve_osqp(w, m, U, b_z, warm_start=False)
        except Exception:
            status, val = None, None
        
        if status is not None and ("optimal_inaccurate" in str(status).lower()):
            _ROTBOX_QP_STATS["opt_inacc"] += 1
        if _ok(status, val):
            _ROTBOX_QP_STATS["rebuild_ok"] += 1

    # ---- 3) hard fallback to SCS ----
    if not _ok(status, val):
        _ROTBOX_QP_STATS["fallback_scs"] += 1
        try:
            status, val = qp.solve_scs(w, m, U, b_z)
        except Exception:
            val = None
        if status is not None and ("optimal_inaccurate" in str(status).lower()):
            _ROTBOX_QP_STATS["opt_inacc"] += 1

    if val is None:
        _ROTBOX_QP_STATS["fail"] += 1
        return w.copy()

    out = np.asarray(val, float).reshape(-1)
    out = clamp01_vec(out)

    ss = float(out.sum())
    if allow_cash:
        if ss > 1.0 + 1e-10:
            out /= max(ss, 1e-12)
    else:
        if ss > 1e-12:
            out /= ss
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
        self.allow_cash = self.cfg.ALLOW_CASH_IN_MV
        
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
        if target_ret is None and float(getattr(self.cfg, "TARGET_ETA", 0.0)) == 0.0:
            self.target_ret_ann = 0.0
        else:
            self.target_ret_ann = float(self.cfg.TARGET_RET_ANN if target_ret is None else target_ret)
        #self.target_ret_ann = float(self.cfg.TARGET_RET_ANN if target_ret is None else target_ret)
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
        self._ret_hist = []   # list of np.ndarray shape (N,)

        return self._make_obs()

    def _make_obs(self):
        """
        per-asset token features: [beta_i, w_i, sigma_i, (R@w)_i, lam]
        global features base: [lam, target_ret_dt, port_var, ||R w||]
        optionally append regime gamma (soft probs): [gamma_0,...,gamma_{K-1}]
        """
        lam_scalar = float(self.lam.mean()) if isinstance(self.lam, np.ndarray) else float(self.lam)
        Y = self.S.sum() + self.C
        w = self.S / (Y + 1e-30)                       # [N]
        beta = self.beta
        # If you want to randomize beta for data-generation but NOT expose it to the policy,
        # #set cfg.OBS_BETA_ZERO = True.
        if bool(getattr(self.cfg, "OBS_BETA_ZERO", False)):
            beta = np.zeros_like(self.beta)
        else:
            beta = self.beta
        sigma = self.sigmas                             # [N]
        Rw = self.R @ w                                 # [N]

        sigma_hat = sigma
        roll_global = None

        if bool(getattr(self.cfg, "ROLL_COV_SUMMARY_ON_OBS", False)):
            lb   = int(getattr(self.cfg, "ROLL_OBS_LOOKBACK", 63))
            topk = int(getattr(self.cfg, "ROLL_TOP_EIGS", 2))

            Cov_hat = None
            if hasattr(self, "_ret_hist") and len(self._ret_hist) >= 2:
                window = np.asarray(self._ret_hist[-lb:], float)  # [L, N]
                if window.shape[0] >= 2:
                    Cov_hat = np.cov(window, rowvar=False, ddof=1) / max(self.dt, 1e-12)  # annualized

            if Cov_hat is None or (not np.all(np.isfinite(Cov_hat))):
                Cov_hat = self.Cov

            diag = np.clip(np.diag(Cov_hat), 0.0, None)
            sigma_hat = np.sqrt(diag)

            # global summaries
            tr = float(np.trace(Cov_hat))
            try:
                eig = np.linalg.eigvalsh(Cov_hat)
                eig = np.sort(np.real(eig))
                eigmin = float(max(eig[0], 1e-12))
                eigmax = float(max(eig[-1], 1e-12))
                cond = float(eigmax / eigmin)
                top = eig[::-1][:topk]
            except Exception:
                cond = float("nan")
                top = np.full(topk, np.nan)

            roll_global = np.concatenate([[tr, cond], np.asarray(top, float)], axis=0)

        port_var = float(w @ self.Cov @ w)
        rw_norm = float(np.linalg.norm(Rw))
        #per_asset = np.stack([beta, w, sigma, Rw, np.full_like(beta, lam_scalar)], axis=0).T  # [N,5]
        #per_asset_flat = per_asset.reshape(-1)                                             # [N*5]
        #base = np.array([lam_scalar, float(self.target_ret_dt), port_var, rw_norm], float)
        per_asset = np.stack(
            [beta, w, sigma, Rw, np.full_like(beta, lam_scalar), sigma_hat],
            axis=0
            ).T  # [N,6]
        per_asset_flat = per_asset.reshape(-1)  # [N*6]
        base = np.array([lam_scalar, float(self.target_ret_dt), port_var, rw_norm], float)
        if roll_global is not None:
            base = np.concatenate([base, roll_global], axis=0)


        # regime gamma (if present)
        if bool(getattr(self.cfg, "REGIME_GAMMA_ON_OBS", True)) and hasattr(self, "gamma") and self.gamma is not None:
            g = np.asarray(self.gamma, float).reshape(-1)
            # safety: normalize
            s = float(g.sum())
            if np.isfinite(s) and s > 0:
                g = g / s
            global_feats = np.concatenate([base, g], axis=0)
        else:
            global_feats = base
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
        w_pre = (self.S / (Y_prev + 1e-30)).copy()
        #self.S, self.C, sold_total = reflect_multi(self.S, self.C, A, B, self.lam)
        inside = bool(np.all(w_pre >= A - 1e-12) and np.all(w_pre <= B + 1e-12))

        sold_total = 0.0
        if not inside:
            # 1) QP target inside [A,B] + simplex/cash constraint
            w_tgt = project_axis_box_qp(
                w_pre, A, B,
                allow_cash=bool(self.allow_cash),
                solver="OSQP",
            )

            # 2) execute sell-only trades toward that target
            self.S, self.C, sold_total = trade_to_target_sellonly(
                self.S, self.C, w_tgt, self.lam
            )
        
        #if not inside:
        #    # after trade:
        #    w_after = self.S / (self.S.sum() + self.C + 1e-30)
        #    print("[A2QPtrade] tgt_gap=",
        #          float(np.max(np.maximum(w_tgt - B, A - w_tgt))),
        #          "after_gap=",
        #          float(np.max(np.maximum(w_after - B, A - w_after))),
        #          "cash=", float(self.C))


        Y_mid0 = self.S.sum() + self.C
        self.w_post_trade = self.S / (Y_mid0 + 1e-30)
        self.sold_total_last = float(sold_total)   # optional

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

                # realized log return for training reward
        gross = Y_next / max(Y_prev, 1e-30)
        r_log = np.log(max(gross, 1e-30))

        # MV/LQ shaping (same)
        Y_mid = self.S.sum() + self.C
        w_mid = self.S / (Y_mid + 1e-30)

        mu = self.r + (self.sigmas**2) * self.beta
        mu_eff = mu - self.r if self.discount_by_bank else mu

        mu_w_dt  = float(mu_eff @ w_mid) * self.dt
        var_w_dt = float(w_mid @ self.Cov @ w_mid) * self.dt

        eta_target = float(getattr(self.cfg, "TARGET_ETA", 0.0))
        shortfall = max(0.0, float(self.target_ret_dt) - mu_w_dt)

        # additional risk penalty
        gamma_risk = float(getattr(self.cfg, "RISK_GAMMA", 0.0))

        u = r_log - 0.5 * gamma_risk * var_w_dt - eta_target * shortfall
        r_step = u - trade_pen

        self.A_prev, self.B_prev = A.copy(), B.copy()
        obs = self._make_obs()
        rlog = np.log(growth + 1e-30)
        self._ret_hist.append(rlog.astype(float))
        return obs, float(r_step), done, float(r_simple)
    
    def step_rotated_box(
        self,
        m: np.ndarray,
        U: np.ndarray,
        b_z: np.ndarray,
        *,
        allow_cash: bool = True,
        solver: str = "SCS",
        use_trade_penalty: bool = False,
        ):
        """
        Level B evaluation step with inside-band no-trade shortcut
        """
        EPS_IN = float(getattr(self.cfg, "ROTBOX_EPS_IN", 1e-6))

        m = np.asarray(m, float).reshape(-1)
        b_z = np.asarray(b_z, float).reshape(-1)
        BZ_FLOOR = float(getattr(self.cfg, "ROTBOX_BZ_FLOOR", 1e-6))
        b_z = np.maximum(b_z, BZ_FLOOR)

        U = np.asarray(U, float)
        N = self.cfg.N_ASSETS
        assert m.size == N and b_z.size == N and U.shape == (N, N)

        Y_prev = self.S.sum() + self.C
        w = self.S / (Y_prev + 1e-30)

        # ======================================================
        # (A) inside-band check in rotated coordinates
        # ======================================================
        z = U.T @ (w - m)
        inside = np.all(np.abs(z) <= b_z + EPS_IN)
        _ROTBOX_QP_STATS["step_calls"] += 1
        if inside:
            _ROTBOX_QP_STATS["inside"] += 1
        else:
            _ROTBOX_QP_STATS["outside"] += 1
        
        margin = float(np.max(np.abs(z) - (b_z + EPS_IN)))   # <=0 なら inside

        #if (self.t % 10000) == 0:
        #    print(f"[RotBoxQP] margin_max={margin:+.3e} | "
        #        f"absz_max={float(np.max(np.abs(z))):.3e} | "
        #        f"bz_min/med/max=({float(np.min(b_z)):.3e}/"
        #        f"{float(np.median(b_z)):.3e}/"
        #        f"{float(np.max(b_z)):.3e})")

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
            z_proj = U.T @ (w_proj - m)
            inside_proj = np.all(np.abs(z_proj) <= b_z + EPS_IN)
            #if (self.t % 10000) == 0:
            #    print(f"[RotBoxQP] inside_proj={inside_proj} | "
            #        f"margin_proj={float(np.max(np.abs(z_proj)-(b_z + EPS_IN))):+.3e}")

            # execute sell-only trades toward that target
            self.S, self.C, sold_total = trade_to_target_sellonly(
                self.S, self.C, w_proj, self.lam
            )
            # (3) BUY free up to target (cash constrained, same as A2)
            self.S, self.C = buy_to_target_free(
                self.S, self.C, w_proj
            )

            # optional debug: after trade
            # Y_mid = float(self.S.sum() + self.C)
            # w_after = self.S / (Y_mid + 1e-30)
            # z_after = U.T @ (w_after - m)
            # inside_after = np.all(np.abs(z_after) <= b_z + EPS_IN)

        # --- after check (tradeした場合も、しない場合も) ---
        Y_mid2 = float(self.S.sum() + self.C)
        w_after = self.S / (Y_mid2 + 1e-30)
        z_after = U.T @ (w_after - m)
        inside_after = np.all(np.abs(z_after) <= b_z + EPS_IN)
        margin_after = float(np.max(np.abs(z_after) - (b_z + EPS_IN)))

        #if (self.t % 10000) == 0:
        #    print(f"[RotBoxQP] inside_before={inside} inside_after={inside_after} "
        #        f"| margin_after={margin_after:+.3e}")

        #if (self.t % 10000) == 0:
        #    print(_rotbox_stats_line())

        #if (self.t % 10000) == 0:
        #    print(f"[RotBoxQP] eps_in={EPS_IN:.1e} bz_floor={BZ_FLOOR:.1e} "
        #        f"| bz_min/med/max=({b_z.min():.3e}/{np.median(b_z):.3e}/{b_z.max():.3e})")
    

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

                # realized log return for training reward
        gross = Y_next / max(Y_prev, 1e-30)
        r_log = np.log(max(gross, 1e-30))

        # MV/LQ shaping (same)
        Y_mid = self.S.sum() + self.C
        w_mid = self.S / (Y_mid + 1e-30)

        mu = self.r + (self.sigmas**2) * self.beta
        mu_eff = mu - self.r if self.discount_by_bank else mu

        mu_w_dt  = float(mu_eff @ w_mid) * self.dt
        var_w_dt = float(w_mid @ self.Cov @ w_mid) * self.dt

        eta_target = float(getattr(self.cfg, "TARGET_ETA", 0.0))
        shortfall = max(0.0, float(self.target_ret_dt) - mu_w_dt)

        # additional risk penalty
        gamma_risk = float(getattr(self.cfg, "RISK_GAMMA", 0.0))

        u = r_log - 0.5 * gamma_risk * var_w_dt - eta_target * shortfall
        r_step = u - trade_pen

        obs = self._make_obs()
        rlog = np.log(growth + 1e-30)
        self._ret_hist.append(rlog.astype(float))
        return obs, float(r_step), done, float(r_simple)