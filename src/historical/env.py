# HistoricalEnv_LQ_v2.py
from __future__ import annotations
from typing import Optional

import numpy as np

from src.regime_gbm.gbm_env import (
    GBMBandEnvMulti,
    reflect_multi,
    project_rotated_box_qp,
    trade_to_target_sellonly,
)
from src.utils.rlopt_helpers import clamp01_vec


class HistoricalBandEnvMulti(GBMBandEnvMulti):
    """
    Drop-in replacement for GBMBandEnvMulti that replays provided log returns.

    returns_log: np.ndarray shape (T_hist, N) of log returns log(P_{t+1}/P_t).

    Bank-discounting (your chosen mode B):
      If cfg.DISCOUNT_BY_BANK=True and returns_are_excess=False:
          r_log_excess = r_log_raw - r*dt
      so cash stays constant (bank_growth=1) and risky assets evolve in excess terms,
      consistent with your GBM env design.
    """

    def __init__(
        self,
        cfg=None,
        R: np.ndarray | None = None,
        *,
        returns_log: Optional[np.ndarray] = None,
        start_idx: int = 0,
        T_days: Optional[int] = None,
        returns_are_excess: bool = False,
    ):
        super().__init__(cfg=cfg, R=R)

        self.returns_log: Optional[np.ndarray] = None
        self.start_idx: int = int(start_idx)
        self.T_days_override: Optional[int] = None if T_days is None else int(T_days)
        self.returns_are_excess: bool = bool(returns_are_excess)

        if returns_log is not None:
            self.set_returns(returns_log)

    # --------- plumbing ---------
    def set_returns(self, returns_log: np.ndarray):
        arr = np.asarray(returns_log, float)
        if arr.ndim != 2:
            raise ValueError(f"returns_log must be 2D (T,N). got {arr.shape}")
        if arr.shape[1] != self.N:
            raise ValueError(
                f"returns_log has N={arr.shape[1]} cols but env expects N={self.N}."
            )
        self.returns_log = arr

    def set_episode(self, *, start_idx: int, T_days: Optional[int] = None):
        self.start_idx = int(start_idx)
        self.T_days_override = None if T_days is None else int(T_days)

    def _compute_episode_T(self) -> int:
        """Clip episode length to fit within historical series."""
        if self.returns_log is None:
            return int(self.cfg.T_days)

        remaining = int(self.returns_log.shape[0] - self.start_idx)
        if remaining <= 0:
            return 0

        base = int(self.cfg.T_days)
        if self.T_days_override is not None:
            base = min(base, int(self.T_days_override))
        return min(base, remaining)

    def _rlog_t(self) -> np.ndarray:
        """Return *excess* log return vector when DISCOUNT_BY_BANK=True (mode B)."""
        if self.returns_log is None:
            raise RuntimeError("returns_log not set. pass returns_log=... or call set_returns().")

        idx = self.start_idx + self.t
        if idx >= self.returns_log.shape[0]:
            return np.zeros(self.N, float)

        rlog = np.asarray(self.returns_log[idx], float).reshape(self.N)

        # --- mode B: subtract r*dt to get excess log return when discount_by_bank ---
        if self.discount_by_bank:
            if not self.returns_are_excess:
                rlog = rlog - float(self.r * self.dt)
        else:
            # If not discounting by bank, but user provided excess, convert back to raw
            if self.returns_are_excess:
                rlog = rlog + float(self.r * self.dt)

        return rlog

    # --------- API-compatible overrides ---------
    def reset(
        self,
        beta: np.ndarray,
        lam: float | np.ndarray,
        target_ret: Optional[float] = None,
        S0: float = 100.0,
        C0: float = 100.0,
        w0: Optional[np.ndarray] = None,
        Z: Optional[np.ndarray] = None,
    ):
        # set horizon from historical window
        T = self._compute_episode_T()
        if T <= 0:
            raise ValueError("Episode window has no data. Check start_idx / returns_log length.")
        self.T = T

        # Z ignored (kept for signature compatibility)
        return super().reset(
            beta=np.asarray(beta, float),
            lam=lam,
            target_ret=target_ret,
            S0=S0,
            C0=C0,
            w0=w0,
            Z=None,
        )

    def step(self, A: np.ndarray, B: np.ndarray, *, use_trade_penalty: bool = True):
        A = clamp01_vec(A)
        B = clamp01_vec(B)
        B = np.maximum(A + 1e-6, B)

        # smoothness penalty (same as parent)
        band_pen = 0.0
        if self.A_prev is not None and self.B_prev is not None:
            dA = A - self.A_prev
            dB = B - self.B_prev
            band_pen = self.cfg.BAND_SMOOTH_COEF * float((dA * dA + dB * dB).sum())

        # reflect / execute trades (same)
        Y_prev = float(self.S.sum() + self.C)
        self.S, self.C, sold_total = reflect_multi(self.S, self.C, A, B, self.lam)

        # ---- historical evolution (replaces GBM) ----
        rlog = self._rlog_t()
        self.S *= np.exp(rlog)
        self.C *= self.bank_growth

        self.t += 1
        done = bool(self.t >= self.T)
        Y_next = float(self.S.sum() + self.C)

        # trade penalty (same rule)
        trade_pen = 0.0
        if use_trade_penalty:
            lam_scalar = float(self.lam.mean()) if isinstance(self.lam, np.ndarray) else float(self.lam)
            trade_pen = self.cfg.TRADE_PEN_COEF * (1.0 - lam_scalar) * float(sold_total / max(Y_prev, 1e-30))

        r_simple = (Y_next / max(Y_prev, 1e-30)) - 1.0

        # reward shaping uses model-implied mu/Cov (same as parent)
        Y_mid = float(self.S.sum() + self.C)
        w_mid = self.S / (Y_mid + 1e-30)

        mu = self.r + (self.sigmas ** 2) * self.beta
        mu_eff = mu - self.r if self.discount_by_bank else mu

        mu_w_dt = float(mu_eff @ w_mid) * self.dt
        var_w_dt = float(w_mid @ self.Cov @ w_mid) * self.dt

        gamma_risk = float(getattr(self.cfg, "RISK_GAMMA", 1.0))
        eta_target = float(getattr(self.cfg, "TARGET_ETA", 0.0))
        shortfall = max(0.0, float(self.target_ret_dt) - mu_w_dt)

        u_mv = r_simple - 0.5 * gamma_risk * var_w_dt - eta_target * shortfall
        r_step = u_mv - band_pen - trade_pen

        self.A_prev, self.B_prev = A.copy(), B.copy()
        obs = self._make_obs()
        return obs, float(r_step), done, float(r_simple)

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
        m = np.asarray(m, float).reshape(-1)
        b_z = np.asarray(b_z, float).reshape(-1)
        U = np.asarray(U, float)
        N = self.cfg.N_ASSETS
        assert m.size == N and b_z.size == N and U.shape == (N, N)

        Y_prev = float(self.S.sum() + self.C)
        w = self.S / (Y_prev + 1e-30)

        # inside-band check
        z = U.T @ (w - m)
        inside = bool(np.all(np.abs(z) <= b_z + 1e-12))

        sold_total = 0.0
        if not inside:
            w_proj = project_rotated_box_qp(
                w, m, U, b_z,
                allow_cash=allow_cash,
                solver=solver
            )
            self.S, self.C, sold_total = trade_to_target_sellonly(self.S, self.C, w_proj, self.lam)

        # ---- historical evolution (replaces GBM) ----
        rlog = self._rlog_t()
        self.S *= np.exp(rlog)
        self.C *= self.bank_growth

        self.t += 1
        done = bool(self.t >= self.T)
        Y_next = float(self.S.sum() + self.C)

        trade_pen = 0.0
        if use_trade_penalty and (sold_total > 0.0):
            lam_scalar = float(self.lam.mean()) if isinstance(self.lam, np.ndarray) else float(self.lam)
            trade_pen = (
                self.cfg.TRADE_PEN_COEF
                * (1.0 - lam_scalar)
                * float(sold_total / max(Y_prev, 1e-30))
            )

        r_simple = (Y_next / max(Y_prev, 1e-30)) - 1.0

        # reward shaping (same as parent)
        Y_mid = float(self.S.sum() + self.C)
        w_mid = self.S / (Y_mid + 1e-30)

        mu = self.r + (self.sigmas ** 2) * self.beta
        mu_eff = mu - self.r if self.discount_by_bank else mu

        mu_w_dt = float(mu_eff @ w_mid) * self.dt
        var_w_dt = float(w_mid @ self.Cov @ w_mid) * self.dt

        gamma_risk = float(getattr(self.cfg, "RISK_GAMMA", 1.0))
        eta_target = float(getattr(self.cfg, "TARGET_ETA", 0.0))
        shortfall = max(0.0, float(self.target_ret_dt) - mu_w_dt)

        u_mv = r_simple - 0.5 * gamma_risk * var_w_dt - eta_target * shortfall
        r_step = u_mv - trade_pen

        obs = self._make_obs()
        return obs, float(r_step), done, float(r_simple)
