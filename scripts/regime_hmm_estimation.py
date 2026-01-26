from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def _to_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df = df.copy()
            df.index = pd.to_datetime(df.index)
        except Exception as e:
            raise ValueError("DataFrame index must be DatetimeIndex or convertible to datetime.") from e
    return df.sort_index()


def _slice_train(df: pd.DataFrame, train_start: str | None, train_end: str | None) -> pd.DataFrame:
    df = _to_datetime_index(df)
    if train_start is not None:
        df = df.loc[pd.Timestamp(train_start):]
    if train_end is not None:
        df = df.loc[:pd.Timestamp(train_end)]
    return df


def _corr_from_cov(Sigma: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(Sigma), eps, None))
    R = Sigma / np.outer(d, d)
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def _ensure_psd(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    S = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(S)
    w = np.maximum(w, eps)
    return (V * w) @ V.T


def _logsumexp(a: np.ndarray, axis: int = -1) -> np.ndarray:
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _mvnorm_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """log N(x | mean, cov) with cov assumed SPD-ish."""
    d = x.shape[0]
    cov = _ensure_psd(cov)
    # Cholesky might still fail if nearly singular; add jitter
    jitter = 1e-10
    for _ in range(6):
        try:
            L = np.linalg.cholesky(cov + jitter * np.eye(d))
            break
        except np.linalg.LinAlgError:
            jitter *= 10
    else:
        # last resort: eigh-based logdet/inv
        w, V = np.linalg.eigh(cov + jitter * np.eye(d))
        w = np.maximum(w, 1e-12)
        logdet = np.sum(np.log(w))
        quad = (V.T @ (x - mean))**2
        quad = np.sum(quad / w)
        return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)

    y = np.linalg.solve(L, x - mean)
    quad = float(y @ y)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


# -----------------------------
# Custom 2-state Gaussian HMM (Baum-Welch) fallback
# -----------------------------
@dataclass
class HMMFitResult:
    P: np.ndarray
    pi: np.ndarray
    mus: np.ndarray
    Sigmas: np.ndarray
    gamma: np.ndarray      # (T,K) state posterior
    loglik: float
    n_iter: int


def _forward_backward_log(logB: np.ndarray, logP: np.ndarray, logpi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    logB: (T,K)  log p(x_t | s_t=k)
    logP: (K,K)  log transition
    logpi:(K,)   log initial prob
    returns: logalpha (T,K), logbeta (T,K), loglik
    """
    T, K = logB.shape
    logalpha = np.empty((T, K))
    logbeta  = np.empty((T, K))

    # forward
    logalpha[0] = logpi + logB[0]
    for t in range(1, T):
        # logalpha[t,k] = logB[t,k] + logsum_j (logalpha[t-1,j] + logP[j,k])
        tmp = logalpha[t - 1][:, None] + logP  # (K,K)
        logalpha[t] = logB[t] + _logsumexp(tmp, axis=0)

    loglik = _logsumexp(logalpha[-1], axis=0)  # scalar

    # backward
    logbeta[-1] = 0.0
    for t in range(T - 2, -1, -1):
        # logbeta[t,j] = logsum_k (logP[j,k] + logB[t+1,k] + logbeta[t+1,k])
        tmp = logP + (logB[t + 1] + logbeta[t + 1])[None, :]  # (K,K)
        logbeta[t] = _logsumexp(tmp, axis=1)

    return logalpha, logbeta, float(loglik)


def _baum_welch_gaussian(
    X: np.ndarray,
    *,
    K: int = 2,
    n_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 123,
    cov_reg: float = 1e-6,
) -> HMMFitResult:
    """
    X: (T,N)
    Gaussian emissions with full covariance per state.
    """
    rng = np.random.default_rng(seed)
    T, N = X.shape

    # init: random responsibilities
    gamma = rng.random((T, K))
    gamma = gamma / gamma.sum(axis=1, keepdims=True)

    # init params
    pi = gamma[0].copy()
    P = np.full((K, K), 1.0 / K)

    def m_step(gamma: np.ndarray, xi_sum: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        Nk = gamma.sum(axis=0) + 1e-12
        mus = (gamma.T @ X) / Nk[:, None]  # (K,N)
        Sigmas = np.zeros((K, N, N))
        for k in range(K):
            Xc = X - mus[k]
            # weighted covariance
            W = gamma[:, k][:, None]
            S = (Xc * W).T @ Xc / Nk[k]
            S = _ensure_psd(S + cov_reg * np.eye(N))
            Sigmas[k] = S
        pi_new = gamma[0].copy()
        P_new = xi_sum / np.maximum(xi_sum.sum(axis=1, keepdims=True), 1e-12)
        return pi_new, P_new, mus, Sigmas

    prev_ll = -np.inf
    xi_sum = np.zeros((K, K))

    for it in range(1, n_iter + 1):
        # E-step: compute logB
        logB = np.empty((T, K))
        # Pre-ensure PSD covs
        # if first iter, compute from gamma init:
        if it == 1:
            # rough Sigma init from overall cov
            S0 = np.cov(X.T) + cov_reg * np.eye(N)
            mus = np.vstack([X.mean(axis=0) + 0.01 * rng.normal(size=N) for _ in range(K)])
            Sigmas = np.stack([S0.copy() for _ in range(K)], axis=0)
        for k in range(K):
            mk = mus[k]
            Sk = Sigmas[k]
            for t in range(T):
                logB[t, k] = _mvnorm_logpdf(X[t], mk, Sk)

        logP = np.log(np.maximum(P, 1e-12))
        logpi = np.log(np.maximum(pi, 1e-12))

        loga, logb, ll = _forward_backward_log(logB, logP, logpi)

        # gamma
        loggamma = loga + logb
        loggamma = loggamma - _logsumexp(loggamma, axis=1)[:, None]
        gamma = np.exp(loggamma)

        # xi (expected transitions)
        xi_sum[:] = 0.0
        for t in range(T - 1):
            # log xi_{j,k} ∝ logalpha[t,j] + logP[j,k] + logB[t+1,k] + logbeta[t+1,k]
            logxi = loga[t][:, None] + logP + logB[t + 1][None, :] + logb[t + 1][None, :]
            logxi = logxi - _logsumexp(logxi.reshape(-1), axis=0)  # normalize over (j,k)
            xi = np.exp(logxi)
            xi_sum += xi

        # M-step
        pi, P, mus, Sigmas = m_step(gamma, xi_sum)

        # convergence
        if np.isfinite(prev_ll) and abs(ll - prev_ll) < tol * (1.0 + abs(prev_ll)):
            prev_ll = ll
            return HMMFitResult(P=P, pi=pi, mus=mus, Sigmas=Sigmas, gamma=gamma, loglik=ll, n_iter=it)
        prev_ll = ll

    return HMMFitResult(P=P, pi=pi, mus=mus, Sigmas=Sigmas, gamma=gamma, loglik=prev_ll, n_iter=n_iter)


# -----------------------------
# Public API
# -----------------------------
def estimate_2regime_gaussian_hmm(
    returns_df: pd.DataFrame,
    *,
    cols: List[str],
    rf_col: str | None = None,
    subtract_rf: bool = True,
    train_start: str | None = None,
    train_end: str | None = None,
    use_hmmlearn_if_available: bool = True,
    seed: int = 123,
    n_iter: int = 200,
    tol: float = 1e-6,
    cov_reg: float = 1e-6,
) -> Tuple[List[Dict], np.ndarray, pd.Series, Dict]:
    """
    Estimate 2-regime (K=2) Gaussian HMM on selected columns over the training period.

    Parameters
    ----------
    returns_df : DataFrame (DatetimeIndex)
        Asset returns (e.g., daily arithmetic returns).
    cols : list[str]
        Columns to use in estimation.
    train_start, train_end : str | None
        Training period bounds (inclusive). E.g. "2007-01-01", "2018-12-31"
    use_hmmlearn_if_available : bool
        If True and hmmlearn is installed, uses hmmlearn.GaussianHMM.
        Otherwise falls back to custom Baum-Welch.
    seed, n_iter, tol, cov_reg : estimation controls

    Returns
    -------
    regimes : list[dict]
        Each dict contains:
          - beta   : (N,)  mu / sig^2  (compatible with mu_eff = sig2 * beta)
          - sigmas : (N,)  sqrt(diag(Sigma))
          - R      : (N,N) corr matrix
          - mu     : (N,)
          - Sigma  : (N,N)
          - weight : float (stationary probability approx or average gamma)
    P : (2,2) ndarray
        Transition matrix.
    state_series : pd.Series
        Most likely state (argmax posterior) for training window.
    info : dict
        Extra diagnostics (loglik, n_iter, gamma, pi, etc.)
    """
    df = _slice_train(returns_df, train_start, train_end)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in returns_df: {missing}")
    # --- build observation matrix ---
    # We assume asset returns are daily log-returns.
    # For portfolio optimization / GBM env alignment, it is usually better to estimate on EXCESS log-returns:
    #   x_t = logret(asset) - logret(TBill)
    # so that "r" is not baked into regime means.
    work = df[cols].copy()
    if subtract_rf and (rf_col is not None):
        if rf_col not in df.columns:
            raise KeyError(f"rf_col={rf_col} not found in returns_df columns")
        # align & subtract risk-free (broadcast)
        rf = df[rf_col].reindex(work.index)
        work = work.sub(rf, axis=0)

    work = work.dropna()
    X = work.to_numpy(dtype=float)
    idx = work.index

    if X.shape[0] < 200:
        raise ValueError(f"Too few observations after slicing/dropna: T={X.shape[0]}")

    K = 2
    N = X.shape[1]

    fitted = None
    used = "custom"

    if use_hmmlearn_if_available:
        try:
            from hmmlearn.hmm import GaussianHMM  # type: ignore
            model = GaussianHMM(
                n_components=K,
                covariance_type="full",
                n_iter=n_iter,
                tol=tol,
                random_state=seed,
                verbose=False,
            )
            model.fit(X)
            P = np.asarray(model.transmat_, float)
            pi = np.asarray(model.startprob_, float)
            mus = np.asarray(model.means_, float)          # (K,N)
            Sigmas = np.asarray(model.covars_, float)      # (K,N,N) for "full"
            gamma = model.predict_proba(X)                 # (T,K)
            loglik = float(model.score(X))
            fitted = HMMFitResult(P=P, pi=pi, mus=mus, Sigmas=Sigmas, gamma=gamma, loglik=loglik, n_iter=int(model.monitor_.iter))
            used = "hmmlearn"
        except Exception:
            fitted = None

    if fitted is None:
        fitted = _baum_welch_gaussian(X, K=K, n_iter=n_iter, tol=tol, seed=seed, cov_reg=cov_reg)
        used = "custom"

    # regime dicts
    regimes: List[Dict] = []
    gamma = fitted.gamma
    w_bar = gamma.mean(axis=0)  # average occupancy as a weight proxy
    for k in range(K):
        mu_log = fitted.mus[k].reshape(-1)  # daily mean of (possibly excess) log-returns
        Sigma = _ensure_psd(fitted.Sigmas[k])
        sig2 = np.clip(np.diag(Sigma), 1e-12, None)
        sigmas = np.sqrt(sig2)
        beta = (mu_log + 0.5 * sig2) / sig2
        R = _corr_from_cov(Sigma)
        regimes.append(
            dict(
                beta=beta,
                sigmas=sigmas,
                R=R,
                mu_log=mu_log,
                Sigma=Sigma,
                weight=float(w_bar[k]),
            )
        )

    # most likely state
    state = np.argmax(gamma, axis=1)
    state_series = pd.Series(state, index=idx, name="hmm_state")

    info = dict(
        used=used,
        loglik=float(fitted.loglik),
        n_iter=int(fitted.n_iter),
        pi=fitted.pi,
        gamma=fitted.gamma,
        train_start=str(idx.min().date()),
        train_end=str(idx.max().date()),
        cols=cols,
    )

    return regimes, fitted.P, state_series, info

def _jsonify_regimes_for_env(
    regimes: List[Dict],
    P: np.ndarray,
    *,
    dt: float,
    annualize_for_env: bool = True,
) -> Dict:
    """
    Convert regimes/P into a JSON-serializable dict for configs/regime_k2.json.

    If env uses GBM form:
      exp((mu_eff - 0.5*sigma^2)*dt + sigma*sqrt(dt)*z),
    and dt is in "years per step" (e.g., 1/252), it is convenient to store sigma as ANNUAL.

    HMM here estimates daily moments (per step). If dt=1/252 and annualize_for_env=True:
      sigma_annual = sigma_step / sqrt(dt)
    Correlation is invariant to scaling.
    """
    out_regimes = []
    for r in regimes:
        sig_step = np.asarray(r["sigmas"], float)
        if annualize_for_env:
            sig_out = (sig_step / np.sqrt(dt)).tolist()
        else:
            sig_out = sig_step.tolist()
        out_regimes.append(
            dict(
                beta=np.asarray(r["beta"], float).tolist(),
                sigmas=sig_out,
                R=np.asarray(r["R"], float).tolist(),
            )
        )
    return {"regimes": out_regimes, "P": np.asarray(P, float).tolist(), "dt": float(dt)}



# -----------------------------
# Example usage (your defaults)
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-start", type=str, default="1991-01-01")
    parser.add_argument("--train-end", type=str, default="2018-12-31")
    parser.add_argument("--cols", type=str, default="LargeCap,MidCap,SmallCap,EAFE,EM",
                        help="comma-separated asset columns")
    parser.add_argument("--rf-col", type=str, default="TBill")
    parser.add_argument("--no-subtract-rf", action="store_true")
    parser.add_argument("--dt", type=float, default=1/252, help="years per step used in env (e.g., 1/252)")
    parser.add_argument("--no-annualize-sigma", action="store_true",
                        help="if set, writes per-step sigma to JSON instead of annualized sigma")
    parser.add_argument("--out", type=str, default="configs/regime_k2.json")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--n-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--cov-reg", type=float, default=1e-6)
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA_PATH = PROJECT_ROOT / "asset_data" / "asset_returns.pkl"
    df = pd.read_pickle(DATA_PATH)

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    regimes, P, st, info = estimate_2regime_gaussian_hmm(
        df,
        cols=cols,
        rf_col=args.rf_col,
        subtract_rf=(not args.no_subtract_rf),
        train_start=args.train_start,
        train_end=args.train_end,
        seed=args.seed,
        n_iter=args.n_iter,
        tol=args.tol,
        cov_reg=args.cov_reg,
    )

    print("used:", info["used"])
    print("loglik:", info["loglik"], "n_iter:", info["n_iter"])
    print("P:\n", P)

    # Pretty print with BOTH daily and annualized sigma for sanity
    dt = float(args.dt)
    for k, r in enumerate(regimes):
        sig_step = np.asarray(r["sigmas"], float)
        sig_ann = sig_step / np.sqrt(dt)
        print(f"\n--- regime {k} weight~{r['weight']:.3f} ---")
        print("beta:", np.round(r["beta"], 3))
        print("sigmas_step:", np.round(sig_step, 4), " | sigmas_ann:", np.round(sig_ann, 3))
        print("R[0:3,0:3]:\n", np.round(r["R"][:3, :3], 3))

    # Write JSON for training
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _jsonify_regimes_for_env(
        regimes, P,
        dt=dt,
        annualize_for_env=(not args.no_annualize_sigma),
    )
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[Wrote] {out_path}")