import numpy as np
from typing import Optional, Tuple, Union

def _solve_minvar_target_return(
    Sigma: np.ndarray,
    mu: np.ndarray,
    r: float,
    target_annual: float,
    *,
    allow_cash: bool = True,
    solver: Optional[str] = None,
    return_fallback: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, bool]]:
    """
    Solve (long-only, cash-allowed) minimum-variance portfolio with target return.
      minimize_w   w^T Sigma w
      s.t.         w >= 0
                   sum(w) <= 1           (cash = 1 - sum(w))
                   mu^T w + r*(1-sum(w)) >= target_annual
    Equivalent: (mu - r*1)^T w >= (target_annual - r)

    Returns:
      if return_fallback=False: w_star (N,)
      if return_fallback=True : (w_star (N,), used_fallback (bool))
    """
    try:
        import cvxpy as cp
    except Exception as e:
        raise ImportError("cvxpy is required for CVX teacher. Install cvxpy.") from e

    N = int(Sigma.shape[0])
    Sigma = np.asarray(Sigma, float)
    mu = np.asarray(mu, float).reshape(-1)
    assert mu.shape == (N,), f"mu shape must be ({N},), got {mu.shape}"

    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma = Sigma + 1e-10 * np.eye(N)  # ridge

    ones = np.ones(N, dtype=float)
    excess = mu - r * ones
    rhs = float(target_annual - r)

    # quick feasibility checks (long-only, sum<=1)
    # if rhs <= 0, all cash feasible
    if rhs <= 0.0:
        w0 = np.zeros(N, dtype=float)
        return (w0, False) if return_fallback else w0

    # max achievable excess return with sum(w)<=1 and w>=0 is max(excess,0)
    max_excess = float(np.max(np.maximum(excess, 0.0)))
    if max_excess <= 1e-14 or rhs > max_excess + 1e-12:
        # infeasible -> fallback
        j = int(np.argmax(excess))
        w_fb = np.zeros(N, dtype=float)
        if excess[j] > 0:
            w_fb[j] = min(1.0, rhs / max(float(excess[j]), 1e-12))
        return (w_fb, True) if return_fallback else w_fb

    w = cp.Variable(N, nonneg=True)

    cons = []
    cons.append(cp.sum(w) <= 1.0 if allow_cash else cp.sum(w) == 1.0)
    cons.append(excess @ w >= rhs)

    prob = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), cons)
    used_fallback = False

    # solve
    solved = False
    status_ok = False
    solve_kwargs = dict(warm_start=True)

    def _status_good(st: str) -> bool:
        # cvxpy statuses: OPTIMAL / OPTIMAL_INACCURATE are acceptable here
        return st in ("optimal", "optimal_inaccurate", "OPTIMAL", "OPTIMAL_INACCURATE")

    if solver is None:
        for sol in ("OSQP", "ECOS", "SCS"):
            try:
                prob.solve(solver=getattr(cp, sol), **solve_kwargs)
                status_ok = _status_good(str(prob.status))
                solved = (w.value is not None) and status_ok
                if solved:
                    break
            except Exception:
                continue
    else:
        try:
            prob.solve(solver=getattr(cp, solver), **solve_kwargs)
            status_ok = _status_good(str(prob.status))
            solved = (w.value is not None) and status_ok
        except Exception:
            solved = False

    if not solved:
        # fallback
        used_fallback = True
        if rhs <= 0:
            w0 = np.zeros(N, dtype=float)
            return (w0, used_fallback) if return_fallback else w0
        j = int(np.argmax(excess))
        w_fb = np.zeros(N, dtype=float)
        if excess[j] > 0:
            w_fb[j] = min(1.0, rhs / max(float(excess[j]), 1e-12))
        return (w_fb, used_fallback) if return_fallback else w_fb

    w_star = np.asarray(w.value, dtype=float).reshape(-1)
    w_star = np.clip(w_star, 0.0, 1.0)

    s = float(w_star.sum())
    if allow_cash:
        if s > 1.0:
            w_star /= max(s, 1e-12)
    else:
        if s > 0.0:
            w_star /= max(s, 1e-12)

    return (w_star, False) if return_fallback else w_star


def mv_teacher_batch(
    Cov_batch: np.ndarray,
    beta_batch: np.ndarray,
    sigmas: np.ndarray,
    r: float,
    target_annual_batch: np.ndarray,
    *,
    allow_cash: bool = True,
    return_info: bool = False,
):
    """
    Batch teacher generator.
    Cov_batch: [B,N,N]
    beta_batch: [B,N]
    mu = r + sigma^2 * beta (consistent with your GBMEnv)
    Returns:
      - return_info=False: W_star [B,N]
      - return_info=True : (W_star [B,N], info dict)
    """
    Cov_batch = np.asarray(Cov_batch, float)
    beta_batch = np.asarray(beta_batch, float)
    sigmas = np.asarray(sigmas, float).reshape(-1)
    target_annual_batch = np.asarray(target_annual_batch, float).reshape(-1)

    B, N, _ = Cov_batch.shape
    assert sigmas.shape == (N,), (sigmas.shape, N)
    assert beta_batch.shape == (B, N), beta_batch.shape
    assert target_annual_batch.shape == (B,), target_annual_batch.shape

    mu = r + (sigmas.reshape(1, N) ** 2) * beta_batch  # [B,N]
    W = np.zeros((B, N), dtype=float)

    info = {
        "fallback": np.zeros(B, dtype=bool),
        "sumw": np.zeros(B),
        "cash": np.zeros(B),
        "maxw": np.zeros(B),
        "entropy_inv": np.full(B, np.nan),
        "effN_inv": np.full(B, np.nan),
    }

    for i in range(B):
        w_star, used_fallback = _solve_minvar_target_return(
            Cov_batch[i], mu[i], r, float(target_annual_batch[i]),
            allow_cash=allow_cash,
            return_fallback=True
        )
        W[i] = w_star
        s = float(w_star.sum())
        info["fallback"][i] = used_fallback
        info["sumw"][i] = s
        info["cash"][i] = 1.0 - s
        info["maxw"][i] = float(w_star.max())

        if s > 1e-8:
            x = w_star / s
            info["entropy_inv"][i] = -float(np.sum(x*np.log(x+1e-12)))
            info["effN_inv"][i] = 1.0 / float(np.sum(x*x))

    return (W, info) if return_info else W
