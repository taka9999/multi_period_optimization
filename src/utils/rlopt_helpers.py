from __future__ import annotations
import numpy as np
import torch

def clamp01_eps(x, eps=1e-8):
    return float(max(eps, min(1.0-eps, x)))

def clamp01_vec(x: np.ndarray, eps=1e-8):
    return np.clip(x, eps, 1.0-eps)

def nearest_psd(A, eps=1e-12):
    """simplified psd projection"""
    A = (A + A.T) * 0.5
    w, v = np.linalg.eigh(A)
    w = np.clip(w, eps, None)
    A_psd = (v * w) @ v.T
    A_psd = (A_psd + A_psd.T) * 0.5
    return A_psd

def build_corr_from_pairs(N, base_rho=0.0, pair_rhos=None, make_psd=True):
    """
    Build a correlation matrix R for N assets. Specify individual correlations with pair_rhos={(i,j): rho_ij, ...}.
    base_rho is the default correlation (initial value for equicorrelation).
    """
    R = (1.0 - base_rho) * np.eye(N) + base_rho * np.ones((N, N))
    if pair_rhos:
        for (i, j), rho in pair_rhos.items():
            if i == j: 
                R[i, i] = 1.0
            else:
                R[i, j] = R[j, i] = float(rho)
    if make_psd:
        # To fix the diagonal to 1, first make PSD, then rescale diagonal to 1
        R = nearest_psd(R)
        d = np.sqrt(np.clip(np.diag(R), 1e-12, None))
        R = (R / d).T / d
        R = (R + R.T) * 0.5
        np.fill_diagonal(R, 1.0)
    return R

def build_cov(sigmas: np.ndarray, R: np.ndarray, make_psd: bool=True):
    D = np.diag(sigmas)
    Cov = D @ R @ D
    if make_psd:
        # Clip small negative eigenvalues
        evals, evecs = np.linalg.eigh(Cov)
        evals = np.clip(evals, 1e-10, None)
        Cov = (evecs * evals) @ evecs.T
    return Cov

def chol_from_sigmas_corr(sigmas: np.ndarray, R: np.ndarray):
    D = np.diag(sigmas)
    Cov = D @ R @ D
    return np.linalg.cholesky(Cov)

def chol_from_cov(Cov: np.ndarray):
    return np.linalg.cholesky(Cov + 1e-12*np.eye(Cov.shape[0]))

def topk_eig_coords(R: np.ndarray, w: np.ndarray, k: int = 2):
    # Represent w using the top k eigenvectors of R (ensure symmetry for stability)
    evals, evecs = np.linalg.eigh(0.5*(R+R.T))
    idx = np.argsort(evals)[::-1][:k]
    Uk = evecs[:, idx]            # [N,k]
    coords = Uk.T @ w             # [k]
    return coords, Uk

def project_simplex_leq1(x: torch.Tensor, eps: float=1e-8):
    """
    Project x in [0,1]^N onto the simplex L1<=1 (Duchi+ 2008)
    Gradient flows through x, projection is stop-grad (commonly used "detached projection" in PPO)
    """
    with torch.no_grad():
        B, N = x.shape
        y = x.clone()
        s = y.sum(dim=1, keepdim=True)
        mask = (s > 1.0)
        if mask.any():
            # Project each batch individually
            for b in torch.where(mask.squeeze(-1))[0].tolist():
                v, _ = torch.sort(y[b], descending=True)
                cssv = torch.cumsum(v, dim=0) - 1
                ind = torch.arange(1, N+1, device=x.device, dtype=x.dtype)
                cond = v - cssv/ind > 0
                rho = torch.nonzero(cond, as_tuple=False)[-1].item() + 1
                theta = cssv[rho-1] / rho
                y[b] = torch.clamp(y[b] - theta, min=0.0)
        return y