# RegimeGBMBandEnvMulti.py
import numpy as np
from src.regime_gbm.gbm_env import GBMBandEnvMulti

def _project_to_corr_psd(R: np.ndarray, *, eps: float = 1e-10) -> np.ndarray:
    """Symmetrize, project to PSD, and renormalize to a correlation matrix (diag=1)."""
    R = np.asarray(R, float)
    A = 0.5 * (R + R.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    A = (V * w) @ V.T
    d = np.sqrt(np.clip(np.diag(A), eps, None))
    A = A / np.outer(d, d)
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 1.0)
    return A

def _sample_regime_dict(reg: dict, rng: np.random.Generator, *,
                        beta_std: float = 0.0, sigma_logstd: float = 0.0,
                        corr_noise: float = 0.0,
                        beta_clip: float = 0.999, sigma_clip: tuple[float,float] = (1e-4, 10.0)) -> dict:
    """Episode-level domain randomization for one regime dict."""
    out = dict(reg)
    if "beta" in reg and beta_std > 0:
        b = np.asarray(reg["beta"], float).reshape(-1)
        b = b + rng.normal(0.0, beta_std, size=b.shape)
        b = np.clip(b, -beta_clip, beta_clip)
        out["beta"] = b
    if "sigmas" in reg and sigma_logstd > 0:
        s = np.asarray(reg["sigmas"], float).reshape(-1)
        mult = np.exp(rng.normal(0.0, sigma_logstd, size=s.shape))
        s = np.clip(s * mult, sigma_clip[0], sigma_clip[1])
        out["sigmas"] = s
    if "R" in reg and corr_noise > 0:
        R = np.asarray(reg["R"], float)
        noise = rng.normal(0.0, corr_noise, size=R.shape)
        noise = 0.5 * (noise + noise.T)
        np.fill_diagonal(noise, 0.0)
        out["R"] = _project_to_corr_psd(R + noise)
    return out


class RegimeGBMBandEnvMulti(GBMBandEnvMulti):
    """
    Regime-switching GBM.
    Regime is (optionally) hidden: we do NOT add regime id to obs here.
    """

    def __init__(self, cfg, regimes, P=None, init_regime=None, R=None):
        """Regime-switching extension of GBMBandEnvMulti.

        Parameters
        ----------
        cfg : globalsetting-like object
            Must have N_ASSETS and seed. `sigmas` will be temporarily overridden for base init.
        regimes : list[dict]
            Each dict should have at least `sigmas` (len N) and either `R` (NxN corr) or `Sigma` (NxN cov).
        P : (K,K) array
            Transition matrix. If None, uses uniform transitions.
        init_regime : int | None
            If provided, start in this regime. Else sample from uniform.
        R : (N,N) array | None
            Placeholder correlation for base env init. If None, derived from the first regime.
        """
        #self.regimes = list(regimes)
        self.cfg = cfg
        self.base_regimes = list(regimes)
        self.regimes = self.base_regimes  # active regimes (may be episode-randomized)
        self.episode_regimes = None
        self.K = len(self.regimes)
        # Markov transition matrix over regimes.
        # If not provided, use uniform transitions (each next regime equally likely).
        self.P = None if P is None else np.asarray(P, float)
        self.init_regime = None if init_regime is None else int(init_regime)

        # ---- choose a valid placeholder (sigmas, R) so GBMBandEnvMulti.__init__ passes its assertions ----
        k0 = 0 if self.init_regime is None else max(0, min(self.K - 1, self.init_regime))
        reg0 = self.regimes[k0]

        sig0 = np.asarray(reg0.get("sigmas", getattr(cfg, "sigmas", None)), float).reshape(-1)
        if sig0.size == 0:
            sig0 = np.ones(int(getattr(cfg, "N_ASSETS", 1)), dtype=float)

        if R is None:
            R0 = reg0.get("R", None)
            if R0 is None:
                Sigma0 = reg0.get("Sigma", None)
                if Sigma0 is not None:
                    Sigma0 = np.asarray(Sigma0, float)
                    d = np.sqrt(np.clip(np.diag(Sigma0), 1e-12, None))
                    R0 = Sigma0 / np.outer(d, d)
                else:
                    R0 = np.eye(sig0.size, dtype=float)
            R = R0

        R = np.asarray(R, float)
        if R.shape != (sig0.size, sig0.size):
            R = np.eye(sig0.size, dtype=float)

        # temporarily override cfg.sigmas (base class reads cfg.sigmas)
        orig_sigmas = getattr(cfg, "sigmas", None)
        cfg.sigmas = sig0

        super().__init__(cfg=cfg, R=R)

        # restore cfg.sigmas for caller hygiene
        if orig_sigmas is not None:
            cfg.sigmas = orig_sigmas

        # initial regime & apply parameters
        self.regime_path = []
        self.regime = self._sample_initial_regime()
        self._apply_regime_params(self.regime)

    def _sample_initial_regime(self):
        """Pick an initial regime.

        IMPORTANT: must not reference self.regime here because this is called
        while initializing self.regime itself.
        """
        if self.init_regime is None:
            return int(self.rng.integers(0, self.K))
        return int(max(0, min(self.K - 1, int(self.init_regime))))

    def _step_regime(self):
        if self.P is None:
            p = None
        else:
            p = np.asarray(self.P[self.regime], float).reshape(-1)
            # be tolerant to small numerical issues
            s = float(p.sum())
            if not np.isfinite(s) or s <= 0:
                p = None
            else:
                p = p / s
        self.regime = int(self.rng.choice(self.K, p=p))
        return self.regime

    def _apply_regime_params(self, k):
        reg = self.regimes[k]
        self.beta = np.asarray(reg["beta"], float).reshape(-1)
        self.sigmas = np.asarray(reg["sigmas"], float).reshape(-1)
        self.R = np.asarray(reg["R"], float)

        # rebuild Cov
        self.Cov = np.diag(self.sigmas) @ self.R @ np.diag(self.sigmas)
        eps = 1e-10
        A = 0.5 * (self.R + self.R.T)
        w, V = np.linalg.eigh(A)
        w = np.maximum(w, eps)
        self.R = (V * w) @ V.T
        self.Cov = np.diag(self.sigmas) @ self.R @ np.diag(self.sigmas)

    def reset(self, beta=None, lam=None, target_ret=None, w0=None):
        # episode-level regime parameter randomization (optional)
        if bool(getattr(self.cfg, "REGIME_EPISODE_RANDOMIZE", False)):
            beta_std     = float(getattr(self.cfg, "REGIME_BETA_STD", 0.0))
            sigma_logstd = float(getattr(self.cfg, "REGIME_SIGMA_LOGSTD", 0.0))
            corr_noise   = float(getattr(self.cfg, "REGIME_CORR_NOISE", 0.0))
            beta_clip    = float(getattr(self.cfg, "REGIME_BETA_CLIP", 0.999))
            sigma_clip   = tuple(getattr(self.cfg, "REGIME_SIGMA_CLIP", (1e-4, 10.0)))
            self.episode_regimes = [
                _sample_regime_dict(r, self.rng, beta_std=beta_std, sigma_logstd=sigma_logstd,
                                   corr_noise=corr_noise, beta_clip=beta_clip, sigma_clip=sigma_clip)
                for r in self.base_regimes
            ]
            self.regimes = self.episode_regimes
        else:
            self.episode_regimes = None
            self.regimes = self.base_regimes

        self.regime = self._sample_initial_regime()
        self.regime_path = [self.regime]
        self._apply_regime_params(self.regime)
        return super().reset(beta=self.beta, lam=lam, target_ret=target_ret, w0=w0)

    def step(self, A, B, use_trade_penalty=True):
        # regime transition (each step)
        self._step_regime()
        self.regime_path.append(self.regime)

        # apply new params
        self._apply_regime_params(self.regime)

        # usual step
        return super().step(A, B, use_trade_penalty=use_trade_penalty)

    def get_regime_path(self):
        return np.asarray(self.regime_path, int)