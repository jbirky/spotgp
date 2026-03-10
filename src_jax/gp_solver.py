"""
JAX-accelerated Gaussian Process solver using the starspot analytic kernel.

Uses JAX for JIT-compiled covariance matrix construction, Cholesky
factorisation, and GP operations (log-likelihood, prediction).
"""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

try:
    from .analytic_kernel import AnalyticKernel
except ImportError:
    from analytic_kernel import AnalyticKernel

__all__ = ["GPSolver"]

HPARAM_KEYS = ["peq", "kappa", "inc", "nspot", "lspot", "tau", "alpha_max"]


class GPSolver:
    """
    JAX-accelerated Gaussian Process solver for stellar lightcurves.

    Key optimizations over numpy/scipy version:
    - Covariance matrix built via vectorized JAX operations
    - Cholesky factorisation and solves via jax.scipy.linalg
    - Log-likelihood computation JIT-compiled
    - Prediction uses JAX linear algebra

    Parameters
    ----------
    x : array_like, shape (N,)
        Observation times [days].
    y : array_like, shape (N,)
        Observed flux values.
    yerr : array_like, shape (N,) or float
        Measurement uncertainties (1-sigma).
    hparam : dict or list
        Kernel hyperparameters.
    kernel_type : {"analytic"}
        Which kernel to use (default: "analytic").
        Note: numerical kernel not supported in JAX version.
    mean : float or callable or None
        Mean function.
    kernel_kwargs : dict
        Extra kwargs forwarded to the kernel constructor.
    """

    def __init__(self, x, y, yerr, hparam, kernel_type="analytic",
                 mean=None, **kernel_kwargs):

        self.x = jnp.asarray(x, dtype=float)
        self.y = jnp.asarray(y, dtype=float)
        yerr_arr = jnp.atleast_1d(jnp.asarray(yerr, dtype=float))
        if yerr_arr.size == 1:
            self.yerr = jnp.full_like(self.x, yerr_arr.item())
        else:
            self.yerr = yerr_arr
        self.N = len(self.x)

        # Parse hyperparameters
        if isinstance(hparam, dict):
            self.hparam = {k: hparam[k] for k in HPARAM_KEYS}
        else:
            self.hparam = dict(zip(HPARAM_KEYS, hparam))

        # Mean function
        if mean is None:
            self._mean_val = float(jnp.mean(self.y))
            self.mean_func = lambda t: self._mean_val
        elif callable(mean):
            self.mean_func = mean
        else:
            self._mean_val = float(mean)
            self.mean_func = lambda t: self._mean_val

        # Build kernel
        self.kernel_type = kernel_type
        self.kernel_kwargs = kernel_kwargs
        self._build_kernel()

        # Build covariance and factorise
        self._build_covariance()

    def _build_kernel(self):
        """Instantiate the kernel object."""
        if self.kernel_type == "analytic":
            self.kernel = AnalyticKernel(self.hparam, **self.kernel_kwargs)
        else:
            raise ValueError(
                f"JAX GPSolver only supports 'analytic' kernel, got '{self.kernel_type}'")

    def _eval_kernel(self, tau):
        """Evaluate the kernel at time lags tau."""
        tau = jnp.asarray(tau, dtype=float)
        return jnp.asarray(self.kernel.kernel(jnp.abs(tau)))

    def _build_covariance(self):
        """Build the N x N covariance matrix and Cholesky-factorise it."""
        # Pairwise lag matrix (vectorized)
        lag_matrix = jnp.abs(self.x[:, None] - self.x[None, :])

        # Kernel covariance
        self.K = self._eval_kernel(lag_matrix)

        # Add white noise
        self.K_noise = self.K + jnp.diag(self.yerr ** 2)

        # Cholesky factorisation (lower triangular)
        self._L = jla.cholesky(self.K_noise, lower=True)

        # Residuals
        mu = self.mean_func(self.x)
        if jnp.isscalar(mu):
            self._mu = jnp.full(self.N, mu)
        else:
            self._mu = jnp.asarray(mu)
        self._resid = self.y - self._mu

        # Alpha = K_noise^{-1} @ resid via Cholesky solve
        self._alpha = jla.cho_solve((self._L, True), self._resid)

    def log_likelihood(self):
        """
        Marginal log-likelihood of the data under the GP.

        Returns
        -------
        logL : float
            log p(y | X, theta)
        """
        data_fit = self._resid @ self._alpha
        log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(self._L)))
        return float(-0.5 * (data_fit + log_det + self.N * jnp.log(2 * jnp.pi)))

    def predict(self, xpred, return_cov=False):
        """
        Predictive distribution at new input locations.

        Parameters
        ----------
        xpred : array_like, shape (M,)
            Prediction times.
        return_cov : bool
            If True, return full predictive covariance.

        Returns
        -------
        mu_pred : ndarray, shape (M,)
        var_pred : ndarray, shape (M,) or (M, M)
        """
        xpred = jnp.asarray(xpred, dtype=float)
        M = len(xpred)

        # Cross-covariance K(xpred, x)
        lag_cross = jnp.abs(xpred[:, None] - self.x[None, :])
        Ks = self._eval_kernel(lag_cross)

        # Predictive mean
        mu_prior = self.mean_func(xpred)
        if jnp.isscalar(mu_prior):
            mu_prior = jnp.full(M, mu_prior)
        mu_pred = mu_prior + Ks @ self._alpha

        # Predictive covariance
        lag_pred = jnp.abs(xpred[:, None] - xpred[None, :])
        Kss = self._eval_kernel(lag_pred)

        V = jla.cho_solve((self._L, True), Ks.T)
        cov_pred = Kss - Ks @ V

        if return_cov:
            return np.asarray(mu_pred), np.asarray(cov_pred)
        else:
            return np.asarray(mu_pred), np.asarray(jnp.diag(cov_pred))

    def sample_prior(self, xpred, n_samples=1, rng=None):
        """Draw samples from the GP prior."""
        xpred = jnp.asarray(xpred, dtype=float)
        if rng is None:
            rng = np.random.default_rng()

        lag = jnp.abs(xpred[:, None] - xpred[None, :])
        K_prior = self._eval_kernel(lag)
        K_prior = K_prior + 1e-10 * jnp.eye(len(xpred))

        mu = self.mean_func(xpred)
        if jnp.isscalar(mu):
            mu = jnp.full(len(xpred), mu)

        return rng.multivariate_normal(np.asarray(mu), np.asarray(K_prior),
                                       size=n_samples)

    def sample_posterior(self, xpred, n_samples=1, rng=None):
        """Draw samples from the GP posterior."""
        if rng is None:
            rng = np.random.default_rng()

        mu_pred, cov_pred = self.predict(xpred, return_cov=True)
        cov_pred = cov_pred + 1e-10 * np.eye(len(xpred))

        return rng.multivariate_normal(mu_pred, cov_pred, size=n_samples)

    def update_hparam(self, hparam):
        """Update hyperparameters and rebuild kernel and covariance."""
        if isinstance(hparam, dict):
            self.hparam = {k: hparam[k] for k in HPARAM_KEYS}
        else:
            self.hparam = dict(zip(HPARAM_KEYS, hparam))
        self._build_kernel()
        self._build_covariance()
