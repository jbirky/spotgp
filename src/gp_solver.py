"""
Gaussian Process solver using the starspot analytic or numerical kernel.

Provides GP regression (conditional mean and variance) and log-likelihood
evaluation for fitting stellar rotation lightcurves.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

try:
    from .analytic_kernel import AnalyticKernel
    from .numerical_kernel import NumericalKernel
except ImportError:
    from analytic_kernel import AnalyticKernel
    from numerical_kernel import NumericalKernel

__all__ = ["GPSolver"]

HPARAM_KEYS = ["peq", "kappa", "inc", "nspot", "lspot", "tau", "alpha_max"]


class GPSolver:
    """
    Gaussian Process solver for stellar lightcurves.

    Builds a covariance matrix from either the AnalyticKernel or the
    NumericalKernel, adds white noise, and provides log-likelihood
    evaluation and predictive (conditional) distributions.

    Parameters
    ----------
    x : array_like, shape (N,)
        Observation times [days].
    y : array_like, shape (N,)
        Observed flux values.
    yerr : array_like, shape (N,) or float
        Measurement uncertainties (1-sigma). Scalar is broadcast.
    hparam : dict or list
        Kernel hyperparameters:
        {peq, kappa, inc, nspot, lspot, tau, alpha_max}.
    kernel_type : {"analytic", "numerical"}
        Which kernel implementation to use (default: "analytic").
    mean : float or callable or None
        Mean function. If float, constant mean; if callable, evaluated
        at x; if None, the sample mean of y is used (default: None).
    kernel_kwargs : dict
        Extra keyword arguments forwarded to the kernel constructor
        (e.g. n_harmonics, n_lat, nsim, tsim, tsamp).
    """

    def __init__(self, x, y, yerr, hparam, kernel_type="analytic",
                 mean=None, **kernel_kwargs):

        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.yerr = np.atleast_1d(np.asarray(yerr, dtype=float))
        if self.yerr.size == 1:
            self.yerr = np.full_like(self.x, self.yerr.item())
        self.N = len(self.x)

        # Parse hyperparameters
        if isinstance(hparam, dict):
            self.hparam = {k: hparam[k] for k in HPARAM_KEYS}
        else:
            self.hparam = dict(zip(HPARAM_KEYS, hparam))

        # Mean function
        if mean is None:
            self._mean_val = np.mean(self.y)
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

        # Build covariance matrix and factorise
        self._build_covariance()

    def _build_kernel(self):
        """Instantiate the kernel object."""
        if self.kernel_type == "analytic":
            self.kernel = AnalyticKernel(self.hparam, **self.kernel_kwargs)
        elif self.kernel_type == "numerical":
            self.kernel = NumericalKernel(self.hparam, **self.kernel_kwargs)
        else:
            raise ValueError(
                f"kernel_type must be 'analytic' or 'numerical', got '{self.kernel_type}'")

    def _eval_kernel(self, tau):
        """Evaluate the kernel at time lags tau."""
        tau = np.asarray(tau, dtype=float)
        if self.kernel_type == "analytic":
            return self.kernel.kernel(np.abs(tau))
        else:
            # NumericalKernel stores an interpolator over positive lags
            return self.kernel.kernel_function(np.abs(tau))

    def _build_covariance(self):
        """Build the N×N covariance matrix and Cholesky-factorise it."""
        # Pairwise lag matrix
        lag_matrix = np.abs(self.x[:, None] - self.x[None, :])

        # Kernel covariance
        self.K = self._eval_kernel(lag_matrix)

        # Add white noise
        self.K_noise = self.K + np.diag(self.yerr ** 2)

        # Cholesky factorisation
        self._cho = cho_factor(self.K_noise)

        # Residuals
        self._mu = self.mean_func(self.x)
        if np.isscalar(self._mu):
            self._mu = np.full(self.N, self._mu)
        self._resid = self.y - self._mu

        # Alpha = K_noise^{-1} @ resid
        self._alpha = cho_solve(self._cho, self._resid)

    def log_likelihood(self):
        """
        Marginal log-likelihood of the data under the GP.

        Returns
        -------
        logL : float
            log p(y | X, theta) = -0.5 * (r^T K^{-1} r + log|K| + N log(2pi))
        """
        # r^T K^{-1} r
        data_fit = self._resid @ self._alpha

        # log|K| from Cholesky
        L = self._cho[0]
        log_det = 2.0 * np.sum(np.log(np.diag(L)))

        return -0.5 * (data_fit + log_det + self.N * np.log(2 * np.pi))

    def predict(self, xpred, return_cov=False):
        """
        Predictive (conditional) distribution at new input locations.

        Parameters
        ----------
        xpred : array_like, shape (M,)
            Prediction times.
        return_cov : bool
            If True, return the full M×M predictive covariance matrix.
            Otherwise return only the predictive variance (diagonal).

        Returns
        -------
        mu_pred : ndarray, shape (M,)
            Predictive mean.
        var_pred : ndarray, shape (M,) or (M, M)
            Predictive variance (or covariance if return_cov=True).
        """
        xpred = np.asarray(xpred, dtype=float)
        M = len(xpred)

        # K(xpred, x) — cross-covariance
        lag_cross = np.abs(xpred[:, None] - self.x[None, :])
        Ks = self._eval_kernel(lag_cross)

        # Predictive mean: mu_pred = mu(xpred) + Ks @ alpha
        mu_prior = self.mean_func(xpred)
        if np.isscalar(mu_prior):
            mu_prior = np.full(M, mu_prior)
        mu_pred = mu_prior + Ks @ self._alpha

        # Predictive covariance: Kss - Ks @ K^{-1} @ Ks^T
        lag_pred = np.abs(xpred[:, None] - xpred[None, :])
        Kss = self._eval_kernel(lag_pred)

        V = cho_solve(self._cho, Ks.T)  # K^{-1} @ Ks^T
        cov_pred = Kss - Ks @ V

        if return_cov:
            return mu_pred, cov_pred
        else:
            return mu_pred, np.diag(cov_pred)

    def sample_prior(self, xpred, n_samples=1, rng=None):
        """
        Draw samples from the GP prior.

        Parameters
        ----------
        xpred : array_like
            Times at which to sample.
        n_samples : int
            Number of samples to draw.
        rng : numpy.random.Generator or None
            Random number generator.

        Returns
        -------
        samples : ndarray, shape (n_samples, M)
        """
        xpred = np.asarray(xpred, dtype=float)
        if rng is None:
            rng = np.random.default_rng()

        lag = np.abs(xpred[:, None] - xpred[None, :])
        K_prior = self._eval_kernel(lag)
        K_prior += 1e-10 * np.eye(len(xpred))  # jitter for numerical stability

        mu = self.mean_func(xpred)
        if np.isscalar(mu):
            mu = np.full(len(xpred), mu)

        return rng.multivariate_normal(mu, K_prior, size=n_samples)

    def sample_posterior(self, xpred, n_samples=1, rng=None):
        """
        Draw samples from the GP posterior (conditional on data).

        Parameters
        ----------
        xpred : array_like
            Times at which to sample.
        n_samples : int
            Number of samples to draw.
        rng : numpy.random.Generator or None
            Random number generator.

        Returns
        -------
        samples : ndarray, shape (n_samples, M)
        """
        if rng is None:
            rng = np.random.default_rng()

        mu_pred, cov_pred = self.predict(xpred, return_cov=True)
        cov_pred += 1e-10 * np.eye(len(xpred))  # jitter

        return rng.multivariate_normal(mu_pred, cov_pred, size=n_samples)

    def update_hparam(self, hparam):
        """
        Update hyperparameters and rebuild the kernel and covariance.

        Parameters
        ----------
        hparam : dict or list
            New hyperparameters.
        """
        if isinstance(hparam, dict):
            self.hparam = {k: hparam[k] for k in HPARAM_KEYS}
        else:
            self.hparam = dict(zip(HPARAM_KEYS, hparam))
        self._build_kernel()
        self._build_covariance()
