"""
MCMC sampling for GP hyperparameters using BlackJAX NUTS.

Provides three methods for estimating the mass matrix:
1. Hessian at the MAP via JAX autodiff
2. Analytic Fisher information via autodiff kernel derivatives
3. Laplace approximation via jax.hessian

All methods use a pure-functional JAX log-posterior that is fully
differentiable, enabling efficient gradient-based sampling with NUTS.
"""
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
import blackjax
from functools import partial

try:
    from .analytic_kernel import (
        _Gamma_hat, _R_Gamma, _cn_general_jax,
        _cn_squared_coefficients_jax, _gauss_legendre_grid,
        HPARAM_KEYS as _KERNEL_KEYS,
    )
except ImportError:
    from analytic_kernel import (
        _Gamma_hat, _R_Gamma, _cn_general_jax,
        _cn_squared_coefficients_jax, _gauss_legendre_grid,
        HPARAM_KEYS as _KERNEL_KEYS,
    )

__all__ = ["MCMCSampler"]

# Kernel-only hyperparameter keys (7 params)
KERNEL_HPARAM_KEYS = list(_KERNEL_KEYS)

# Full hyperparameter keys including optional white noise
HPARAM_KEYS_WITH_NOISE = KERNEL_HPARAM_KEYS + ["sigma_n"]


# =====================================================================
# Pure-functional GP log-likelihood (differentiable end-to-end)
# =====================================================================

def _kernel_eval(theta_arr, lag_flat, n_harmonics, n_lat, lat_range, n_omega,
                  quad_nodes=None, quad_weights=None):
    """
    Pure-functional kernel evaluation: theta_arr -> kernel values.

    Parameters
    ----------
    theta_arr : jnp.ndarray, shape (7,)
        [peq, kappa, inc, nspot, lspot, tau, alpha_max]
        (kernel parameters only, no white noise)
    lag_flat : jnp.ndarray, shape (M,)
        Flattened time lags.
    n_harmonics, n_lat, lat_range, n_omega : kernel config (static).
    quad_nodes, quad_weights : jnp.ndarray or None
        Gauss-Legendre nodes and weights. If None, uses trapezoid rule.

    Returns
    -------
    K_flat : jnp.ndarray, shape (M,)
        Kernel values at each lag.
    """
    peq, kappa, inc, nspot, lspot, tau, alpha_max = theta_arr

    # R_Gamma (independent of latitude)
    R = _R_Gamma(lag_flat, lspot, tau, alpha_max, n_omega=n_omega)

    # Latitude integration
    def _lat_contribution(phi):
        ns = jnp.arange(n_harmonics + 1)
        cn_vals = jax.vmap(lambda n: _cn_general_jax(n, inc, phi))(ns)
        cn_sq = cn_vals ** 2
        w0 = 2 * jnp.pi * (1 - kappa * jnp.sin(phi) ** 2) / peq
        harm_ns = jnp.arange(1, n_harmonics + 1)
        cosine_terms = jnp.sum(
            cn_sq[1:] * jnp.cos(harm_ns * w0 * lag_flat[:, None]), axis=1
        )
        return cn_sq[0] + 2 * cosine_terms

    if quad_nodes is not None:
        # Gauss-Legendre quadrature
        phi_grid = quad_nodes
        all_contribs = jax.vmap(_lat_contribution)(phi_grid)
        # quad_weights already include the Jacobian (b-a)/2
        norm = jnp.sum(quad_weights)
        K = jnp.sum(quad_weights[:, None] * all_contribs, axis=0)
        K = K / norm
    else:
        # Trapezoid rule
        phi_min, phi_max = lat_range
        phi_grid = jnp.linspace(phi_min, phi_max, n_lat)
        dphi = phi_grid[1] - phi_grid[0]
        all_contribs = jax.vmap(_lat_contribution)(phi_grid)
        K = jnp.sum(all_contribs, axis=0) * dphi
        K = K / (phi_max - phi_min)

    K = R * K * nspot / jnp.pi ** 2

    return K


def _gp_log_likelihood(theta_full, x, y, yerr, mean_val,
                       n_harmonics, n_lat, lat_range, n_omega,
                       fit_sigma_n, quad_nodes=None, quad_weights=None):
    """
    Pure-functional GP marginal log-likelihood.

    Parameters
    ----------
    theta_full : jnp.ndarray, shape (7,) or (8,)
        Kernel params [peq, kappa, inc, nspot, lspot, tau, alpha_max],
        optionally followed by sigma_n (white noise amplitude).
    x, y, yerr : jnp.ndarray
        Observations.
    mean_val : float
        Constant mean.
    n_harmonics, n_lat, lat_range, n_omega : kernel config.
    fit_sigma_n : bool
        If True, theta_full has 8 elements (last is sigma_n).
    quad_nodes, quad_weights : jnp.ndarray or None
        Gauss-Legendre nodes/weights. If None, uses trapezoid rule.

    Returns
    -------
    logL : scalar
    """
    N = x.shape[0]

    # Split kernel params and white noise
    if fit_sigma_n:
        theta_kernel = theta_full[:7]
        sigma_n = theta_full[7]
    else:
        theta_kernel = theta_full
        sigma_n = 0.0

    # Build covariance matrix
    lag_matrix = jnp.abs(x[:, None] - x[None, :])
    lag_flat = lag_matrix.ravel()
    K_flat = _kernel_eval(theta_kernel, lag_flat,
                          n_harmonics, n_lat, lat_range, n_omega,
                          quad_nodes=quad_nodes, quad_weights=quad_weights)
    K = K_flat.reshape(N, N)

    # Add measurement noise + white noise + jitter
    noise_var = yerr ** 2 + sigma_n ** 2
    K_noise = K + jnp.diag(noise_var) + 1e-8 * jnp.eye(N)

    # Cholesky
    L = jla.cholesky(K_noise, lower=True)

    # Residuals
    resid = y - mean_val

    # alpha = K_noise^{-1} @ resid
    alpha = jla.cho_solve((L, True), resid)

    # log-likelihood
    data_fit = resid @ alpha
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    return -0.5 * (data_fit + log_det + N * jnp.log(2 * jnp.pi))


# =====================================================================
# Prior definitions
# =====================================================================

def _default_log_prior(theta_arr, bounds):
    """
    Default log-prior: soft uniform within bounds.
    Uses sigmoid barriers to keep gradients finite near boundaries.
    """
    lo = bounds[:, 0]
    hi = bounds[:, 1]
    k = 100.0  # steepness
    log_lo = jnp.sum(jax.nn.log_sigmoid(k * (theta_arr - lo)))
    log_hi = jnp.sum(jax.nn.log_sigmoid(k * (hi - theta_arr)))
    log_vol = jnp.sum(jnp.log(hi - lo))
    return log_lo + log_hi - log_vol


# =====================================================================
# MCMCSampler class
# =====================================================================

class MCMCSampler:
    """
    MCMC sampler for GP hyperparameters using BlackJAX NUTS.

    Provides three mass-matrix estimation strategies:
    1. ``mass_matrix_hessian_map``: Hessian of neg-log-posterior at MAP
    2. ``mass_matrix_fisher``: Fisher information via autodiff kernel derivatives
    3. ``mass_matrix_laplace``: Laplace approximation via jax.hessian

    Parameters
    ----------
    x : array_like, shape (N,)
        Observation times [days].
    y : array_like, shape (N,)
        Observed flux values.
    yerr : array_like, shape (N,) or float
        Measurement uncertainties.
    theta0 : dict or array_like
        Initial hyperparameter guess. Dict keys or positional order:
        peq, kappa, inc, nspot, lspot, tau, alpha_max[, sigma_n].
    bounds : dict or array_like, optional
        Parameter bounds. If None, uses broad defaults.
    fit_sigma_n : bool
        If True, include white noise amplitude sigma_n as an extra
        free parameter (default False). When True, theta0 and bounds
        should include a sigma_n entry.
    log_prior : callable, optional
        Custom log-prior function f(theta_arr) -> scalar.
        If None, uses soft uniform within bounds.
    mean : float or None
        Constant GP mean. If None, uses sample mean of y.
    n_harmonics : int
        Kernel harmonics (default 2).
    n_lat : int
        Latitude grid points (default 64).
    lat_range : tuple
        Latitude integration range (default (0, pi)).
    n_omega : int
        Frequency grid points for R_Gamma (default 4096).
    """

    DEFAULT_BOUNDS = {
        "peq":       (0.5, 50.0),
        "kappa":     (0.001, 0.999),
        "inc":       (0.01, np.pi - 0.01),
        "nspot":     (1.0, 500.0),
        "lspot":     (0.1, 20.0),
        "tau":       (0.05, 10.0),
        "alpha_max": (0.001, 1.0),
        "sigma_n":   (1e-6, 0.1),
    }

    def __init__(self, x, y, yerr, theta0, bounds=None,
                 fit_sigma_n=False, log_prior=None, mean=None,
                 n_harmonics=2, n_lat=64,
                 lat_range=(0, np.pi), n_omega=4096,
                 quadrature="trapezoid"):

        self.fit_sigma_n = fit_sigma_n
        self.param_keys = (HPARAM_KEYS_WITH_NOISE if fit_sigma_n
                           else KERNEL_HPARAM_KEYS)
        self.n_params = len(self.param_keys)

        self.x = jnp.asarray(x, dtype=jnp.float64)
        self.y = jnp.asarray(y, dtype=jnp.float64)
        yerr_arr = jnp.atleast_1d(jnp.asarray(yerr, dtype=jnp.float64))
        if yerr_arr.size == 1:
            self.yerr = jnp.full_like(self.x, yerr_arr.item())
        else:
            self.yerr = yerr_arr

        # Parse theta0
        if isinstance(theta0, dict):
            self.theta0 = jnp.array(
                [float(theta0[k]) for k in self.param_keys],
                dtype=jnp.float64)
        else:
            self.theta0 = jnp.asarray(theta0, dtype=jnp.float64)

        # Parse bounds
        if bounds is None:
            self.bounds = jnp.array(
                [self.DEFAULT_BOUNDS[k] for k in self.param_keys],
                dtype=jnp.float64)
        elif isinstance(bounds, dict):
            self.bounds = jnp.array(
                [bounds.get(k, self.DEFAULT_BOUNDS[k]) for k in self.param_keys],
                dtype=jnp.float64)
        else:
            self.bounds = jnp.asarray(bounds, dtype=jnp.float64)

        # Mean
        if mean is None:
            self.mean_val = float(jnp.mean(self.y))
        else:
            self.mean_val = float(mean)

        # Kernel config (static)
        self.n_harmonics = n_harmonics
        self.n_lat = n_lat
        self.lat_range = lat_range
        self.n_omega = n_omega
        self.quadrature = quadrature

        # Precompute quadrature grid
        if quadrature == "gauss-legendre":
            self._quad_nodes, self._quad_weights = _gauss_legendre_grid(
                n_lat, lat_range[0], lat_range[1])
        else:
            self._quad_nodes = None
            self._quad_weights = None

        # Prior
        self._custom_log_prior = log_prior

        # Build JIT-compiled log-posterior
        self._build_logposterior()

        # Precompute lag matrix (constant for fixed x)
        self._lag_matrix = jnp.abs(self.x[:, None] - self.x[None, :])
        self._lag_flat = self._lag_matrix.ravel()

        # Storage
        self.map_estimate = None
        self.inverse_mass_matrix = None
        self.samples = None
        self._nuts_info = None

    def _build_logposterior(self):
        """Build JIT-compiled log-posterior and its gradient."""
        bounds = self.bounds
        x, y, yerr = self.x, self.y, self.yerr
        mean_val = self.mean_val
        n_h, n_l, lr, n_o = (self.n_harmonics, self.n_lat,
                              self.lat_range, self.n_omega)
        custom_prior = self._custom_log_prior
        fit_sn = self.fit_sigma_n
        qn, qw = self._quad_nodes, self._quad_weights

        @jax.jit
        def log_posterior(theta_arr):
            ll = _gp_log_likelihood(theta_arr, x, y, yerr, mean_val,
                                    n_h, n_l, lr, n_o, fit_sn,
                                    quad_nodes=qn, quad_weights=qw)
            if custom_prior is not None:
                lp = custom_prior(theta_arr)
            else:
                lp = _default_log_prior(theta_arr, bounds)
            return ll + lp

        @jax.jit
        def neg_log_posterior(theta_arr):
            return -log_posterior(theta_arr)

        self.log_posterior = log_posterior
        self.neg_log_posterior = neg_log_posterior
        self.grad_log_posterior = jax.jit(jax.grad(log_posterior))
        self.grad_neg_log_posterior = jax.jit(jax.grad(neg_log_posterior))

    # =================================================================
    # MAP estimation
    # =================================================================

    def find_map(self, theta0=None, method="L-BFGS-B", maxiter=500):
        """
        Find the maximum a posteriori (MAP) estimate.

        Uses scipy.optimize.minimize with JAX-computed gradients.

        Parameters
        ----------
        theta0 : array_like, optional
            Starting point. If None, uses self.theta0.
        method : str
            Scipy optimizer method (default "L-BFGS-B").
        maxiter : int
            Maximum iterations.

        Returns
        -------
        theta_map : jnp.ndarray, shape (n_params,)
            MAP estimate.
        result : scipy OptimizeResult
            Full optimizer output.
        """
        from scipy.optimize import minimize

        if theta0 is None:
            theta0 = self.theta0

        theta0_np = np.asarray(theta0, dtype=np.float64)
        bounds_np = np.asarray(self.bounds)

        @jax.jit
        def val_and_grad(theta):
            return jax.value_and_grad(self.neg_log_posterior)(theta)

        def objective(theta_np):
            theta_jax = jnp.array(theta_np, dtype=jnp.float64)
            val, grad = val_and_grad(theta_jax)
            v = float(val)
            g = np.asarray(grad, dtype=np.float64)
            if np.isnan(v):
                v = 1e30
            g = np.where(np.isnan(g), 0.0, g)
            return v, g

        result = minimize(
            objective, theta0_np, jac=True, method=method,
            bounds=[(float(lo), float(hi)) for lo, hi in bounds_np],
            options={"maxiter": maxiter, "disp": False},
        )

        self.map_estimate = jnp.array(result.x, dtype=jnp.float64)
        self._map_result = result
        return self.map_estimate, result

    # =================================================================
    # Mass matrix estimation: Method 1 — Hessian at MAP
    # =================================================================

    def mass_matrix_hessian_map(self, theta_map=None):
        """
        Estimate the inverse mass matrix from the Hessian of the
        negative log-likelihood at the MAP (excluding the prior, which
        can cause numerical issues at boundaries).

        M^{-1} = H^{-1}  where  H = d^2(-log L)/d theta^2 |_{MAP}

        Parameters
        ----------
        theta_map : array_like, optional
            MAP estimate. If None, calls find_map() first.

        Returns
        -------
        inv_mass_matrix : jnp.ndarray, shape (n_params, n_params)
        """
        if theta_map is None:
            if self.map_estimate is None:
                self.find_map()
            theta_map = self.map_estimate
        else:
            theta_map = jnp.asarray(theta_map, dtype=jnp.float64)

        # Use negative log-likelihood only (not posterior) for stable Hessian
        x, y, yerr = self.x, self.y, self.yerr
        mean_val = self.mean_val
        n_h, n_l, lr, n_o = (self.n_harmonics, self.n_lat,
                              self.lat_range, self.n_omega)
        fit_sn = self.fit_sigma_n
        qn, qw = self._quad_nodes, self._quad_weights

        @jax.jit
        def neg_log_lik(theta_arr):
            return -_gp_log_likelihood(theta_arr, x, y, yerr, mean_val,
                                       n_h, n_l, lr, n_o, fit_sn,
                                       quad_nodes=qn, quad_weights=qw)

        H = jax.hessian(neg_log_lik)(theta_map)

        # Regularise: ensure positive-definite
        eigvals, eigvecs = jnp.linalg.eigh(H)
        eigvals = jnp.maximum(eigvals, 1e-6)
        H_reg = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

        self.inverse_mass_matrix = jnp.linalg.inv(H_reg)
        self._hessian = H
        return self.inverse_mass_matrix

    # =================================================================
    # Mass matrix estimation: Method 2 — Fisher information
    # =================================================================

    def mass_matrix_fisher(self, theta_map=None):
        """
        Estimate the inverse mass matrix from the Fisher information.

        For the GP log-likelihood:

            I_{ij} = (1/2) tr(K^{-1} dK/dtheta_i  K^{-1} dK/dtheta_j)

        The kernel derivatives dK/dtheta_i are computed via JAX forward-mode
        autodiff (jacfwd), which is memory-efficient for few parameters
        mapping to many outputs (N*N).

        Parameters
        ----------
        theta_map : array_like, optional
            Point at which to evaluate Fisher. If None, uses MAP.

        Returns
        -------
        inv_mass_matrix : jnp.ndarray, shape (n_params, n_params)
        """
        if theta_map is None:
            if self.map_estimate is None:
                self.find_map()
            theta_map = self.map_estimate
        else:
            theta_map = jnp.asarray(theta_map, dtype=jnp.float64)

        x = self.x
        N = x.shape[0]
        n_params = theta_map.shape[0]
        lag_flat = self._lag_flat
        fit_sn = self.fit_sigma_n

        qn, qw = self._quad_nodes, self._quad_weights

        def K_noise_flat_from_theta(theta_arr):
            """Return the full K_noise matrix as a flat vector."""
            if fit_sn:
                theta_kernel = theta_arr[:7]
                sigma_n = theta_arr[7]
            else:
                theta_kernel = theta_arr
                sigma_n = 0.0

            K_flat = _kernel_eval(theta_kernel, lag_flat,
                                  self.n_harmonics, self.n_lat,
                                  self.lat_range, self.n_omega,
                                  quad_nodes=qn, quad_weights=qw)
            K = K_flat.reshape(N, N)
            noise_var = self.yerr ** 2 + sigma_n ** 2
            K_noise = K + jnp.diag(noise_var) + 1e-8 * jnp.eye(N)
            return K_noise.ravel()

        # Forward-mode Jacobian: (n_params,) -> (N*N,) gives (N*N, n_params)
        dK_flat_dtheta = jax.jacfwd(K_noise_flat_from_theta)(theta_map)
        dK_dtheta = dK_flat_dtheta.reshape(N, N, n_params)

        # Build K_noise and invert
        K_noise_flat = K_noise_flat_from_theta(theta_map)
        K = K_noise_flat.reshape(N, N)
        K_inv = jnp.linalg.inv(K)

        # K_inv @ dK_i for all i: (N, N) @ (N, N, n_params) -> (N, N, n_params)
        K_inv_dK = jnp.einsum('ab,bcj->acj', K_inv, dK_dtheta)

        # Fisher: I_{ij} = 0.5 * tr(K^{-1} dK_i K^{-1} dK_j)
        fisher = 0.5 * jnp.einsum('abi,baj->ij', K_inv_dK, K_inv_dK)

        # Regularise
        eigvals, eigvecs = jnp.linalg.eigh(fisher)
        eigvals = jnp.maximum(eigvals, 1e-6)
        fisher_reg = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

        self.inverse_mass_matrix = jnp.linalg.inv(fisher_reg)
        self._fisher_matrix = fisher
        return self.inverse_mass_matrix

    # =================================================================
    # Mass matrix estimation: Method 3 — Laplace approximation
    # =================================================================

    def mass_matrix_laplace(self, theta_map=None):
        """
        Laplace approximation: inverse mass matrix = inverse Hessian
        of the negative log-likelihood at the MAP, computed via jax.hessian.

        The posterior is approximated as:

            p(theta | data) ~ N(theta_MAP, H^{-1})

        Parameters
        ----------
        theta_map : array_like, optional
            MAP estimate. If None, calls find_map() first.

        Returns
        -------
        inv_mass_matrix : jnp.ndarray, shape (n_params, n_params)
        """
        if theta_map is None:
            if self.map_estimate is None:
                self.find_map()
            theta_map = self.map_estimate
        else:
            theta_map = jnp.asarray(theta_map, dtype=jnp.float64)

        # Use log-likelihood only for stable Hessian
        x, y, yerr = self.x, self.y, self.yerr
        mean_val = self.mean_val
        n_h, n_l, lr, n_o = (self.n_harmonics, self.n_lat,
                              self.lat_range, self.n_omega)
        fit_sn = self.fit_sigma_n
        qn, qw = self._quad_nodes, self._quad_weights

        @jax.jit
        def neg_log_lik(theta_arr):
            return -_gp_log_likelihood(theta_arr, x, y, yerr, mean_val,
                                       n_h, n_l, lr, n_o, fit_sn,
                                       quad_nodes=qn, quad_weights=qw)

        H = jax.hessian(neg_log_lik)(theta_map)

        # Regularise
        eigvals, eigvecs = jnp.linalg.eigh(H)
        eigvals = jnp.maximum(eigvals, 1e-6)
        H_reg = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

        self.inverse_mass_matrix = jnp.linalg.inv(H_reg)
        self._laplace_hessian = H
        self._laplace_mean = theta_map
        return self.inverse_mass_matrix

    def laplace_samples(self, n_samples=1000, rng_key=None):
        """
        Draw samples from the Laplace (Gaussian) approximation
        to the posterior.

        Parameters
        ----------
        n_samples : int
        rng_key : jax.random.PRNGKey, optional

        Returns
        -------
        samples : jnp.ndarray, shape (n_samples, n_params)
        """
        if self.inverse_mass_matrix is None:
            self.mass_matrix_laplace()

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        mean = self.map_estimate
        cov = self.inverse_mass_matrix

        samples = jax.random.multivariate_normal(
            rng_key, mean, cov, shape=(n_samples,)
        )
        return samples

    # =================================================================
    # BlackJAX NUTS sampling
    # =================================================================

    def run_nuts(self, n_samples=1000, n_warmup=500, theta_init=None,
                 mass_matrix_method="hessian_map", step_size=None,
                 rng_key=None, target_accept=0.8):
        """
        Run BlackJAX NUTS sampler.

        Uses a manual warmup loop with dual averaging for step-size
        adaptation, then a JIT-compiled sampling loop via lax.scan.

        Parameters
        ----------
        n_samples : int
            Number of post-warmup samples (default 1000).
        n_warmup : int
            Number of warmup steps for step-size adaptation (default 500).
        theta_init : array_like, optional
            Initial position. If None, uses MAP estimate.
        mass_matrix_method : {"hessian_map", "fisher", "laplace", "diagonal", None}
            Method to estimate the mass matrix.
        step_size : float, optional
            NUTS step size. If None, adapted via dual averaging.
        rng_key : jax.random.PRNGKey, optional
            Random key. Default: PRNGKey(0).
        target_accept : float
            Target acceptance rate for dual averaging (default 0.8).

        Returns
        -------
        samples : jnp.ndarray, shape (n_samples, n_params)
            Posterior samples.
        info : dict
            Sampling diagnostics.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        # Initial position
        if theta_init is None:
            if self.map_estimate is None:
                self.find_map()
            theta_init = self.map_estimate
        else:
            theta_init = jnp.asarray(theta_init, dtype=jnp.float64)

        # Estimate mass matrix
        inv_mass = self._get_mass_matrix(mass_matrix_method, theta_init)

        # For NUTS, use diagonal mass matrix (more robust than full)
        inv_mass_diag = jnp.diag(inv_mass)
        # Clamp extreme values for stability
        median_var = jnp.median(inv_mass_diag)
        inv_mass_diag = jnp.clip(inv_mass_diag,
                                  median_var * 1e-4, median_var * 1e4)

        # Initial step size: heuristic based on mass matrix scale
        if step_size is None:
            step_size = float(0.5 * jnp.min(jnp.sqrt(inv_mass_diag)))
            step_size = max(step_size, 1e-5)

        # ── Warmup: dual averaging for step size ─────────────────────
        print(f"Warmup: {n_warmup} steps (dual averaging, "
              f"init step_size={step_size:.6f})...")
        log_step = jnp.log(step_size)
        log_step_bar = jnp.log(step_size)
        mu = jnp.log(10.0 * step_size)
        gamma = 0.05
        t0 = 10
        kappa = 0.75
        H_bar = 0.0

        state = blackjax.nuts(
            self.log_posterior,
            step_size=step_size,
            inverse_mass_matrix=inv_mass_diag,
        ).init(theta_init)

        warmup_key, sample_key = jax.random.split(rng_key)

        for m in range(1, n_warmup + 1):
            warmup_key, step_key = jax.random.split(warmup_key)
            current_step = max(float(jnp.exp(log_step)), 1e-10)

            kernel = blackjax.nuts(
                self.log_posterior,
                step_size=current_step,
                inverse_mass_matrix=inv_mass_diag,
            )
            state, step_info = kernel.step(step_key, state)

            accept = float(step_info.acceptance_rate)
            # Dual averaging update (Hoffman & Gelman 2014, Algorithm 5)
            w = 1.0 / (m + t0)
            H_bar = (1 - w) * H_bar + w * (target_accept - accept)
            log_step = mu - jnp.sqrt(m) / gamma * H_bar
            m_w = m ** (-kappa)
            log_step_bar = m_w * log_step + (1 - m_w) * log_step_bar

        final_step_size = max(float(jnp.exp(log_step_bar)), 1e-8)
        print(f"  Adapted step size: {final_step_size:.6f}")

        # ── Sampling: JIT-compiled scan ──────────────────────────────
        print(f"Sampling {n_samples} post-warmup iterations...")

        nuts_kernel = blackjax.nuts(
            self.log_posterior,
            step_size=final_step_size,
            inverse_mass_matrix=inv_mass_diag,
        )

        def one_step(carry, key):
            state = carry
            state, info = nuts_kernel.step(key, state)
            return state, (state, info)

        sample_keys = jax.random.split(sample_key, n_samples)

        final_state, (states, infos) = jax.lax.scan(
            one_step, state, sample_keys
        )

        self.samples = states.position
        self._nuts_info = {
            "divergences": np.asarray(infos.is_divergent),
            "acceptance_rate": np.asarray(infos.acceptance_rate),
            "num_steps": np.asarray(infos.num_integration_steps),
            "step_size": final_step_size,
            "n_warmup": n_warmup,
            "n_samples": n_samples,
            "n_divergent": int(jnp.sum(infos.is_divergent)),
        }

        n_div = self._nuts_info["n_divergent"]
        mean_accept = float(jnp.mean(infos.acceptance_rate))
        print(f"NUTS complete: {n_samples} samples, "
              f"{n_div} divergences, "
              f"mean acceptance rate = {mean_accept:.3f}")

        return self.samples, self._nuts_info

    def _get_mass_matrix(self, method, theta_ref):
        """Compute inverse mass matrix using the specified method."""
        if method is None:
            n = theta_ref.shape[0]
            self.inverse_mass_matrix = jnp.eye(n)
        elif method == "hessian_map":
            self.mass_matrix_hessian_map(theta_ref)
        elif method == "fisher":
            self.mass_matrix_fisher(theta_ref)
        elif method == "laplace":
            self.mass_matrix_laplace(theta_ref)
        elif method == "diagonal":
            H = jax.hessian(self.neg_log_posterior)(theta_ref)
            diag = jnp.maximum(jnp.diag(H), 1e-6)
            self.inverse_mass_matrix = jnp.diag(1.0 / diag)
        else:
            raise ValueError(f"Unknown mass_matrix_method: {method}")

        return self.inverse_mass_matrix

    # =================================================================
    # Diagnostics
    # =================================================================

    def summary(self):
        """
        Print summary statistics of the posterior samples.

        Returns
        -------
        stats : dict
            Parameter names mapped to (mean, std, 16%, 50%, 84%) quantiles.
        """
        if self.samples is None:
            raise RuntimeError("No samples available. Run run_nuts() first.")

        samples = np.asarray(self.samples)
        stats = {}

        print(f"{'param':>12s}  {'mean':>10s}  {'std':>10s}  "
              f"{'16%':>10s}  {'50%':>10s}  {'84%':>10s}")
        print("-" * 68)
        for i, key in enumerate(self.param_keys):
            col = samples[:, i]
            q16, q50, q84 = np.percentile(col, [16, 50, 84])
            m, s = np.mean(col), np.std(col)
            stats[key] = {"mean": m, "std": s,
                          "q16": q16, "q50": q50, "q84": q84}
            print(f"{key:>12s}  {m:10.5f}  {s:10.5f}  "
                  f"{q16:10.5f}  {q50:10.5f}  {q84:10.5f}")

        if self._nuts_info is not None:
            print(f"\nDivergences: {self._nuts_info['n_divergent']}")
            print(f"Mean acceptance: "
                  f"{np.mean(self._nuts_info['acceptance_rate']):.3f}")

        return stats

    def plot_covariance(self, method="fisher", theta_map=None,
                        n_sigma=2, n_grid=200, samples=None,
                        figsize=None, color="C0", alpha=0.3,
                        true_params=None, savefig=None):
        """
        Corner plot of 2D covariance ellipses from the Hessian or Fisher
        matrix at the MAP, with 1D marginal Gaussians on the diagonal.

        Parameters
        ----------
        method : {"fisher", "hessian_map", "laplace"}
            Which matrix to use. If the corresponding matrix has already
            been computed it is reused; otherwise it is computed here.
        theta_map : array_like, optional
            Center of the ellipses. If None, uses self.map_estimate.
        n_sigma : float
            Number of sigma for the ellipse contours (default 2).
        n_grid : int
            Grid resolution for the ellipse curves (default 200).
        samples : array_like, optional
            If provided, scatter MCMC samples behind the ellipses.
        figsize : tuple, optional
            Figure size. If None, auto-scaled to number of parameters.
        color : str
            Color for ellipses and Gaussians (default "C0").
        alpha : float
            Fill alpha for the ellipse interiors (default 0.3).
        true_params : dict or array_like, optional
            True parameter values to mark with crosshairs.
        savefig : str, optional
            If provided, save figure to this path.

        Returns
        -------
        fig, axes : matplotlib Figure and 2D array of Axes.
        """
        import matplotlib.pyplot as plt

        # Get MAP center
        if theta_map is None:
            if self.map_estimate is None:
                self.find_map()
            theta_map = self.map_estimate
        mu = np.asarray(theta_map, dtype=np.float64)

        # Get covariance matrix
        if method == "fisher":
            if not hasattr(self, '_fisher_matrix') or self._fisher_matrix is None:
                self.mass_matrix_fisher(theta_map)
            cov = np.asarray(self.inverse_mass_matrix)
        elif method in ("hessian_map", "laplace"):
            if method == "hessian_map":
                if not hasattr(self, '_hessian') or self._hessian is None:
                    self.mass_matrix_hessian_map(theta_map)
            else:
                if not hasattr(self, '_laplace_hessian') or self._laplace_hessian is None:
                    self.mass_matrix_laplace(theta_map)
            cov = np.asarray(self.inverse_mass_matrix)
        else:
            raise ValueError(f"Unknown method: {method!r}")

        # Parse true_params
        if true_params is not None:
            if isinstance(true_params, dict):
                true_arr = np.array([true_params.get(k, np.nan)
                                     for k in self.param_keys])
            else:
                true_arr = np.asarray(true_params, dtype=np.float64)
        else:
            true_arr = None

        if samples is not None:
            samples = np.asarray(samples)

        n = self.n_params
        keys = self.param_keys
        if figsize is None:
            figsize = (2.5 * n, 2.5 * n)

        fig, axes = plt.subplots(n, n, figsize=figsize)
        if n == 1:
            axes = np.array([[axes]])

        t_ellipse = np.linspace(0, 2 * np.pi, n_grid)

        for i in range(n):
            for j in range(n):
                ax = axes[i, j]

                if j > i:
                    ax.set_visible(False)
                    continue

                if i == j:
                    # Diagonal: 1D marginal Gaussian
                    sigma_i = np.sqrt(cov[i, i])
                    x_range = np.linspace(mu[i] - 4 * sigma_i,
                                          mu[i] + 4 * sigma_i, 300)
                    pdf = np.exp(-0.5 * ((x_range - mu[i]) / sigma_i) ** 2)
                    pdf /= pdf.max()
                    ax.plot(x_range, pdf, color=color, lw=1.5)
                    ax.fill_between(x_range, pdf, alpha=alpha, color=color)
                    ax.axvline(mu[i], color=color, ls="--", lw=0.8)
                    if true_arr is not None and np.isfinite(true_arr[i]):
                        ax.axvline(true_arr[i], color="k", ls=":", lw=1)
                    if samples is not None:
                        ax.hist(samples[:, i], bins=30, density=True,
                                alpha=0.15, color="gray",
                                histtype="stepfilled")
                    ax.set_yticks([])
                else:
                    # Off-diagonal: 2D covariance ellipse
                    sub_cov = np.array([[cov[j, j], cov[j, i]],
                                        [cov[i, j], cov[i, i]]])
                    eigvals, eigvecs = np.linalg.eigh(sub_cov)
                    eigvals = np.maximum(eigvals, 0)

                    for ns in [1, n_sigma]:
                        xy = eigvecs @ np.diag(np.sqrt(eigvals) * ns) @ \
                            np.array([np.cos(t_ellipse), np.sin(t_ellipse)])
                        ax.plot(mu[j] + xy[0], mu[i] + xy[1],
                                color=color, lw=1.2)
                    # Fill the 1-sigma ellipse
                    xy1 = eigvecs @ np.diag(np.sqrt(eigvals)) @ \
                        np.array([np.cos(t_ellipse), np.sin(t_ellipse)])
                    ax.fill(mu[j] + xy1[0], mu[i] + xy1[1],
                            color=color, alpha=alpha)

                    ax.plot(mu[j], mu[i], "+", color=color, ms=8, mew=1.5)
                    if true_arr is not None:
                        if np.isfinite(true_arr[j]) and np.isfinite(true_arr[i]):
                            ax.plot(true_arr[j], true_arr[i], "x",
                                    color="k", ms=6, mew=1.2)
                    if samples is not None:
                        ax.scatter(samples[:, j], samples[:, i],
                                   s=1, alpha=0.1, color="gray",
                                   rasterized=True)

                # Axis labels
                if i == n - 1:
                    ax.set_xlabel(keys[j])
                else:
                    ax.set_xticklabels([])
                if j == 0 and i != 0:
                    ax.set_ylabel(keys[i])
                elif j != 0:
                    ax.set_yticklabels([])

        fig.tight_layout()

        if savefig is not None:
            fig.savefig(savefig, dpi=150, bbox_inches="tight")

        return fig, axes

    def to_dict(self, samples=None):
        """
        Convert samples array to a dict keyed by parameter name.

        Parameters
        ----------
        samples : jnp.ndarray, optional
            Shape (n_samples, n_params). If None, uses self.samples.

        Returns
        -------
        d : dict
            {param_name: array of shape (n_samples,)}
        """
        if samples is None:
            samples = self.samples
        if samples is None:
            raise RuntimeError("No samples available.")
        samples = np.asarray(samples)
        return {k: samples[:, i] for i, k in enumerate(self.param_keys)}
