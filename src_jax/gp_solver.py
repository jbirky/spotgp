"""
JAX-accelerated Gaussian Process solver using the starspot analytic kernel.

Uses JAX for JIT-compiled covariance matrix construction, Cholesky
factorization, and GP operations (log-likelihood, prediction, MAP
estimation, and mass matrix computation for MCMC).
"""
import os
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", "cpu"))

import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

try:
    from .analytic_kernel import (
        AnalyticKernel, _REQUIRED_KEYS, _AMPLITUDE_KEYS_SIGMA,
        _AMPLITUDE_KEYS_PHYSICAL, _R_Gamma, _cn_general_jax,
        _gauss_legendre_grid,
    )
    from .banded_cholesky import (banded_cholesky, banded_solve,
                                   banded_cholesky_compact, banded_solve_compact)
except ImportError:
    from analytic_kernel import (
        AnalyticKernel, _REQUIRED_KEYS, _AMPLITUDE_KEYS_SIGMA,
        _AMPLITUDE_KEYS_PHYSICAL, _R_Gamma, _cn_general_jax,
        _gauss_legendre_grid,
    )
    from banded_cholesky import (banded_cholesky, banded_solve,
                                 banded_cholesky_compact, banded_solve_compact)

__all__ = ["GPSolver"]

# Fixed parameter ordering for the theta vector.
# The kernel always works in terms of sigma_k (the amplitude prefactor).
# Users who prefer (nspot, fspot, alpha_max) should compute sigma_k via
# sigma_k = sqrt(nspot) * (1 - fspot) * alpha_max^2 / pi  (see AnalyticKernel).
KERNEL_HPARAM_KEYS = ["peq", "kappa", "inc", "lspot", "tau", "sigma_k"]

# Full hyperparameter keys including optional white noise
HPARAM_KEYS_WITH_NOISE = KERNEL_HPARAM_KEYS + ["sigma_n"]


# =====================================================================
# Pure-functional GP helpers (module-level for clean JAX tracing)
# =====================================================================

def _kernel_eval(theta_arr, lag_flat, n_harmonics, n_lat, lat_range,
                  quad_nodes=None, quad_weights=None):
    """
    Pure-functional kernel evaluation: theta_arr -> kernel values.

    Uses ``jax.vmap`` over the latitude grid so that all latitudes are
    processed in parallel.  cn_sq coefficients for all latitudes are
    precomputed with a double-vmap (outer: latitudes, inner: harmonics)
    before entering the per-latitude cosine sum, avoiding repeated vmap
    traces inside a sequential loop.

    Memory scales as O(n_lat * M) instead of O(M), which is acceptable
    on GPU where M = N*bandwidth is small relative to device memory.

    Parameters
    ----------
    theta_arr : jnp.ndarray, shape (6,)
        [peq, kappa, inc, lspot, tau, sigma_k]
    lag_flat : jnp.ndarray, shape (M,)
        Flattened time lags.
    n_harmonics, n_lat, lat_range : kernel config (static).
    quad_nodes, quad_weights : jnp.ndarray or None
        Gauss-Legendre nodes and weights. If None, uses trapezoid rule.

    Returns
    -------
    K_flat : jnp.ndarray, shape (M,)
        Kernel values at each lag.
    """
    peq, kappa, inc, lspot, tau, sigma_k = theta_arr

    # R_Gamma (independent of latitude, closed-form piecewise polynomial)
    R = _R_Gamma(lag_flat, lspot, tau)

    # Quadrature grid
    if quad_nodes is not None:
        phi_grid = quad_nodes
        weights = quad_weights
        norm = jnp.sum(weights)
    else:
        phi_min, phi_max = lat_range
        phi_grid = jnp.linspace(phi_min, phi_max, n_lat)
        dphi = phi_grid[1] - phi_grid[0]
        weights = jnp.ones_like(phi_grid) * dphi
        norm = phi_max - phi_min

    # Precompute cn_sq for all latitudes at once: shape (n_lat, n_harmonics+1).
    # Outer vmap over latitudes, inner vmap over harmonics — avoids tracing
    # the inner vmap once per latitude step as was the case inside scan.
    ns = jnp.arange(n_harmonics + 1)
    cn_sq_all = jax.vmap(
        lambda phi: jax.vmap(lambda n: _cn_general_jax(n, inc, phi))(ns) ** 2
    )(phi_grid)  # (n_lat, n_harmonics+1)

    harm_ns = jnp.arange(1, n_harmonics + 1)

    # Per-latitude cosine sum given pre-computed cn_sq: returns shape (M,)
    def _lat_contribution(phi, cn_sq):
        w0 = 2 * jnp.pi * (1 - kappa * jnp.sin(phi) ** 2) / peq
        cosine_terms = jnp.sum(
            cn_sq[1:] * jnp.cos(harm_ns * w0 * lag_flat[:, None]), axis=1
        )
        return cn_sq[0] + 2 * cosine_terms

    # vmap over all latitudes simultaneously: shape (n_lat, M)
    all_contribs = jax.vmap(_lat_contribution)(phi_grid, cn_sq_all)

    # Weighted sum over latitudes: shape (M,)
    K = jnp.sum(weights[:, None] * all_contribs, axis=0) / norm

    K = R * K * sigma_k ** 2
    return K


def _gp_log_likelihood(theta_full, x, y, yerr, mean_val,
                       n_harmonics, n_lat, lat_range,
                       fit_sigma_n, quad_nodes=None, quad_weights=None):
    """
    Pure-functional GP marginal log-likelihood.

    Exploits the symmetry of the covariance matrix by evaluating the
    kernel only on the upper-triangular lags (N*(N+1)/2 instead of N^2),
    halving memory and compute for the kernel evaluation step.

    Parameters
    ----------
    theta_full : jnp.ndarray, shape (6,) or (7,)
        Kernel params [peq, kappa, inc, lspot, tau, sigma_k],
        optionally followed by sigma_n (white noise amplitude).
    x, y, yerr : jnp.ndarray
        Observations.
    mean_val : float
        Constant mean.
    n_harmonics, n_lat, lat_range : kernel config.
    fit_sigma_n : bool
        If True, theta_full has 8 elements (last is sigma_n).
    quad_nodes, quad_weights : jnp.ndarray or None
        Gauss-Legendre nodes/weights. If None, uses trapezoid rule.

    Returns
    -------
    logL : scalar
    """
    N = x.shape[0]

    n_kernel = 6
    if fit_sigma_n:
        theta_kernel = theta_full[:n_kernel]
        sigma_n = theta_full[n_kernel]
    else:
        theta_kernel = theta_full
        sigma_n = 0.0

    # Upper-triangular indices (includes diagonal)
    row_idx, col_idx = jnp.triu_indices(N)
    lag_upper = jnp.abs(x[row_idx] - x[col_idx])

    K_upper = _kernel_eval(theta_kernel, lag_upper,
                           n_harmonics, n_lat, lat_range,
                           quad_nodes=quad_nodes, quad_weights=quad_weights)

    # Reconstruct symmetric matrix from upper triangle
    K = jnp.zeros((N, N))
    K = K.at[row_idx, col_idx].set(K_upper)
    K = K + K.T - jnp.diag(jnp.diag(K))

    noise_var = yerr ** 2 + sigma_n ** 2
    K_noise = K + jnp.diag(noise_var) + 1e-8 * jnp.eye(N)

    L = jla.cholesky(K_noise, lower=True)
    resid = y - mean_val
    alpha = jla.cho_solve((L, True), resid)

    data_fit = resid @ alpha
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    return -0.5 * (data_fit + log_det + N * jnp.log(2 * jnp.pi))


def _build_banded_kernel_jax(theta_kernel, x, bandwidth,
                              n_harmonics, n_lat, lat_range,
                              quad_nodes=None, quad_weights=None):
    """
    Build the kernel covariance directly in compact (b+1, N) banded storage.

    Evaluates only the O(N*b) within-band lags instead of the full N^2 matrix.

    Parameters
    ----------
    theta_kernel : jnp.ndarray, shape (6,)
        Kernel hyperparameters.
    x : jnp.ndarray, shape (N,)
        Observation times.
    bandwidth : int
        Number of sub-diagonals.
    n_harmonics, n_lat, lat_range : kernel config.
    quad_nodes, quad_weights : jnp.ndarray or None

    Returns
    -------
    cb : jnp.ndarray, shape (b+1, N)
        Compact lower-banded storage: ``cb[d, i] = K(x[i+d], x[i])``.
    """
    N = x.shape[0]
    b = bandwidth

    # Build index arrays: for diagonal d, entry i has lag |x[i+d] - x[i]|
    d_idx = jnp.arange(b + 1)[:, None]   # (b+1, 1)
    i_idx = jnp.arange(N)[None, :]       # (1, N)
    j_idx = i_idx + d_idx                 # (b+1, N)  row index = i+d
    j_safe = jnp.minimum(j_idx, N - 1)
    valid = j_idx < N                     # (b+1, N)

    lags_flat = jnp.abs(x[j_safe] - x[i_idx]).ravel()  # ((b+1)*N,)
    K_flat = _kernel_eval(theta_kernel, lags_flat,
                          n_harmonics, n_lat, lat_range,
                          quad_nodes=quad_nodes, quad_weights=quad_weights)
    cb = K_flat.reshape(b + 1, N)
    cb = jnp.where(valid, cb, 0.0)
    return cb


def _gp_log_likelihood_banded(theta_full, x, y, yerr, mean_val,
                               n_harmonics, n_lat, lat_range,
                               fit_sigma_n, bandwidth,
                               quad_nodes=None, quad_weights=None):
    """
    Pure-functional GP marginal log-likelihood using banded Cholesky.

    Builds the covariance directly in compact (b+1, N) banded storage,
    evaluating only the O(N*b) within-band lags.  The Cholesky factorization
    and solve operate entirely in compact storage — no (N, N) matrix is
    ever allocated.

    ``bandwidth`` must be a Python ``int`` (captured in the JIT closure
    at trace time so the inner scan loop can be unrolled statically).

    Parameters
    ----------
    bandwidth : int
        Number of sub-diagonals to retain (compile-time constant).
    All other parameters are the same as ``_gp_log_likelihood``.
    """
    N = x.shape[0]

    n_kernel = 6
    if fit_sigma_n:
        theta_kernel = theta_full[:n_kernel]
        sigma_n = theta_full[n_kernel]
    else:
        theta_kernel = theta_full
        sigma_n = 0.0

    # Build covariance in compact banded storage: O(N*b) instead of O(N^2)
    cb = _build_banded_kernel_jax(theta_kernel, x, bandwidth,
                                   n_harmonics, n_lat, lat_range,
                                   quad_nodes=quad_nodes,
                                   quad_weights=quad_weights)

    # Add noise to diagonal (row 0 of compact storage)
    noise_var = yerr ** 2 + sigma_n ** 2
    cb = cb.at[0, :].add(noise_var + 1e-8)

    # Factorize and solve in compact storage
    Lc = banded_cholesky_compact(cb, bandwidth)
    resid = y - mean_val
    alpha = banded_solve_compact(Lc, resid, bandwidth)

    data_fit = resid @ alpha
    log_det = 2.0 * jnp.sum(jnp.log(Lc[0, :]))

    return -0.5 * (data_fit + log_det + N * jnp.log(2 * jnp.pi))


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


def _validate_hparam(hparam):
    """Validate hparam dict (shared by __init__ and update_hparam)."""
    if not isinstance(hparam, dict):
        raise TypeError("hparam must be a dict")
    missing = _REQUIRED_KEYS - set(hparam.keys())
    if missing:
        raise ValueError(f"hparam dict is missing required keys: {missing}")
    has_sigma = _AMPLITUDE_KEYS_SIGMA <= set(hparam.keys())
    has_physical = _AMPLITUDE_KEYS_PHYSICAL <= set(hparam.keys())
    if not has_sigma and not has_physical:
        raise ValueError(
            "hparam must contain either 'sigma_k' or both 'nspot' and 'fspot'")


# =====================================================================
# GPSolver class
# =====================================================================

class GPSolver:
    """
    JAX-accelerated Gaussian Process solver for stellar lightcurves.

    Handles covariance matrix construction, Cholesky factorization,
    log-likelihood evaluation, prediction, MAP estimation, and mass
    matrix computation.

    Parameters
    ----------
    x : array_like, shape (N,)
        Observation times [days].
    y : array_like, shape (N,)
        Observed flux values.
    yerr : array_like, shape (N,) or float
        Measurement uncertainties (1-sigma).
    hparam : dict
        Kernel hyperparameters. Required keys: peq, kappa, inc, lspot,
        tau. For amplitude, provide either sigma_k directly or all of
        nspot, fspot, and alpha_max (sigma_k computed automatically).
    kernel_type : {"analytic"}
        Which kernel to use (default: "analytic").
    mean : float or callable or None
        Mean function.
    fit_sigma_n : bool
        If True, include white noise amplitude sigma_n as a free
        parameter for optimization/sampling (default False).
    bounds : dict or None
        Parameter bounds for optimization. If None, uses defaults.
    log_prior : callable or None
        Custom log-prior function f(theta_arr) -> scalar.
        If None, uses soft uniform within bounds.
    kernel_kwargs : dict
        Extra kwargs forwarded to the kernel constructor.
    """

    DEFAULT_BOUNDS = {
        "peq":       (0.5, 50.0),
        "kappa":     (0.001, 0.999),
        "inc":       (0.01, np.pi - 0.01),
        "lspot":     (0.1, 20.0),
        "tau":       (0.05, 10.0),
        "sigma_k":   (1e-6, 1.0),
        "sigma_n":   (1e-6, 0.1),
    }

    def __init__(self, x, y, yerr, hparam, kernel_type="analytic",
                 mean=None, fit_sigma_n=False, bounds=None,
                 log_prior=None, matrix_solver="cholesky_banded",
                 **kernel_kwargs):

        self.x = jnp.asarray(x, dtype=jnp.float64)
        self.y = jnp.asarray(y, dtype=jnp.float64)
        yerr_arr = jnp.atleast_1d(jnp.asarray(yerr, dtype=jnp.float64))
        if yerr_arr.size == 1:
            self.yerr = jnp.full_like(self.x, yerr_arr.item())
        else:
            self.yerr = yerr_arr
        self.N = len(self.x)

        # Validate and store hyperparameters
        _validate_hparam(hparam)
        self.hparam = dict(hparam)

        # Mean function
        if mean is None:
            self.mean_val = float(jnp.mean(self.y))
            self.mean_func = lambda t: self.mean_val
        elif callable(mean):
            self.mean_func = mean
            self.mean_val = float(mean(self.x[0]))
        else:
            self.mean_val = float(mean)
            self.mean_func = lambda t: self.mean_val

        # Matrix solver
        _valid_solvers = ("cholesky_full", "cholesky_banded")
        if matrix_solver not in _valid_solvers:
            raise ValueError(
                f"matrix_solver must be one of {_valid_solvers}, "
                f"got '{matrix_solver}'")
        self.matrix_solver = matrix_solver

        # Build kernel
        self.kernel_type = kernel_type
        self.kernel_kwargs = kernel_kwargs
        self._build_kernel()

        # Update hparam with computed sigma_k (if nspot/fspot were given)
        self.hparam = dict(self.kernel.hparam)

        # Optimization/sampling config — must be set before bounds parsing
        # so that the log-param remapping can identify which physical keys
        # to replace with their log-space counterparts.
        self.fit_sigma_n = fit_sigma_n
        _base_keys = HPARAM_KEYS_WITH_NOISE if fit_sigma_n else KERNEL_HPARAM_KEYS

        # Detect log-space parameters: keys prefixed with "log_" in the
        # bounds dict indicate sampling in log10 space.  The physical key
        # is the suffix (e.g. "log_sigma_k" → physical key "sigma_k").
        self._log_param_map = {}  # {log_key: phys_key}
        if isinstance(bounds, dict):
            for k in bounds:
                if k.startswith("log_"):
                    self._log_param_map[k] = k[4:]

        # Remap param_keys: replace physical keys with their log-space names
        # where applicable.  All other code uses self.param_keys, so this
        # single remap propagates everywhere (MAP dict keys, corner labels, etc.)
        _phys_to_log = {v: k for k, v in self._log_param_map.items()}
        self.param_keys = tuple(_phys_to_log.get(k, k) for k in _base_keys)
        self.n_params = len(self.param_keys)

        # Parse bounds — must precede _compute_bandwidth so that the upper
        # bounds of lspot and tau are available for the bandwidth calculation.
        # For log-space keys the supplied bounds are already in log10 units;
        # for physical keys without explicit bounds, fall back to DEFAULT_BOUNDS.
        if bounds is None:
            self.bounds = jnp.array(
                [self.DEFAULT_BOUNDS[k] for k in _base_keys],
                dtype=jnp.float64)
        elif isinstance(bounds, dict):
            self.bounds = jnp.array(
                [bounds.get(_pk, self.DEFAULT_BOUNDS[_bk])
                 for _pk, _bk in zip(self.param_keys, _base_keys)],
                dtype=jnp.float64)
        else:
            self.bounds = jnp.asarray(bounds, dtype=jnp.float64)

        # Bandwidth for banded solver (compile-time constant derived from
        # prior upper bounds so it is sufficient for any sampled parameters).
        if self.matrix_solver == "cholesky_banded":
            self.bandwidth = self._compute_bandwidth()
            _n_banded = (self.bandwidth + 1) * self.N
            _n_full   = self.N * self.N
            _sparsity = 100.0 * (1.0 - _n_banded / _n_full)
            print(f"Banded Cholesky: bandwidth={self.bandwidth}, "
                  f"N={self.N}, sparsity={_sparsity:.1f}%")

        # Build covariance and factorize
        self._build_covariance()

        # Initial theta: physical hparam values, converted to log10 for any
        # log-parameterized keys.
        self.theta0 = jnp.array([
            np.log10(float(self.hparam.get(self._log_param_map[k], 0.0)))
            if k in self._log_param_map
            else float(self.hparam.get(k, 0.0))
            for k in self.param_keys
        ], dtype=jnp.float64)

        # Kernel config (extract from kernel object)
        self.n_harmonics = self.kernel.n_harmonics
        self.n_lat = self.kernel.n_lat
        self.lat_range = self.kernel.lat_range

        # Quadrature nodes — precomputed at init so they are captured as
        # array constants in JIT closures rather than recomputed per trace.
        #
        # GL weights are pre-normalized (sum = 1.0) so the /norm division
        # inside _kernel_eval reduces to /1.0, which XLA eliminates.
        #
        # Trapezoid weights are left as None: lat_range and n_lat are Python
        # scalars captured in the JIT closure, so XLA constant-folds the
        # linspace and weight construction at compile time anyway.  Pre-
        # normalizing trapezoid weights is non-trivial because jnp.sum(dphi*
        # ones) = n_lat*dphi != phi_max-phi_min = (n_lat-1)*dphi, which
        # would silently change the kernel normalization by n_lat/(n_lat-1).
        if self.kernel.quadrature == "gauss-legendre":
            raw_w = self.kernel._quad_weights
            _norm = float(jnp.sum(raw_w))
            self._quad_nodes = self.kernel._quad_nodes
            self._quad_weights = raw_w / _norm   # sum = 1.0
        else:  # trapezoid: keep None, XLA constant-folds the else branch
            self._quad_nodes = None
            self._quad_weights = None

        # Prior
        self._custom_log_prior = log_prior

        # Lag matrix is computed lazily (only needed for Fisher info)
        self._lag_flat = None

        # Build JAX function that maps sampling theta → physical theta
        # (applies 10^x for any log-parameterized indices).
        self._build_transform()

        # Build JIT-compiled log-posterior
        self._build_logposterior()

        # Storage for optimization results
        self.map_estimate = None
        self.inverse_mass_matrix = None
        self._hessian = None
        self._fisher_matrix = None
        self._laplace_hessian = None
        self._laplace_mean = None

    def _build_kernel(self):
        """Instantiate the kernel object."""
        if self.kernel_type == "analytic":
            self.kernel = AnalyticKernel(self.hparam, **self.kernel_kwargs)
        else:
            raise ValueError(
                f"GPSolver only supports 'analytic' kernel, "
                f"got '{self.kernel_type}'")

    def _build_transform(self):
        """Build _to_physical: sampling theta → physical theta.

        For each log-parameterized key (e.g. ``log_sigma_k``), the
        corresponding element of the theta array is stored in log10 space
        during sampling/optimization.  This function raises 10 to the power
        of those elements so that the kernel always receives physical values.

        If no log-space parameters are present, ``_to_physical`` is the
        identity function (zero overhead).
        """
        if not self._log_param_map:
            self._to_physical = lambda x: x
            return

        keys = list(self.param_keys)
        log_indices = jnp.array(
            [keys.index(k) for k in self._log_param_map], dtype=jnp.int32)

        @jax.jit
        def to_physical(theta_arr):
            return theta_arr.at[log_indices].set(10.0 ** theta_arr[log_indices])

        self._to_physical = to_physical

    def _eval_kernel(self, tau):
        """Evaluate the kernel at time lags tau."""
        tau = jnp.asarray(tau, dtype=float)
        return jnp.asarray(self.kernel.kernel(jnp.abs(tau)))

    def _compute_bandwidth(self):
        """
        Bandwidth in samples for the banded Cholesky solver.

        Derived from the **upper bounds** of ``lspot`` and ``tau`` so that
        the banded approximation is valid for any parameters within the
        prior, regardless of the current ``hparam`` values.  This is
        required for JIT-compiled sampling: the bandwidth is a compile-time
        constant (it determines array shapes), so it must cover the entire
        prior support rather than just the initial hyperparameters.

        Assumes uniform sampling; uses ``x[1] - x[0]`` as the step size.
        """
        if self.N < 2:
            return self.N
        dt = float(self.x[1] - self.x[0])
        keys = list(self.param_keys)

        # Resolve upper bound for lspot and tau, accounting for the
        # possibility that either is sampled in log10 space.
        if "lspot" in keys:
            lspot_max = float(self.bounds[keys.index("lspot"), 1])
        elif "log_lspot" in keys:
            lspot_max = 10.0 ** float(self.bounds[keys.index("log_lspot"), 1])
        else:
            lspot_max = float(self.DEFAULT_BOUNDS["lspot"][1])

        if "tau" in keys:
            tau_max = float(self.bounds[keys.index("tau"), 1])
        elif "log_tau" in keys:
            tau_max = 10.0 ** float(self.bounds[keys.index("log_tau"), 1])
        else:
            tau_max = float(self.DEFAULT_BOUNDS["tau"][1])

        b = int(np.ceil((lspot_max + 2.0 * tau_max) / dt))
        return min(b, self.N - 1)

    def _build_covariance(self):
        """Build the covariance matrix and Cholesky-factorize it.

        For ``cholesky_full``: builds the full N x N matrix (upper-triangle
        symmetry trick, N*(N+1)/2 kernel evals).

        For ``cholesky_banded``: builds directly in compact (b+1, N) banded
        storage, evaluating only the O(N*b) within-band entries.  No (N, N)
        matrix is ever allocated; ``self.K`` and ``self.K_noise`` are set
        to ``None``.
        """
        mu = self.mean_func(self.x)
        if jnp.isscalar(mu):
            self._mu = jnp.full(self.N, mu)
        else:
            self._mu = jnp.asarray(mu)
        self._resid = self.y - self._mu

        if self.matrix_solver == "cholesky_banded":
            # Build kernel directly in compact (b+1, N) storage
            N = self.N
            b = self.bandwidth
            d_idx = jnp.arange(b + 1)[:, None]   # (b+1, 1)
            i_idx = jnp.arange(N)[None, :]       # (1, N)
            j_idx = i_idx + d_idx                 # (b+1, N)
            j_safe = jnp.minimum(j_idx, N - 1)
            valid = j_idx < N

            lags_flat = jnp.abs(self.x[j_safe] - self.x[i_idx]).ravel()
            K_flat = self._eval_kernel(lags_flat)
            cb = K_flat.reshape(b + 1, N)
            cb = jnp.where(valid, cb, 0.0)

            # Add noise to diagonal (row 0)
            cb = cb.at[0, :].add(self.yerr ** 2 + 1e-8)

            self.K = None
            self.K_noise = None
            self._Lc = banded_cholesky_compact(cb, self.bandwidth)
            self._L = None
            self._alpha = banded_solve_compact(
                self._Lc, self._resid, self.bandwidth)
        else:
            N = self.N
            row_idx, col_idx = jnp.triu_indices(N)
            lag_upper = jnp.abs(self.x[row_idx] - self.x[col_idx])
            K_upper = self._eval_kernel(lag_upper)

            K = jnp.zeros((N, N))
            K = K.at[row_idx, col_idx].set(K_upper)
            K = K + K.T - jnp.diag(jnp.diag(K))
            self.K = K
            self.K_noise = K + jnp.diag(self.yerr ** 2) + 1e-8 * jnp.eye(N)

            self._Lc = None
            self._L = jla.cholesky(self.K_noise, lower=True)
            self._alpha = jla.cho_solve((self._L, True), self._resid)

    def _build_logposterior(self):
        """Build JIT-compiled log-posterior and its gradient."""
        bounds = self.bounds
        x, y, yerr = self.x, self.y, self.yerr
        mean_val = self.mean_val
        n_h, n_l, lr = self.n_harmonics, self.n_lat, self.lat_range
        custom_prior = self._custom_log_prior
        fit_sn = self.fit_sigma_n
        qn, qw = self._quad_nodes, self._quad_weights
        to_phys = self._to_physical  # sampling theta → physical theta

        if self.matrix_solver == "cholesky_banded":
            # Capture bandwidth as a Python int in the closure so that
            # jax.lax.scan inside banded_cholesky is unrolled statically.
            b = self.bandwidth

            @jax.jit
            def log_posterior(theta_arr):
                ll = _gp_log_likelihood_banded(
                    to_phys(theta_arr), x, y, yerr, mean_val,
                    n_h, n_l, lr, fit_sn, b,
                    quad_nodes=qn, quad_weights=qw)
                lp = (custom_prior(theta_arr) if custom_prior is not None
                      else _default_log_prior(theta_arr, bounds))
                return ll + lp
        else:
            @jax.jit
            def log_posterior(theta_arr):
                ll = _gp_log_likelihood(to_phys(theta_arr), x, y, yerr, mean_val,
                                        n_h, n_l, lr, fit_sn,
                                        quad_nodes=qn, quad_weights=qw)
                lp = (custom_prior(theta_arr) if custom_prior is not None
                      else _default_log_prior(theta_arr, bounds))
                return ll + lp

        @jax.jit
        def neg_log_posterior(theta_arr):
            return -log_posterior(theta_arr)

        self.log_posterior = log_posterior
        self.neg_log_posterior = neg_log_posterior
        self.grad_log_posterior = jax.jit(jax.grad(log_posterior))
        self.grad_neg_log_posterior = jax.jit(jax.grad(neg_log_posterior))

    # =================================================================
    # Log-likelihood
    # =================================================================

    def log_likelihood(self):
        """
        Marginal log-likelihood of the data under the GP.

        Returns
        -------
        logL : float
            log p(y | X, theta)
        """
        data_fit = self._resid @ self._alpha
        if self._Lc is not None:
            log_det = 2.0 * jnp.sum(jnp.log(self._Lc[0, :]))
        else:
            log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(self._L)))
        return float(
            -0.5 * (data_fit + log_det + self.N * jnp.log(2 * jnp.pi)))

    # =================================================================
    # Prediction
    # =================================================================

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

        lag_cross = jnp.abs(xpred[:, None] - self.x[None, :])
        Ks = self._eval_kernel(lag_cross)

        mu_prior = self.mean_func(xpred)
        if jnp.isscalar(mu_prior):
            mu_prior = jnp.full(M, mu_prior)
        mu_pred = mu_prior + Ks @ self._alpha

        if self.matrix_solver == "cholesky_banded":
            V = banded_solve_compact(self._Lc, Ks.T, self.bandwidth)
        else:
            V = jla.cho_solve((self._L, True), Ks.T)

        if return_cov:
            lag_pred = jnp.abs(xpred[:, None] - xpred[None, :])
            Kss = self._eval_kernel(lag_pred)
            cov_pred = Kss - Ks @ V
            return np.asarray(mu_pred), np.asarray(cov_pred)
        else:
            # Only need diagonal: k(0) - sum_j V_ji * Ks_ij
            k0 = self._eval_kernel(jnp.zeros(1))[0]
            var_pred = k0 - jnp.einsum('ij,ji->i', Ks, V)
            return np.asarray(mu_pred), np.asarray(var_pred)

    def plot_prediction(self, theta=None, n_points=2000, n_sigma=(1, 2),
                        ax=None, data_color="k", model_color="r",
                        show_legend=True):
        """
        Plot the GP posterior mean and uncertainty bands over the data.

        If ``theta`` is provided the GP is temporarily updated to those
        hyperparameters before predicting, so the prediction reflects the
        given parameter values rather than whatever was last set internally.

        Parameters
        ----------
        theta : dict or array_like, shape (6,), optional
            Kernel parameters.  Accepts a physical dict with keys from
            ``KERNEL_HPARAM_KEYS``, a sampling-space dict with ``log_``-
            prefixed keys (e.g. ``log_sigma_k``), or a length-6 array.
            If None, uses the current internal hyperparameters.
        n_points : int
            Number of prediction points spanning the data baseline.
        n_sigma : int or sequence of int
            Which sigma levels to shade.  E.g. ``(1, 2)`` draws both
            ±1σ and ±2σ bands (default).  Pass a single int for one band.
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates a new figure.
        data_color, model_color : str
            Colors for data points and model curve/bands.
        show_legend : bool
            Whether to draw a legend.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if theta is not None:
            phys = {}
            for k, v in (theta.items() if isinstance(theta, dict)
                         else zip(KERNEL_HPARAM_KEYS, theta)):
                if isinstance(k, str) and k.startswith("log_"):
                    phys[k[4:]] = 10.0 ** float(v)
                else:
                    phys[k] = float(v)
            self.update_hparam(phys)

        xpred = np.linspace(float(self.x[0]), float(self.x[-1]), n_points)
        mu, var = self.predict(xpred)
        sigma = np.sqrt(np.maximum(var, 0.0))

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        ax.errorbar(np.asarray(self.x), np.asarray(self.y),
                    yerr=np.asarray(self.yerr),
                    fmt=".", color=data_color, capsize=0, alpha=0.5,
                    label="Data")
        ax.plot(xpred, mu, color=model_color, lw=1.5, label="MAP mean")

        alphas = {1: 0.35, 2: 0.18, 3: 0.10}
        for ns in (n_sigma if hasattr(n_sigma, "__iter__") else (n_sigma,)):
            ax.fill_between(xpred, mu - ns * sigma, mu + ns * sigma,
                            color=model_color,
                            alpha=alphas.get(ns, 0.15),
                            label=rf"$\pm{ns}\sigma$")

        ax.set_xlabel("Time [days]", fontsize=22)
        ax.set_ylabel("Flux", fontsize=22)
        if show_legend:
            ax.legend()

        return ax

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

    # =================================================================
    # ACF and kernel evaluation
    # =================================================================

    def compute_acf(self, tlags, normalize=True):
        """
        Compute the empirical autocorrelation function of (x, y, yerr)
        by binning data pairs into time-lag bins.

        Parameters
        ----------
        tlags : array_like, shape (M,)
            Bin edges for time lags [days]. The ACF is evaluated at
            bin centers: 0.5*(tlags[:-1] + tlags[1:]).
        normalize : bool
            If True (default), normalize so ACF(0) = variance of y.

        Returns
        -------
        lag_centers : ndarray, shape (M-1,)
            Bin centers.
        acf : ndarray, shape (M-1,)
            Empirical ACF at each bin center.
        """
        tlags = np.asarray(tlags, dtype=np.float64)
        x = np.asarray(self.x)
        resid = np.asarray(self.y) - float(self.mean_val)

        n_bins = len(tlags) - 1
        acf_sum = np.zeros(n_bins)
        acf_count = np.zeros(n_bins)

        # Compute all pairwise lags and products
        dt = np.abs(x[:, None] - x[None, :])
        prod = resid[:, None] * resid[None, :]

        for k in range(n_bins):
            mask = (dt >= tlags[k]) & (dt < tlags[k + 1])
            acf_count[k] = np.sum(mask)
            if acf_count[k] > 0:
                acf_sum[k] = np.sum(prod[mask]) / acf_count[k]

        lag_centers = 0.5 * (tlags[:-1] + tlags[1:])

        if normalize:
            var = np.mean(resid ** 2)
            if var > 0:
                acf_sum = acf_sum / var

        return lag_centers, acf_sum

    def compute_kernel(self, tlags):
        """
        Evaluate the analytic kernel at the given time lags.

        Parameters
        ----------
        tlags : array_like, shape (M,)
            Time lags [days].

        Returns
        -------
        K : ndarray, shape (M,)
            Kernel values at each lag.
        """
        tlags = jnp.asarray(tlags, dtype=jnp.float64)
        return np.asarray(self._eval_kernel(jnp.abs(tlags)))

    def fit_acf(self, theta0=None, keys=None, tlags=None, n_bins=50,
                method="L-BFGS-B", maxiter=500, ftol=0, gtol=1e-8,
                disp=False):
        """
        Fit the analytic kernel to the empirical ACF via least-squares.

        Minimizes sum_i (ACF_data(lag_i) - K(lag_i; theta))^2 over the
        kernel hyperparameters, using JAX gradients and scipy.

        Parameters
        ----------
        theta0 : dict or array_like, optional
            Starting point. Can be:
              - None: uses self.theta0 (kernel params only, no sigma_n).
              - dict: values for any subset of kernel keys set the
                starting point. If ``keys`` is not given, the dict
                keys that overlap with ``KERNEL_HPARAM_KEYS`` are
                treated as the free variables; the rest are held fixed.
                Extra keys not in ``KERNEL_HPARAM_KEYS`` are ignored.
              - array_like: full kernel theta vector (6 elements).
        keys : list of str, optional
            Which parameters to vary during optimization. Overrides
            the automatic inference from a dict ``theta0``. Parameters
            not listed are held fixed at their current values. If None
            and theta0 is not a dict, all kernel parameters are varied.
        tlags : array_like, optional
            Bin edges for compute_acf. If None, linearly spaced from 0 to
            half the baseline with n_bins+1 edges.
        n_bins : int
            Number of lag bins (used when tlags is None).
        method : str
            Scipy optimizer method.
        maxiter : int
            Maximum iterations.
        ftol : float
            Function-value convergence tolerance (default 0, disabled).
        gtol : float
            Gradient-norm convergence tolerance (default 1e-8).
        disp : bool
            If True, print optimizer convergence messages (default False).

        Returns
        -------
        theta_dict : dict
            Full dictionary of all kernel hyperparameters (fixed + optimized).
        result : scipy.optimize.OptimizeResult
            Full optimizer output.
        """
        from scipy.optimize import minimize

        # Build lag bins
        if tlags is None:
            baseline = float(jnp.max(self.x) - jnp.min(self.x))
            tlags = np.linspace(0, baseline / 2, n_bins + 1)

        # Compute empirical ACF (unnormalized so units match the kernel)
        lag_centers, acf_data = self.compute_acf(tlags, normalize=False)
        lag_centers_jax = jnp.asarray(lag_centers)
        acf_data_jax = jnp.asarray(acf_data)

        # --- Parse theta0 -------------------------------------------------
        n_kernel = len(KERNEL_HPARAM_KEYS)
        kernel_keys = KERNEL_HPARAM_KEYS
        if theta0 is None:
            theta0_arr = self.theta0[:n_kernel]
        elif isinstance(theta0, dict):
            theta0_arr = self.theta0[:n_kernel].copy()
            dict_keys_in_kernel = []
            for k, v in theta0.items():
                if k in kernel_keys:
                    idx = kernel_keys.index(k)
                    theta0_arr = theta0_arr.at[idx].set(float(v))
                    dict_keys_in_kernel.append(k)
            if keys is None and dict_keys_in_kernel:
                keys = dict_keys_in_kernel
        else:
            theta0_arr = jnp.asarray(theta0, dtype=jnp.float64)

        # Resolve free vs fixed parameters (within kernel keys only)
        if keys is None:
            free_idx = list(range(n_kernel))
            fixed_idx = []
            fixed_vals = jnp.array([])
        else:
            for k in keys:
                if k not in kernel_keys:
                    raise ValueError(
                        f"Unknown key '{k}'. Valid kernel keys: {kernel_keys}")
            free_idx = [i for i, k in enumerate(kernel_keys) if k in keys]
            fixed_idx = [i for i, k in enumerate(kernel_keys) if k not in keys]
            fixed_vals = (theta0_arr[jnp.array(fixed_idx)]
                          if fixed_idx else jnp.array([]))

        free0_theta = theta0_arr[jnp.array(free_idx)]
        bounds_kernel = self.bounds[:n_kernel]
        free_bounds = bounds_kernel[jnp.array(free_idx)]
        blo = free_bounds[:, 0]
        bhi = free_bounds[:, 1]
        brange = bhi - blo

        # Optimize in normalized coordinates u = (theta - lo) / (hi - lo)
        u0 = np.asarray((free0_theta - blo) / brange, dtype=np.float64)

        qn, qw = self._quad_nodes, self._quad_weights
        n_h, n_l, lr = self.n_harmonics, self.n_lat, self.lat_range

        @jax.jit
        def loss_u(u_arr):
            free_theta = blo + u_arr * brange
            theta_full = self._theta_from_free(
                free_theta, free_idx, fixed_idx, fixed_vals)
            K_model = _kernel_eval(theta_full, lag_centers_jax,
                                   n_h, n_l, lr,
                                   quad_nodes=qn, quad_weights=qw)
            return jnp.sum((acf_data_jax - K_model) ** 2)

        vg_fn = jax.jit(jax.value_and_grad(loss_u))

        n_free = len(free_idx)
        _gradient_free = method.lower() in ("nelder-mead", "cobyla", "powell")

        if _gradient_free:
            def objective(u_np):
                u_jax = jnp.array(u_np, dtype=jnp.float64)
                val, _ = vg_fn(u_jax)
                v = float(val)
                return 1e30 if not np.isfinite(v) else v
        else:
            def objective(u_np):
                u_jax = jnp.array(u_np, dtype=jnp.float64)
                val, grad = vg_fn(u_jax)
                v = float(val)
                g = np.asarray(grad, dtype=np.float64)
                if not np.isfinite(v):
                    return 1e30, np.zeros_like(g)
                if not np.all(np.isfinite(g)):
                    return v, np.zeros_like(g)
                return v, g
        if _gradient_free:
            _minimize_kwargs = dict(
                method=method,
                options={"maxiter": maxiter, "xatol": ftol, "fatol": ftol,
                         "disp": disp},
            )
        else:
            _minimize_kwargs = dict(
                jac=True, method=method,
                bounds=[(0.0, 1.0)] * n_free,
                options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol,
                         "disp": disp},
            )
        result = minimize(objective, u0, **_minimize_kwargs)

        # Transform back to physical coordinates
        free_best = blo + jnp.array(result.x, dtype=jnp.float64) * brange
        theta_full = self._theta_from_free(
            free_best, free_idx, fixed_idx, fixed_vals)

        # Store results
        self.acf_fit_theta = theta_full
        self._acf_fit_result = result
        self._acf_lag_centers = lag_centers
        self._acf_data = acf_data

        theta_dict = {k: float(theta_full[i])
                      for i, k in enumerate(kernel_keys)}
        return theta_dict, result

    def fit_acf_parallel(self, nopt=10, ncore=None, keys=None,
                         tlags=None, n_bins=50, method="nelder-mead",
                         maxiter=500, ftol=0, gtol=1e-8, disp=False,
                         return_all=False, rng=None):
        """
        Run ``fit_acf`` from multiple random starting points in parallel.

        Starting points are drawn uniformly within the kernel parameter
        bounds.

        Parameters
        ----------
        nopt : int
            Number of independent optimization trials (default 10).
        ncore : int or None
            Number of parallel workers. If None, uses ``nopt`` or the
            number of available CPUs, whichever is smaller.
        keys : list of str, optional
            Free parameters (forwarded to ``fit_acf``).
        tlags, n_bins
            Forwarded to ``fit_acf``.
        method : str
            Optimizer method (default "nelder-mead").
        maxiter, ftol, gtol, disp
            Forwarded to ``fit_acf``.
        return_all : bool
            If True, return all solutions sorted by objective value.
            If False (default), return only the best solution.
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        theta_best : dict  (or list of dict if ``return_all=True``)
            Best-fit kernel hyperparameters.
        result_best : scipy.optimize.OptimizeResult
            (or list of OptimizeResult if ``return_all=True``)
        """
        from concurrent.futures import ThreadPoolExecutor
        import os

        if ncore is None:
            ncore = min(nopt, os.cpu_count() or 1)
        if rng is None:
            rng = np.random.default_rng()

        # Determine which kernel indices are free
        n_kernel = len(KERNEL_HPARAM_KEYS)
        kernel_keys = KERNEL_HPARAM_KEYS
        if keys is None:
            free_keys = list(kernel_keys)
            free_bounds_np = np.asarray(self.bounds[:n_kernel])
        else:
            for k in keys:
                if k not in kernel_keys:
                    raise ValueError(
                        f"Unknown key '{k}'. Valid kernel keys: {kernel_keys}")
            free_keys = [k for k in kernel_keys if k in keys]
            idx = [kernel_keys.index(k) for k in free_keys]
            free_bounds_np = np.asarray(self.bounds[jnp.array(idx)])
        blo = free_bounds_np[:, 0]
        bhi = free_bounds_np[:, 1]

        # Generate random starting points (uniform within bounds)
        starts = []
        for _ in range(nopt):
            theta0_dict = {}
            u = rng.uniform(size=len(free_keys))
            for j, k in enumerate(free_keys):
                theta0_dict[k] = float(blo[j] + u[j] * (bhi[j] - blo[j]))
            starts.append(theta0_dict)

        def _run_one(theta0_dict):
            return self.fit_acf(theta0=theta0_dict, keys=keys,
                                tlags=tlags, n_bins=n_bins,
                                method=method, maxiter=maxiter,
                                ftol=ftol, gtol=gtol, disp=disp)

        with ThreadPoolExecutor(max_workers=ncore) as pool:
            futures = [pool.submit(_run_one, s) for s in starts]
            results = [f.result() for f in futures]

        # Sort by objective value (lower is better)
        results.sort(key=lambda tr: float(tr[1].fun))

        if return_all:
            return ([r[0] for r in results], [r[1] for r in results])

        best_theta, best_result = results[0]
        # Store the best
        self.acf_fit_theta = jnp.array(
            [float(best_theta[k]) for k in kernel_keys],
            dtype=jnp.float64)
        self._acf_fit_result = best_result
        return best_theta, best_result

    def fit_acf_psd(self, theta0=None, keys=None,
                    tlags=None, n_bins=50,
                    n_freq=200, dt_kernel=None,
                    acf_weight=1.0, psd_weight=1.0,
                    method="L-BFGS-B", maxiter=500, ftol=0, gtol=1e-8,
                    disp=False):
        """
        Fit kernel parameters jointly to the empirical ACF and PSD.

        Minimizes a weighted sum of two normalized mean-squared-error terms:

            loss = acf_weight * acf_loss + psd_weight * psd_loss

        where

            acf_loss = mean((ACF_data - K_model)^2) / mean(ACF_data^2)

        is the relative MSE of the kernel against the empirical ACF
        (unnormalized autocovariance), and

            psd_loss = mean((PSD_data_norm - PSD_model_norm)^2)

        is the MSE between the Lomb-Scargle periodogram and the analytic
        kernel PSD, both normalized to unit integral so the comparison is
        independent of overall amplitude.

        The model PSD is computed via a direct cosine transform of the kernel
        evaluated on a uniform lag grid, making it fully differentiable with
        respect to the kernel parameters.

        Parameters
        ----------
        theta0 : dict or array_like, optional
            Starting point in ``self.param_keys`` space (sampling space,
            with ``log_``-prefixed keys where applicable).  Follows the same
            convention as ``find_map``: None uses ``self.theta0``, a dict
            overrides named entries and infers free keys, an array is used
            directly.
        keys : list of str, optional
            Parameters to vary during optimization (names from
            ``self.param_keys``).  Defaults to all kernel parameters
            (first 6 entries of ``self.param_keys``, i.e. excluding
            ``sigma_n`` if present).
        tlags : array_like, optional
            Bin edges for the empirical ACF. If None, ``n_bins+1`` edges
            linearly spaced from 0 to half the baseline.
        n_bins : int
            Number of ACF lag bins when ``tlags`` is None (default 50).
        n_freq : int
            Number of frequency points for the Lomb-Scargle periodogram
            (default 200).
        dt_kernel : float, optional
            Uniform lag spacing [days] for evaluating the analytic kernel
            before the direct cosine transform.  Defaults to one-fifth of
            the median data spacing.
        acf_weight : float
            Weight for the ACF loss term (default 1.0).
        psd_weight : float
            Weight for the PSD loss term (default 1.0).
        method : str
            Scipy optimizer method (default ``"L-BFGS-B"``).
        maxiter : int
            Maximum optimizer iterations (default 500).
        ftol, gtol : float
            Convergence tolerances forwarded to scipy.
        disp : bool
            Print optimizer messages if True.

        Returns
        -------
        theta_dict : dict
            Best-fit parameters in ``self.param_keys`` space.
        result : scipy.optimize.OptimizeResult
        """
        from scipy.optimize import minimize
        from scipy.signal import lombscargle

        n_kernel = 6  # always the first 6 param_keys are kernel params
        to_phys = self._to_physical
        qn, qw = self._quad_nodes, self._quad_weights
        n_h, n_l, lr = self.n_harmonics, self.n_lat, self.lat_range

        # --- Empirical ACF (precomputed, not differentiated through) --------
        if tlags is None:
            baseline = float(jnp.max(self.x) - jnp.min(self.x))
            tlags = np.linspace(0, baseline / 2, n_bins + 1)

        lag_centers, acf_data = self.compute_acf(tlags, normalize=False)
        lag_jax  = jnp.asarray(lag_centers, dtype=jnp.float64)
        acf_jax  = jnp.asarray(acf_data,   dtype=jnp.float64)
        # Normalize so the ACF loss is dimensionless (relative MSE)
        acf_rms  = jnp.sqrt(jnp.mean(acf_jax ** 2)) + 1e-30

        # --- Empirical PSD via Lomb-Scargle (precomputed) -------------------
        x     = np.asarray(self.x)
        resid = np.asarray(self.y) - self.mean_val
        dt_med    = float(np.median(np.diff(x)))
        baseline  = float(x[-1] - x[0])
        freq_min  = 1.0 / baseline
        freq_max  = 1.0 / (2.0 * dt_med)
        freqs     = np.linspace(freq_min, freq_max, n_freq)
        pgram     = lombscargle(x, resid, 2.0 * np.pi * freqs, normalize=False)
        df        = freqs[1] - freqs[0]
        psd_data_norm = pgram / (np.sum(pgram) * df)   # normalize to unit integral
        psd_jax   = jnp.asarray(psd_data_norm, dtype=jnp.float64)
        freqs_jax = jnp.asarray(freqs,         dtype=jnp.float64)

        # --- Lag grid for direct cosine transform ---------------------------
        if dt_kernel is None:
            dt_kernel = dt_med / 5.0
        tau_grid = np.arange(0.0, baseline, dt_kernel)
        tau_jax  = jnp.asarray(tau_grid, dtype=jnp.float64)
        # Precompute cosine matrix: shape (n_tau, n_freq)
        # S(f) = dt * [K(0) + 2 * sum_{k>0} K(tau_k) cos(2pi f tau_k)]
        cos_mat  = jnp.cos(2.0 * jnp.pi * tau_jax[:, None] * freqs_jax[None, :])

        # --- Parse theta0 ---------------------------------------------------
        if theta0 is None:
            theta0_arr = self.theta0[:n_kernel].copy()
        elif isinstance(theta0, dict):
            theta0_arr = self.theta0[:n_kernel].copy()
            dict_keys_in_params = []
            for k, v in theta0.items():
                if k in self.param_keys[:n_kernel]:
                    idx = list(self.param_keys).index(k)
                    theta0_arr = theta0_arr.at[idx].set(float(v))
                    dict_keys_in_params.append(k)
            if keys is None and dict_keys_in_params:
                keys = dict_keys_in_params
        else:
            theta0_arr = jnp.asarray(theta0, dtype=jnp.float64)

        # --- Resolve free vs fixed parameters -------------------------------
        kernel_param_keys = list(self.param_keys[:n_kernel])
        if keys is None:
            free_idx  = list(range(n_kernel))
            fixed_idx = []
            fixed_vals = jnp.array([])
        else:
            for k in keys:
                if k not in kernel_param_keys:
                    raise ValueError(
                        f"Unknown key '{k}'. Valid keys: {kernel_param_keys}")
            free_idx  = [i for i, k in enumerate(kernel_param_keys) if k in keys]
            fixed_idx = [i for i, k in enumerate(kernel_param_keys) if k not in keys]
            fixed_vals = (theta0_arr[jnp.array(fixed_idx)]
                          if fixed_idx else jnp.array([]))

        free0     = theta0_arr[jnp.array(free_idx)]
        bds       = self.bounds[:n_kernel]
        free_bds  = bds[jnp.array(free_idx)]
        blo, bhi  = free_bds[:, 0], free_bds[:, 1]
        brange    = bhi - blo
        u0        = np.asarray((free0 - blo) / brange, dtype=np.float64)

        w_acf = float(acf_weight)
        w_psd = float(psd_weight)

        @jax.jit
        def loss_u(u_arr):
            free_theta  = blo + u_arr * brange
            theta_samp  = self._theta_from_free(
                free_theta, free_idx, fixed_idx, fixed_vals)
            theta_phys  = to_phys(theta_samp)

            # ACF component
            K_acf  = _kernel_eval(theta_phys, lag_jax, n_h, n_l, lr,
                                  quad_nodes=qn, quad_weights=qw)
            acf_loss = jnp.mean(((acf_jax - K_acf) / acf_rms) ** 2)

            # PSD component: direct cosine transform at LS frequencies
            K_tau = _kernel_eval(theta_phys, tau_jax, n_h, n_l, lr,
                                 quad_nodes=qn, quad_weights=qw)
            psd_model = jnp.maximum(
                dt_kernel * (K_tau[0] + 2.0 * jnp.dot(K_tau[1:], cos_mat[1:])),
                0.0)
            psd_norm  = jnp.sum(psd_model) * (freqs_jax[1] - freqs_jax[0])
            psd_model_norm = psd_model / (psd_norm + 1e-30)
            psd_loss = jnp.mean((psd_jax - psd_model_norm) ** 2)

            return w_acf * acf_loss + w_psd * psd_loss

        vg_fn = jax.jit(jax.value_and_grad(loss_u))

        n_free = len(free_idx)
        _gradient_free = method.lower() in ("nelder-mead", "cobyla", "powell")

        if _gradient_free:
            def objective(u_np):
                val, _ = vg_fn(jnp.array(u_np, dtype=jnp.float64))
                v = float(val)
                return 1e30 if not np.isfinite(v) else v
        else:
            def objective(u_np):
                val, grad = vg_fn(jnp.array(u_np, dtype=jnp.float64))
                v = float(val)
                g = np.asarray(grad, dtype=np.float64)
                if not np.isfinite(v):
                    return 1e30, np.zeros_like(g)
                if not np.all(np.isfinite(g)):
                    return v, np.zeros_like(g)
                return v, g

        if _gradient_free:
            _minimize_kwargs = dict(
                method=method,
                options={"maxiter": maxiter, "xatol": ftol, "fatol": ftol,
                         "disp": disp},
            )
        else:
            _minimize_kwargs = dict(
                jac=True, method=method,
                bounds=[(0.0, 1.0)] * n_free,
                options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol,
                         "disp": disp},
            )
        result = minimize(objective, u0, **_minimize_kwargs)

        free_best  = blo + jnp.array(result.x, dtype=jnp.float64) * brange
        theta_full = self._theta_from_free(
            free_best, free_idx, fixed_idx, fixed_vals)

        theta_dict = {k: float(theta_full[i])
                      for i, k in enumerate(kernel_param_keys)}
        return theta_dict, result

    def _theta_dict_to_phys_array(self, theta):
        """Convert a theta dict or array to a physical KERNEL_HPARAM_KEYS array.

        Handles sampling-space dicts that may contain ``log_``-prefixed keys
        (e.g. ``log_sigma_k``), converting them to physical values via
        ``10 ** value`` before building the array.  Plain arrays are returned
        as-is (assumed already physical).

        Parameters
        ----------
        theta : dict or array_like
            Kernel parameters, either as a dict (physical or log-space keys)
            or a length-6 array in KERNEL_HPARAM_KEYS order.

        Returns
        -------
        theta_arr : jnp.ndarray, shape (6,)
            Physical kernel parameters in KERNEL_HPARAM_KEYS order.
        """
        if isinstance(theta, dict):
            phys = {}
            for k, v in theta.items():
                if k.startswith("log_"):
                    phys[k[4:]] = 10.0 ** float(v)
                else:
                    phys[k] = float(v)
            return jnp.array([float(phys[k]) for k in KERNEL_HPARAM_KEYS],
                             dtype=jnp.float64)
        return jnp.asarray(theta, dtype=jnp.float64)

    def plot_acf(self, theta=None, tlags=None, n_bins=50, ax=None,
                 normalize=True, data_color="k", model_color="r",
                 show_legend=True):
        """
        Plot the empirical ACF and optionally the analytic kernel.

        Parameters
        ----------
        theta : dict or array_like, shape (6,), optional
            Kernel parameters.  Accepts a physical dict with keys from
            ``KERNEL_HPARAM_KEYS``, a sampling-space dict with ``log_``-
            prefixed keys (e.g. ``log_sigma_k``), or a length-6 array.
            If provided, the analytic kernel is overplotted.
        tlags : array_like, optional
            Bin edges for compute_acf. If None, linearly spaced from 0 to
            half the baseline with n_bins+1 edges.
        n_bins : int
            Number of lag bins (used when tlags is None).
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates a new figure.
        normalize : bool
            If True (default), normalize both curves by the data variance
            so ACF(0) ≈ 1.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if tlags is None:
            baseline = float(jnp.max(self.x) - jnp.min(self.x))
            tlags = np.linspace(0, baseline / 2, n_bins + 1)

        lag_centers, acf_data = self.compute_acf(tlags, normalize=normalize)

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(lag_centers, acf_data, color=data_color, label="Data ACF")

        if theta is not None:
            theta_arr = self._theta_dict_to_phys_array(theta)
            lag_fine = np.linspace(0.0, float(tlags[-1]), 300)
            K_model = np.asarray(_kernel_eval(
                theta_arr, jnp.asarray(lag_fine),
                self.n_harmonics, self.n_lat, self.lat_range,
                quad_nodes=self._quad_nodes, quad_weights=self._quad_weights))
            if normalize:
                var = np.mean((np.asarray(self.y) - self.mean_val) ** 2)
                if var > 0:
                    K_model = K_model / var
            ax.plot(lag_fine, K_model, color=model_color, label="Analytic kernel")

        ax.set_xlabel("Time lag [days]", fontsize=22)
        ax.set_ylabel("ACF" if normalize else "Autocovariance", fontsize=22)
        if show_legend:
            ax.legend()

        return ax

    def plot_psd(self, theta=None, n_freq=500, dt_kernel=None, ax=None,
                 data_color="k", model_color="r", show_legend=True):
        """
        Plot the empirical PSD (Lomb-Scargle) and optionally the analytic
        kernel PSD (FFT of the autocovariance function).

        Both curves are normalized so their integral over positive frequencies
        equals the data variance, making them directly comparable.

        Parameters
        ----------
        theta : dict or array_like, shape (6,), optional
            Kernel parameters.  Accepts a physical dict with keys from
            ``KERNEL_HPARAM_KEYS``, a sampling-space dict with ``log_``-
            prefixed keys (e.g. ``log_sigma_k``), or a length-6 array.
            If provided, the analytic kernel PSD is overplotted.
        n_freq : int
            Number of frequency points for the Lomb-Scargle periodogram.
        dt_kernel : float, optional
            Time step [days] for evaluating the analytic kernel on a uniform
            grid before FFT.  Defaults to one-fifth of the median data spacing.
        ax : matplotlib Axes, optional
            Axes to plot on. If None, creates a new figure.
        data_color, model_color : str
            Colors for the data and model curves.
        show_legend : bool
            Whether to draw a legend.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt
        from scipy.signal import lombscargle

        x = np.asarray(self.x)
        resid = np.asarray(self.y) - self.mean_val
        var = float(np.mean(resid ** 2))

        baseline = float(x[-1] - x[0])
        dt_med = float(np.median(np.diff(x)))
        freq_min = 1.0 / baseline
        freq_max = 1.0 / (2.0 * dt_med)
        freqs = np.linspace(freq_min, freq_max, n_freq)

        # Lomb-Scargle uses angular frequencies internally
        pgram = lombscargle(x, resid, 2.0 * np.pi * freqs, normalize=False)
        # Normalize so ∫PSD df = var(data)
        psd_data = pgram * var / np.trapezoid(pgram, freqs)

        if ax is None:
            fig, ax = plt.subplots()

        ax.semilogy(freqs, psd_data, color=data_color, lw=0.8,
                    label="Lomb-Scargle")

        if theta is not None:
            theta_arr = self._theta_dict_to_phys_array(theta)

            if dt_kernel is None:
                dt_kernel = dt_med / 5.0
            tau_grid = np.arange(0.0, baseline, dt_kernel)
            K = np.asarray(_kernel_eval(
                theta_arr, jnp.asarray(tau_grid),
                self.n_harmonics, self.n_lat, self.lat_range,
                quad_nodes=self._quad_nodes, quad_weights=self._quad_weights))
            # Extend to two-sided symmetric sequence, then rfft → one-sided PSD
            K_twosided = np.concatenate([K[::-1], K[1:]])
            psd_model = np.abs(np.fft.rfft(K_twosided)) * dt_kernel
            freqs_model = np.fft.rfftfreq(len(K_twosided), d=dt_kernel)
            # Restrict to the data frequency range and skip DC
            mask = (freqs_model > 0) & (freqs_model <= freq_max)
            fm, pm = freqs_model[mask], psd_model[mask]
            # Normalize so ∫PSD df = var(data)
            pm = pm * var / np.trapezoid(pm, fm)
            ax.semilogy(fm, pm, color=model_color, lw=1.5,
                        label="Analytic kernel")

        ax.set_xlabel("Frequency [1/day]", fontsize=22)
        ax.set_ylabel("PSD", fontsize=22)
        if show_legend:
            ax.legend()

        return ax

    def get_theta(self):
        """
        Return the current kernel hyperparameters as a dictionary.

        Returns
        -------
        theta : dict
            Keys and values for all kernel (and optionally noise)
            hyperparameters, e.g. {"peq": 5.0, "kappa": 0.2, ...}.
        """
        return {k: float(self.theta0[i]) for i, k in enumerate(self.param_keys)}

    def _resolve_keys(self, keys):
        """
        Validate a user-supplied list of free-parameter keys and return
        the integer indices into self.param_keys plus the complementary
        fixed indices and fixed values.

        Returns
        -------
        free_idx : list of int
        fixed_idx : list of int
        fixed_vals : jnp.ndarray
        """
        if keys is None:
            return list(range(len(self.param_keys))), [], jnp.array([])
        for k in keys:
            if k not in self.param_keys:
                raise ValueError(
                    f"Unknown key '{k}'. Valid keys: {self.param_keys}")
        free_idx = [i for i, k in enumerate(self.param_keys) if k in keys]
        fixed_idx = [i for i, k in enumerate(self.param_keys) if k not in keys]
        fixed_vals = self.theta0[jnp.array(fixed_idx)] if fixed_idx else jnp.array([])
        return free_idx, fixed_idx, fixed_vals

    def _theta_from_free(self, free_vals, free_idx, fixed_idx, fixed_vals):
        """Reconstruct the full theta vector from free and fixed parts."""
        n = len(free_idx) + len(fixed_idx)
        theta = jnp.zeros(n)
        theta = theta.at[jnp.array(free_idx)].set(free_vals)
        if len(fixed_idx) > 0:
            theta = theta.at[jnp.array(fixed_idx)].set(fixed_vals)
        return theta

    def _result_dict(self, theta_arr):
        """Convert a full theta array to a labeled dictionary."""
        return {k: float(theta_arr[i]) for i, k in enumerate(self.param_keys)}

    def update_hparam(self, hparam):
        """Update hyperparameters and rebuild kernel and covariance."""
        _validate_hparam(hparam)
        self.hparam = dict(hparam)
        self._build_kernel()
        self.hparam = dict(self.kernel.hparam)
        if self.matrix_solver == "cholesky_banded":
            new_bw = self._compute_bandwidth()
            if new_bw != self.bandwidth:
                self.bandwidth = new_bw
                self._build_covariance()
                self._build_logposterior()
            else:
                self._build_covariance()
        else:
            self._build_covariance()
        self.theta0 = jnp.array([
            np.log10(float(self.hparam.get(self._log_param_map[k], 0.0)))
            if k in self._log_param_map
            else float(self.hparam.get(k, 0.0))
            for k in self.param_keys
        ], dtype=jnp.float64)

    # =================================================================
    # MAP estimation
    # =================================================================

    def find_map(self, theta0=None, keys=None, method="L-BFGS-B",
                 maxiter=500, ftol=0, gtol=1e-8, disp=False):
        """
        Find the maximum a posteriori (MAP) estimate.

        Uses scipy.optimize.minimize with JAX-computed gradients.

        Parameters
        ----------
        theta0 : dict or array_like, optional
            Starting point. Can be:
              - None: uses self.theta0 (current hyperparameters).
              - dict: values for any subset of param_keys set the
                starting point. If ``keys`` is not given, the dict
                keys that overlap with ``self.param_keys`` are treated
                as the free variables to optimize; the rest are held
                fixed. Extra keys not in ``param_keys`` are ignored.
              - array_like: full theta vector (length ``n_params``).
        keys : list of str, optional
            Which parameters to vary during optimization. Overrides
            the automatic inference from a dict ``theta0``. Parameters
            not listed are held fixed at their current values. If None
            and theta0 is not a dict, all parameters are varied.
        method : str
            Scipy optimizer method (default "L-BFGS-B").
        maxiter : int
            Maximum iterations.
        ftol : float
            Function-value convergence tolerance for L-BFGS-B
            (default 0, i.e. disabled so that convergence is
            controlled by ``gtol``).
        gtol : float
            Gradient-norm convergence tolerance (default 1e-8).
        disp : bool
            If True, print optimizer convergence messages (default False).

        Returns
        -------
        theta_dict : dict
            Full dictionary of all hyperparameters (fixed + optimized).
        result : scipy OptimizeResult
            Full optimizer output.
        """
        from scipy.optimize import minimize

        # --- Parse theta0 -------------------------------------------------
        if theta0 is None:
            theta0_arr = self.theta0.copy()
        elif isinstance(theta0, dict):
            # Build full array from current values, overriding with dict
            theta0_arr = self.theta0.copy()
            dict_keys_in_params = []
            for k, v in theta0.items():
                if k in self.param_keys:
                    idx = self.param_keys.index(k)
                    theta0_arr = theta0_arr.at[idx].set(float(v))
                    dict_keys_in_params.append(k)
            # Infer free keys from dict when keys is not explicitly given
            if keys is None and dict_keys_in_params:
                keys = dict_keys_in_params
        else:
            theta0_arr = jnp.asarray(theta0, dtype=jnp.float64)

        free_idx, fixed_idx, fixed_vals = self._resolve_keys(keys)

        free0_theta = theta0_arr[jnp.array(free_idx)]
        free_bounds = self.bounds[jnp.array(free_idx)]
        blo = free_bounds[:, 0]
        bhi = free_bounds[:, 1]
        brange = bhi - blo

        # Optimize in normalized coordinates u = (theta - lo) / (hi - lo)
        # so all free parameters live in [0, 1] with comparable scale.
        u0 = np.asarray((free0_theta - blo) / brange, dtype=np.float64)

        # Build objective directly from raw functions (no nested JIT).
        x, y, yerr = self.x, self.y, self.yerr
        mean_val = self.mean_val
        n_h, n_l, lr = self.n_harmonics, self.n_lat, self.lat_range
        fit_sn = self.fit_sigma_n
        qn, qw = self._quad_nodes, self._quad_weights
        all_bounds = self.bounds
        custom_prior = self._custom_log_prior
        to_phys = self._to_physical

        @jax.jit
        def neg_logpost_u(u_arr):
            free_theta = blo + u_arr * brange
            theta_full = self._theta_from_free(
                free_theta, free_idx, fixed_idx, fixed_vals)
            ll = _gp_log_likelihood(to_phys(theta_full), x, y, yerr, mean_val,
                                    n_h, n_l, lr, fit_sn,
                                    quad_nodes=qn, quad_weights=qw)
            if custom_prior is not None:
                lp = custom_prior(theta_full)
            else:
                lp = _default_log_prior(theta_full, all_bounds)
            return -(ll + lp)

        vg_fn = jax.jit(jax.value_and_grad(neg_logpost_u))

        n_free = len(free_idx)
        _gradient_free = method.lower() in ("nelder-mead", "cobyla", "powell")

        if _gradient_free:
            def objective(u_np):
                u_jax = jnp.array(u_np, dtype=jnp.float64)
                val, _ = vg_fn(u_jax)
                v = float(val)
                return 1e30 if not np.isfinite(v) else v
        else:
            def objective(u_np):
                u_jax = jnp.array(u_np, dtype=jnp.float64)
                val, grad = vg_fn(u_jax)
                v = float(val)
                g = np.asarray(grad, dtype=np.float64)
                if not np.isfinite(v):
                    return 1e30, np.zeros_like(g)
                if not np.all(np.isfinite(g)):
                    return v, np.zeros_like(g)
                return v, g
        if _gradient_free:
            _minimize_kwargs = dict(
                method=method,
                options={"maxiter": maxiter, "xatol": ftol, "fatol": ftol,
                         "disp": disp},
            )
        else:
            _minimize_kwargs = dict(
                jac=True, method=method,
                bounds=[(0.0, 1.0)] * n_free,
                options={"maxiter": maxiter, "ftol": ftol, "gtol": gtol,
                         "disp": disp},
            )
        result = minimize(objective, u0, **_minimize_kwargs)

        # Transform back to physical coordinates
        free_best = blo + jnp.array(result.x, dtype=jnp.float64) * brange
        theta_full = self._theta_from_free(
            free_best, free_idx, fixed_idx, fixed_vals)

        self.map_estimate = theta_full
        self._map_result = result
        return self._result_dict(theta_full), result

    def find_map_parallel(self, nopt=10, ncore=None, keys=None,
                          method="nelder-mead", maxiter=500, ftol=0,
                          gtol=1e-8, disp=False, return_all=False,
                          rng=None):
        """
        Run ``find_map`` from multiple random starting points in parallel.

        Starting points are drawn uniformly within the parameter bounds.

        Parameters
        ----------
        nopt : int
            Number of independent optimization trials (default 10).
        ncore : int or None
            Number of parallel workers. If None, uses ``nopt`` or the
            number of available CPUs, whichever is smaller.
        keys : list of str, optional
            Free parameters (forwarded to ``find_map``).
        method : str
            Optimizer method (default "nelder-mead").
        maxiter, ftol, gtol, disp
            Forwarded to ``find_map``.
        return_all : bool
            If True, return all solutions sorted by objective value.
            If False (default), return only the best solution.
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        theta_best : dict  (or list of dict if ``return_all=True``)
            Best-fit hyperparameters.
        result_best : scipy.optimize.OptimizeResult
            (or list of OptimizeResult if ``return_all=True``)
        """
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        import os

        if ncore is None:
            ncore = min(nopt, os.cpu_count() or 1)
        if rng is None:
            rng = np.random.default_rng()

        # Determine which indices are free
        free_idx, fixed_idx, _ = self._resolve_keys(keys)
        free_keys = [self.param_keys[i] for i in free_idx]
        free_bounds = np.asarray(self.bounds[jnp.array(free_idx)])
        blo = free_bounds[:, 0]
        bhi = free_bounds[:, 1]

        # Generate random starting points (uniform within bounds)
        starts = []
        for _ in range(nopt):
            theta0_dict = {}
            u = rng.uniform(size=len(free_keys))
            for j, k in enumerate(free_keys):
                theta0_dict[k] = float(blo[j] + u[j] * (bhi[j] - blo[j]))
            starts.append(theta0_dict)

        # Worker function (must be top-level-picklable for ProcessPool,
        # so we use ThreadPool which shares memory with JAX)
        def _run_one(theta0_dict):
            return self.find_map(theta0=theta0_dict, keys=keys,
                                 method=method, maxiter=maxiter,
                                 ftol=ftol, gtol=gtol, disp=disp)

        # Run in parallel using threads (JAX releases the GIL during
        # compiled computation, so threads give real parallelism here
        # without pickling issues)
        with ThreadPoolExecutor(max_workers=ncore) as pool:
            futures = [pool.submit(_run_one, s) for s in starts]
            results = [f.result() for f in futures]

        # Sort by objective value (lower is better for neg_log_posterior)
        results.sort(key=lambda tr: float(tr[1].fun))

        if return_all:
            return ([r[0] for r in results], [r[1] for r in results])

        best_theta, best_result = results[0]
        # Store the best as the MAP estimate
        self.map_estimate = jnp.array(
            [float(best_theta[k]) for k in self.param_keys],
            dtype=jnp.float64)
        self._map_result = best_result
        return best_theta, best_result

    # =================================================================
    # Mass matrix estimation: Method 1 -- Hessian at MAP
    # =================================================================

    def mass_matrix_hessian_map(self, theta_map=None):
        """
        Estimate the inverse mass matrix from the Hessian of the
        negative log-likelihood at the MAP.

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

        x, y, yerr = self.x, self.y, self.yerr
        mean_val = self.mean_val
        n_h, n_l, lr = self.n_harmonics, self.n_lat, self.lat_range
        fit_sn = self.fit_sigma_n
        qn, qw = self._quad_nodes, self._quad_weights
        to_phys = self._to_physical

        @jax.jit
        def neg_log_lik(theta_arr):
            return -_gp_log_likelihood(to_phys(theta_arr), x, y, yerr, mean_val,
                                       n_h, n_l, lr, fit_sn,
                                       quad_nodes=qn, quad_weights=qw)

        H = jax.hessian(neg_log_lik)(theta_map)

        # Regularize: ensure positive-definite
        eigvals, eigvecs = jnp.linalg.eigh(H)
        eigvals = jnp.maximum(eigvals, 1e-6)
        H_reg = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

        self.inverse_mass_matrix = jnp.linalg.inv(H_reg)
        self._hessian = H
        return self.inverse_mass_matrix

    # =================================================================
    # Mass matrix estimation: Method 2 -- Fisher information
    # =================================================================

    def mass_matrix_fisher(self, theta_map=None):
        """
        Estimate the inverse mass matrix from the Fisher information.

        For the GP log-likelihood:

            I_{ij} = (1/2) tr(K^{-1} dK/dtheta_i  K^{-1} dK/dtheta_j)

        The kernel derivatives dK/dtheta_i are computed via JAX
        forward-mode autodiff (jacfwd).

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

        N = self.N
        n_params = theta_map.shape[0]
        if self._lag_flat is None:
            self._lag_flat = jnp.abs(
                self.x[:, None] - self.x[None, :]).ravel()
        lag_flat = self._lag_flat
        fit_sn = self.fit_sigma_n
        qn, qw = self._quad_nodes, self._quad_weights

        to_phys = self._to_physical

        def K_noise_flat_from_theta(theta_arr):
            """Return the full K_noise matrix as a flat vector."""
            theta_arr = to_phys(theta_arr)
            n_kernel = 6
            if fit_sn:
                theta_kernel = theta_arr[:n_kernel]
                sigma_n = theta_arr[n_kernel]
            else:
                theta_kernel = theta_arr
                sigma_n = 0.0

            K_flat = _kernel_eval(theta_kernel, lag_flat,
                                  self.n_harmonics, self.n_lat,
                                  self.lat_range,
                                  quad_nodes=qn, quad_weights=qw)
            K = K_flat.reshape(N, N)
            noise_var = self.yerr ** 2 + sigma_n ** 2
            K_noise = K + jnp.diag(noise_var) + 1e-8 * jnp.eye(N)
            return K_noise.ravel()

        dK_flat_dtheta = jax.jacfwd(K_noise_flat_from_theta)(theta_map)
        dK_dtheta = dK_flat_dtheta.reshape(N, N, n_params)

        K_noise_flat = K_noise_flat_from_theta(theta_map)
        K = K_noise_flat.reshape(N, N)
        K_inv = jnp.linalg.inv(K)

        K_inv_dK = jnp.einsum('ab,bcj->acj', K_inv, dK_dtheta)
        fisher = 0.5 * jnp.einsum('abi,baj->ij', K_inv_dK, K_inv_dK)

        # Regularize
        eigvals, eigvecs = jnp.linalg.eigh(fisher)
        eigvals = jnp.maximum(eigvals, 1e-6)
        fisher_reg = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

        self.inverse_mass_matrix = jnp.linalg.inv(fisher_reg)
        self._fisher_matrix = fisher
        return self.inverse_mass_matrix

    # =================================================================
    # Mass matrix estimation: Method 3 -- Laplace approximation
    # =================================================================

    def mass_matrix_laplace(self, theta_map=None):
        """
        Laplace approximation: inverse mass matrix = inverse Hessian
        of the negative log-likelihood at the MAP.

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

        x, y, yerr = self.x, self.y, self.yerr
        mean_val = self.mean_val
        n_h, n_l, lr = self.n_harmonics, self.n_lat, self.lat_range
        fit_sn = self.fit_sigma_n
        qn, qw = self._quad_nodes, self._quad_weights
        to_phys = self._to_physical

        @jax.jit
        def neg_log_lik(theta_arr):
            return -_gp_log_likelihood(to_phys(theta_arr), x, y, yerr, mean_val,
                                       n_h, n_l, lr, fit_sn,
                                       quad_nodes=qn, quad_weights=qw)

        H = jax.hessian(neg_log_lik)(theta_map)

        # Regularize
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

        return jax.random.multivariate_normal(
            rng_key, mean, cov, shape=(n_samples,))

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
