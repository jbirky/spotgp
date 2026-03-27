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
    from .params import (
        resolve_hparam, KERNEL_HPARAM_KEYS, HPARAM_KEYS_WITH_NOISE,
    )
    from .envelope import _R_Gamma_symmetric
    from .spot_model import (
        SpotEvolutionModel, VisibilityFunction, EdgeOnVisibilityFunction,
        _cn_general_jax, _gauss_legendre_grid,
    )
    from .analytic_kernel import AnalyticKernel
    from .banded_cholesky import (banded_cholesky, banded_solve,
                                   banded_cholesky_compact, banded_solve_compact)
except ImportError:
    from params import (
        resolve_hparam, KERNEL_HPARAM_KEYS, HPARAM_KEYS_WITH_NOISE,
    )
    from envelope import _R_Gamma_symmetric
    from spot_model import (
        SpotEvolutionModel, VisibilityFunction, EdgeOnVisibilityFunction,
        _cn_general_jax, _gauss_legendre_grid,
    )
    from analytic_kernel import AnalyticKernel
    from banded_cholesky import (banded_cholesky, banded_solve,
                                 banded_cholesky_compact, banded_solve_compact)

__all__ = ["GPSolver"]

# KERNEL_HPARAM_KEYS and HPARAM_KEYS_WITH_NOISE are imported from params.
# Re-export as lists for any callers that expect list type.
KERNEL_HPARAM_KEYS = list(KERNEL_HPARAM_KEYS)
HPARAM_KEYS_WITH_NOISE = list(HPARAM_KEYS_WITH_NOISE)


# =====================================================================
# Pure-functional GP helpers (module-level for clean JAX tracing)
# =====================================================================

def _kernel_eval_edgeon(theta_arr, lag_flat, n_harmonics,
                        r_gamma_func=None, cn_sq_fixed=None):
    """
    Fast-path kernel evaluation for EdgeOnVisibilityFunction.

    Skips the latitude quadrature loop entirely.  The latitude-averaged
    |c_n|^2 are known constants passed in via ``cn_sq_fixed``.

    Parameters
    ----------
    theta_arr : jnp.ndarray
        Kernel parameters.  theta_arr[0] = peq, theta_arr[-1] = sigma_k.
    lag_flat : jnp.ndarray, shape (M,)
    n_harmonics : int
    r_gamma_func : callable or None
    cn_sq_fixed : jnp.ndarray, shape (n_harmonics+1,)
        Pre-computed latitude-averaged |c_n|^2.

    Returns
    -------
    K_flat : jnp.ndarray, shape (M,)
    """
    peq = theta_arr[0]
    sigma_k = theta_arr[-1]

    if r_gamma_func is not None:
        R = r_gamma_func(theta_arr, lag_flat)
    else:
        lspot    = theta_arr[1]
        tau_spot = theta_arr[2]
        R = _R_Gamma_symmetric(lag_flat, lspot, tau_spot)

    w0 = 2.0 * jnp.pi / peq
    harm_ns = jnp.arange(1, n_harmonics + 1)
    cosine_terms = jnp.sum(
        cn_sq_fixed[1:] * jnp.cos(harm_ns * w0 * lag_flat[:, None]), axis=1)
    K = sigma_k ** 2 * R * (cn_sq_fixed[0] + 2.0 * cosine_terms)
    return K


def _kernel_eval(theta_arr, lag_flat, n_harmonics, n_lat, lat_range,
                  quad_nodes=None, quad_weights=None,
                  r_gamma_func=None,
                  edgeon_cn_sq=None,
                  lat_weight_func=None):
    """
    Pure-functional kernel evaluation: theta_arr -> kernel values.

    Uses ``jax.vmap`` over the latitude grid so that all latitudes are
    processed in parallel.  cn_sq coefficients for all latitudes are
    precomputed with a double-vmap (outer: latitudes, inner: harmonics)
    before entering the per-latitude cosine sum, avoiding repeated vmap
    traces inside a sequential loop.

    Memory scales as O(n_lat * M) instead of O(M), which is acceptable
    on GPU where M = N*bandwidth is small relative to device memory.

    When ``edgeon_cn_sq`` is not None, delegates to the fast edge-on
    path that skips all latitude quadrature.

    Parameters
    ----------
    theta_arr : jnp.ndarray, shape (n_params,)
        Kernel parameters. First three are always [peq, kappa, inc];
        last is always sigma_k. Layout otherwise depends on envelope type.
        Default (backward-compat) layout: [peq, kappa, inc, lspot, tau_spot, sigma_k].
    lag_flat : jnp.ndarray, shape (M,)
        Flattened time lags.
    n_harmonics, n_lat, lat_range : kernel config (static).
    quad_nodes, quad_weights : jnp.ndarray or None
        Gauss-Legendre nodes and weights. If None, uses trapezoid rule.
    r_gamma_func : callable or None
        JAX-traceable function ``f(theta_arr, lag) -> R_Gamma``.
        If None, uses the symmetric-trapezoid closed-form (backward compat).
    edgeon_cn_sq : jnp.ndarray or None
        If not None, use the edge-on fast path with these fixed |c_n|^2.
    lat_weight_func : callable or None
        JAX-traceable function ``f(theta_arr, phi_grid) -> weights`` that
        computes per-node latitude weights from the theta vector. If None,
        all quadrature nodes are weighted equally (or by the static
        quad_weights).

    Returns
    -------
    K_flat : jnp.ndarray, shape (M,)
        Kernel values at each lag.
    """
    # Fast path for EdgeOnVisibilityFunction
    if edgeon_cn_sq is not None:
        return _kernel_eval_edgeon(theta_arr, lag_flat, n_harmonics,
                                   r_gamma_func=r_gamma_func,
                                   cn_sq_fixed=edgeon_cn_sq)

    peq    = theta_arr[0]
    kappa  = theta_arr[1]
    inc    = theta_arr[2]
    sigma_k = theta_arr[-1]

    # R_Gamma: envelope-specific or default symmetric trapezoid
    if r_gamma_func is not None:
        R = r_gamma_func(theta_arr, lag_flat)
    else:
        lspot    = theta_arr[3]
        tau_spot = theta_arr[4]
        R = _R_Gamma_symmetric(lag_flat, lspot, tau_spot)

    # Quadrature grid
    if quad_nodes is not None:
        phi_grid = quad_nodes
        weights = quad_weights
    else:
        phi_min, phi_max = lat_range
        phi_grid = jnp.linspace(phi_min, phi_max, n_lat)
        dphi = phi_grid[1] - phi_grid[0]
        weights = jnp.ones_like(phi_grid) * dphi

    # Apply dynamic latitude weights from theta (e.g. band boundaries)
    if lat_weight_func is not None:
        lat_w = lat_weight_func(theta_arr, phi_grid)
        weights = weights * lat_w

    norm = jnp.sum(weights)

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
                       fit_sigma_n, n_kernel=6,
                       r_gamma_func=None,
                       quad_nodes=None, quad_weights=None,
                       edgeon_cn_sq=None,
                       lat_weight_func=None):
    """
    Pure-functional GP marginal log-likelihood.

    Exploits the symmetry of the covariance matrix by evaluating the
    kernel only on the upper-triangular lags (N*(N+1)/2 instead of N^2),
    halving memory and compute for the kernel evaluation step.

    Parameters
    ----------
    theta_full : jnp.ndarray, shape (n_kernel,) or (n_kernel+1,)
        Kernel params, optionally followed by sigma_n (white noise).
    x, y, yerr : jnp.ndarray
        Observations.
    mean_val : float
        Constant mean.
    n_harmonics, n_lat, lat_range : kernel config.
    fit_sigma_n : bool
        If True, last element of theta_full is sigma_n.
    n_kernel : int
        Number of kernel parameters (default 6 for backward compat).
    r_gamma_func : callable or None
        JAX-traceable envelope R_Gamma function (see _kernel_eval).
    quad_nodes, quad_weights : jnp.ndarray or None
        Gauss-Legendre nodes/weights. If None, uses trapezoid rule.

    Returns
    -------
    logL : scalar
    """
    N = x.shape[0]

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
                           quad_nodes=quad_nodes, quad_weights=quad_weights,
                           r_gamma_func=r_gamma_func,
                           edgeon_cn_sq=edgeon_cn_sq,
                           lat_weight_func=lat_weight_func)

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
                              r_gamma_func=None,
                              quad_nodes=None, quad_weights=None,
                              edgeon_cn_sq=None,
                              lat_weight_func=None):
    """
    Build the kernel covariance directly in compact (b+1, N) banded storage.

    Evaluates only the O(N*b) within-band lags instead of the full N^2 matrix.

    Parameters
    ----------
    theta_kernel : jnp.ndarray, shape (n_params,)
        Kernel hyperparameters.
    x : jnp.ndarray, shape (N,)
        Observation times.
    bandwidth : int
        Number of sub-diagonals.
    n_harmonics, n_lat, lat_range : kernel config.
    r_gamma_func : callable or None
        JAX-traceable envelope R_Gamma function (see _kernel_eval).
    quad_nodes, quad_weights : jnp.ndarray or None

    Returns
    -------
    cb : jnp.ndarray, shape (b+1, N)
        Compact lower-banded storage: ``cb[d, i] = K(x[i+d], x[i])``.
    """
    N = x.shape[0]
    b = bandwidth

    d_idx = jnp.arange(b + 1)[:, None]
    i_idx = jnp.arange(N)[None, :]
    j_idx = i_idx + d_idx
    j_safe = jnp.minimum(j_idx, N - 1)
    valid = j_idx < N

    lags_flat = jnp.abs(x[j_safe] - x[i_idx]).ravel()
    K_flat = _kernel_eval(theta_kernel, lags_flat,
                          n_harmonics, n_lat, lat_range,
                          quad_nodes=quad_nodes, quad_weights=quad_weights,
                          r_gamma_func=r_gamma_func,
                          edgeon_cn_sq=edgeon_cn_sq,
                          lat_weight_func=lat_weight_func)
    cb = K_flat.reshape(b + 1, N)
    cb = jnp.where(valid, cb, 0.0)
    return cb


def _gp_log_likelihood_banded(theta_full, x, y, yerr, mean_val,
                               n_harmonics, n_lat, lat_range,
                               fit_sigma_n, bandwidth,
                               n_kernel=6,
                               r_gamma_func=None,
                               quad_nodes=None, quad_weights=None,
                               edgeon_cn_sq=None,
                               lat_weight_func=None):
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
    n_kernel : int
        Number of kernel parameters (default 6 for backward compat).
    r_gamma_func : callable or None
        JAX-traceable envelope R_Gamma function (see _kernel_eval).
    All other parameters are the same as ``_gp_log_likelihood``.
    """
    N = x.shape[0]

    if fit_sigma_n:
        theta_kernel = theta_full[:n_kernel]
        sigma_n = theta_full[n_kernel]
    else:
        theta_kernel = theta_full
        sigma_n = 0.0

    # Build covariance in compact banded storage: O(N*b) instead of O(N^2)
    cb = _build_banded_kernel_jax(theta_kernel, x, bandwidth,
                                   n_harmonics, n_lat, lat_range,
                                   r_gamma_func=r_gamma_func,
                                   quad_nodes=quad_nodes,
                                   quad_weights=quad_weights,
                                   edgeon_cn_sq=edgeon_cn_sq,
                                   lat_weight_func=lat_weight_func)

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
    """Validate hparam dict (shared by __init__ and update_hparam).

    Delegates to params.resolve_hparam for all validation logic; raises
    on the first error encountered.
    """
    resolve_hparam(hparam)  # raises TypeError / ValueError on bad input


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
    data_or_x : TimeSeriesData or array_like, shape (N,)
        Either a ``TimeSeriesData`` object, or observation times [days].
        When a ``TimeSeriesData`` is passed, ``y`` and ``yerr`` must be
        None (they are read from the data object).
    y : array_like, shape (N,), optional
        Observed flux values. Required when ``data_or_x`` is an array.
    yerr : array_like, shape (N,) or float, optional
        Measurement uncertainties (1-sigma). Required when ``data_or_x``
        is an array.
    model_or_hparam : SpotEvolutionModel or dict
        Either a SpotEvolutionModel (new API) or a raw hparam dict
        (backward-compatible old API).
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
        "tau_spot":  (0.05, 10.0),
        "tau_em":    (0.05, 10.0),
        "tau_dec":   (0.05, 10.0),
        "sigma_sn":  (0.05, 10.0),
        "n_sn":      (-10.0, 10.0),
        "lat_min":   (0.0, np.pi / 2),
        "lat_max":   (0.0, np.pi / 2),
        "sigma_k":   (1e-6, 1.0),
        "sigma_n":   (1e-6, 0.1),
    }

    def __init__(self, data_or_x, y=None, yerr=None, model_or_hparam=None,
                 kernel_type="analytic",
                 mean=None, fit_sigma_n=False, bounds=None,
                 log_prior=None, matrix_solver="cholesky_banded",
                 bandwidth=None, save_dir=None, **kernel_kwargs):

        # ── Parse data source ────────────────────────────────────────────
        from .observations import TimeSeriesData

        if isinstance(data_or_x, TimeSeriesData):
            # New API: GPSolver(data, model_or_hparam, ...)
            #   GPSolver(data, model)
            #   GPSolver(data, model, bounds)
            self.data = data_or_x
            # The second positional arg is the model, not y
            if model_or_hparam is None:
                model_or_hparam = y
                y = None
            # The third positional arg may be bounds (dict/array), not yerr
            if isinstance(yerr, (dict, np.ndarray)) and bounds is None:
                bounds = yerr
                yerr = None
        else:
            # Legacy API: GPSolver(x, y, yerr, model_or_hparam, ...)
            self.data = TimeSeriesData(data_or_x, y, yerr)

        self.x = jnp.asarray(self.data.x, dtype=jnp.float64)
        self.y = jnp.asarray(self.data.y, dtype=jnp.float64)
        self.yerr = jnp.asarray(self.data.yerr, dtype=jnp.float64)
        self.N = len(self.x)

        # Accept SpotEvolutionModel or legacy hparam dict
        if isinstance(model_or_hparam, SpotEvolutionModel):
            self.spot_model = model_or_hparam
            self.hparam = model_or_hparam.to_hparam()
        else:
            _validate_hparam(model_or_hparam)
            self.hparam = dict(model_or_hparam)
            self.spot_model = SpotEvolutionModel.from_hparam(self.hparam)

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

        # Use envelope-aware param_keys from the SpotEvolutionModel.
        # This replaces the old hardcoded KERNEL_HPARAM_KEYS approach.
        _model_keys = self.spot_model.param_keys  # e.g. (peq, kappa, inc, lspot, tau_spot, sigma_k)
        _base_keys = _model_keys + ("sigma_n",) if fit_sigma_n else _model_keys

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
        # bounds of lspot and tau_spot are available for the bandwidth calculation.
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

        # Bandwidth for banded solver (compile-time constant).
        # Can be set explicitly via the bandwidth argument; otherwise derived
        # from the upper bounds of envelope parameters so that the banded
        # approximation is valid for any parameters within the prior.
        if self.matrix_solver == "cholesky_banded":
            if bandwidth is not None:
                self.bandwidth = min(int(bandwidth), self.N - 1)
            else:
                self.bandwidth = self._compute_bandwidth()
            _n_banded = (self.bandwidth + 1) * self.N
            _n_full   = self.N * self.N
            _sparsity = 100.0 * (1.0 - _n_banded / _n_full)
            print(f"Banded Cholesky: bandwidth={self.bandwidth}, "
                  f"N={self.N}, sparsity={_sparsity:.1f}%")

        # Build covariance and factorize
        self._build_covariance()

        # Initial theta: physical values from spot_model, converted to log10
        # for any log-parameterized keys.
        _phys_theta0 = dict(zip(self.spot_model.param_keys,
                                self.spot_model.theta0))
        if fit_sigma_n:
            _phys_theta0["sigma_n"] = float(
                self.hparam.get("sigma_n", self.DEFAULT_BOUNDS["sigma_n"][0]))
        self.theta0 = jnp.array([
            np.log10(float(_phys_theta0.get(self._log_param_map[k], 0.0)))
            if k in self._log_param_map
            else float(_phys_theta0.get(k, 0.0))
            for k in self.param_keys
        ], dtype=jnp.float64)

        # Kernel config (extract from kernel object)
        self.n_harmonics = self.kernel.n_harmonics
        self.n_lat = self.kernel.n_lat
        # When latitude params are free, the quadrature grid must cover
        # the full hemisphere so the dynamic weights can select any sub-range.
        if self.spot_model.latitude_distribution.param_dict:
            self.lat_range = (-np.pi / 2, np.pi / 2)
        else:
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
            if self.spot_model.latitude_distribution.param_dict:
                # Latitude params are free: recompute GL nodes/weights
                # over the full hemisphere without baked-in lat_dist weights.
                gl_nodes, gl_weights = _gauss_legendre_grid(
                    self.n_lat, -np.pi / 2, np.pi / 2)
                _norm = float(jnp.sum(gl_weights))
                self._quad_nodes = gl_nodes
                self._quad_weights = gl_weights / _norm
            else:
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

        # Output directory (optional)
        if save_dir is not None:
            import os as _os
            _os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def _autosave(self, filename, **arrays):
        """Save ``arrays`` to ``save_dir/filename`` if ``save_dir`` is set."""
        if self.save_dir is None:
            return
        import os as _os
        path = _os.path.join(self.save_dir, filename)
        np.savez(path, **arrays)
        print(f"Saved {filename} → {path}")

    def _build_kernel(self):
        """Instantiate the kernel object from the SpotEvolutionModel."""
        if self.kernel_type == "analytic":
            self.kernel = AnalyticKernel(self.spot_model, **self.kernel_kwargs)
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

        Derived from the **upper bounds** of envelope parameters so that
        the banded approximation is valid for any parameters within the
        prior, regardless of the current hparam values.  This is required
        for JIT-compiled sampling: the bandwidth is a compile-time constant
        (it determines array shapes), so it must cover the entire prior
        support rather than just the initial hyperparameters.

        Assumes uniform sampling; uses ``x[1] - x[0]`` as the step size.
        """
        if self.N < 2:
            return self.N
        dt = float(self.x[1] - self.x[0])
        support = self.spot_model.bandwidth_support(self.param_keys, self.bounds)
        b = int(np.ceil(support / dt))
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

        # Envelope-specific R_Gamma function (JAX-traceable, captured in closure)
        r_gamma_fn = self.spot_model.get_r_gamma_func()
        # Latitude weight function (JAX-traceable, or None for static weights)
        lat_wt_fn = self.spot_model.get_lat_weight_func()
        # Number of kernel params (excludes sigma_n)
        n_kernel = len(self.spot_model.param_keys)

        # Edge-on fast path: pre-compute fixed |c_n|^2 as a JAX array
        if isinstance(self.spot_model.visibility, EdgeOnVisibilityFunction):
            eo_cn = jnp.array(self.spot_model.visibility.cn_squared(
                0.0, self.n_harmonics))
        else:
            eo_cn = None

        if self.matrix_solver == "cholesky_banded":
            # Capture bandwidth as a Python int in the closure so that
            # jax.lax.scan inside banded_cholesky is unrolled statically.
            b = self.bandwidth

            @jax.jit
            def log_posterior(theta_arr):
                ll = _gp_log_likelihood_banded(
                    to_phys(theta_arr), x, y, yerr, mean_val,
                    n_h, n_l, lr, fit_sn, b,
                    n_kernel=n_kernel, r_gamma_func=r_gamma_fn,
                    quad_nodes=qn, quad_weights=qw,
                    edgeon_cn_sq=eo_cn,
                    lat_weight_func=lat_wt_fn)
                lp = (custom_prior(theta_arr) if custom_prior is not None
                      else _default_log_prior(theta_arr, bounds))
                return ll + lp
        else:
            @jax.jit
            def log_posterior(theta_arr):
                ll = _gp_log_likelihood(to_phys(theta_arr), x, y, yerr, mean_val,
                                        n_h, n_l, lr, fit_sn,
                                        n_kernel=n_kernel, r_gamma_func=r_gamma_fn,
                                        quad_nodes=qn, quad_weights=qw,
                                        edgeon_cn_sq=eo_cn,
                                        lat_weight_func=lat_wt_fn)
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
    # JAX compilation warmup
    # =================================================================

    def build_jax(self, recompute=True):
        """
        Pre-compile and warm up all JAX JIT functions for this solver.

        ``_build_logposterior()`` creates four ``@jax.jit``-decorated
        functions (``log_posterior``, ``neg_log_posterior``,
        ``grad_log_posterior``, ``grad_neg_log_posterior``) that each
        trigger a separate XLA compilation on their first call.  Combined
        with the banded Cholesky solver, that can add up to 10–30 s of
        invisible overhead before a fit or MCMC run starts.

        Call ``build_jax()`` once after constructing the solver to pay
        that cost upfront.

        Returns
        -------
        self : GPSolver
            Returns ``self`` so the call can be chained:
            ``gp = GPSolver(...).build_jax()``.
        """
        import time

        theta0 = self.theta0

        t0 = time.time()
        jax.block_until_ready(self.log_posterior(theta0))
        jax.block_until_ready(self.neg_log_posterior(theta0))
        jax.block_until_ready(self.grad_log_posterior(theta0))
        jax.block_until_ready(self.grad_neg_log_posterior(theta0))
        print(f"JAX GP solver compiled in {np.round(time.time() - t0, 2)}s")

        if recompute:
            t0 = time.time()
            jax.block_until_ready(self.log_posterior(theta0))
            jax.block_until_ready(self.neg_log_posterior(theta0))
            jax.block_until_ready(self.grad_log_posterior(theta0))
            jax.block_until_ready(self.grad_neg_log_posterior(theta0))
            print(f"JAX GP solver recompute in {np.round(time.time() - t0, 2)}s")

        return self

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
                        show_legend=True, xlim=None, ylim=None, 
                        model_label="GP mean", data_label="Data"):
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
        xlim : tuple, optional
            Limits for the x-axis. If None, defaults to the data range.
        ylim : tuple, optional
            Limits for the y-axis. If None, defaults to the data range.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if theta is not None:
            phys = {}
            for k, v in (theta.items() if isinstance(theta, dict)
                         else zip(self.spot_model.param_keys, theta)):
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
                    label=data_label)
        ax.plot(xpred, mu, color=model_color, lw=1.5, label=model_label)

        alphas = {1: 0.35, 2: 0.18, 3: 0.10}
        for ns in (n_sigma if hasattr(n_sigma, "__iter__") else (n_sigma,)):
            ax.fill_between(xpred, mu - ns * sigma, mu + ns * sigma,
                            color=model_color,
                            alpha=alphas.get(ns, 0.15),
                            label=rf"$\pm{ns}\sigma$")
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(float(self.x[0]), float(self.x[-1]))
        if ylim is not None:
            ax.set_ylim(ylim)

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

    def compute_acf(self, tlags=None, n_bins=50, normalize=True):
        """
        Compute the empirical autocorrelation function of the data.

        Delegates to ``self.data.compute_acf()``.

        Parameters
        ----------
        tlags : array_like, optional
            Bin edges for time lags [days]. If provided, ``n_bins`` is
            inferred as ``len(tlags) - 1``.  If None, ``n_bins`` linearly
            spaced bins from 0 to half the baseline are used.
        n_bins : int
            Number of lag bins (used when ``tlags`` is None, default 50).
        normalize : bool
            If True (default), normalize so ACF(0) ~ 1.

        Returns
        -------
        lag_centers : ndarray, shape (n_bins,)
            Bin centers.
        acf : ndarray, shape (n_bins,)
            Empirical ACF at each bin center.
        """
        if tlags is not None:
            tlags = np.asarray(tlags, dtype=np.float64)
            max_lag = float(tlags[-1])
            n_bins = len(tlags) - 1
        else:
            max_lag = self.data.baseline / 2.0

        lag_centers, acf = self.data.compute_acf(
            n_bins=n_bins, max_lag=max_lag)

        if not normalize:
            var = np.var(np.asarray(self.y) - float(self.mean_val))
            acf = acf * var

        return lag_centers, acf

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
                disp=False, nopt=1, ncore=None, rng=None, _save=True):
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

        nopt : int
            Number of independent optimisation trials (default 1).
            When > 1, ``fit_acf_parallel`` is called and the best
            result across all trials is returned.
        ncore : int or None
            Number of parallel workers. Only used when ``nopt > 1``.
        rng : numpy.random.Generator, optional
            RNG for random starting points. Only used when ``nopt > 1``.

        Returns
        -------
        theta_dict : dict
            Full dictionary of all kernel hyperparameters (fixed + optimized).
        result : scipy.optimize.OptimizeResult
            Full optimizer output.
        """
        if nopt > 1:
            return self.fit_acf_parallel(
                nopt=nopt, ncore=ncore, keys=keys, tlags=tlags,
                n_bins=n_bins, method=method, maxiter=maxiter,
                ftol=ftol, gtol=gtol, disp=disp, rng=rng,
            )

        from scipy.optimize import minimize

        # Build lag bins
        if tlags is None:
            baseline = float(jnp.max(self.x) - jnp.min(self.x))
            tlags = np.linspace(0, baseline / 2, n_bins + 1)

        # Compute empirical ACF (unnormalized so units match the kernel)
        lag_centers, acf_data = self.compute_acf(tlags=tlags, n_bins=n_bins,
                                                  normalize=False)
        lag_centers_jax = jnp.asarray(lag_centers)
        acf_data_jax = jnp.asarray(acf_data)

        # --- Parse theta0 -------------------------------------------------
        # Use envelope-aware param_keys (excludes sigma_n)
        kernel_keys = list(self.spot_model.param_keys)
        n_kernel = len(kernel_keys)
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
        r_gamma_fn = self.spot_model.get_r_gamma_func()
        lat_wt_fn = self.spot_model.get_lat_weight_func()

        @jax.jit
        def loss_u(u_arr):
            free_theta = blo + u_arr * brange
            theta_full = self._theta_from_free(
                free_theta, free_idx, fixed_idx, fixed_vals)
            K_model = _kernel_eval(theta_full, lag_centers_jax,
                                   n_h, n_l, lr,
                                   quad_nodes=qn, quad_weights=qw,
                                   r_gamma_func=r_gamma_fn,
                                   lat_weight_func=lat_wt_fn)
            return jnp.sum((acf_data_jax - K_model) ** 2)

        vg_fn = jax.jit(jax.value_and_grad(loss_u))

        # Warm up the JIT-compiled function before the optimizer starts so
        # the CUDA kernel is compiled and timed accurately from the first call.
        jax.block_until_ready(vg_fn(jnp.array(u0, dtype=jnp.float64)))

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
        if _save:
            self._autosave("acf_fit_results.npz", theta_acf=theta_dict)
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
        kernel_keys = list(self.spot_model.param_keys)
        n_kernel = len(kernel_keys)
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

        # Generate random starting points using independent child RNGs
        seeds = rng.integers(0, 2**31, size=nopt)
        starts = []
        for i in range(nopt):
            child_rng = np.random.default_rng(int(seeds[i]))
            theta0_dict = {}
            u = child_rng.uniform(size=len(free_keys))
            for j, k in enumerate(free_keys):
                theta0_dict[k] = float(blo[j] + u[j] * (bhi[j] - blo[j]))
            starts.append(theta0_dict)

        def _run_one(theta0_dict):
            return self.fit_acf(theta0=theta0_dict, keys=keys,
                                tlags=tlags, n_bins=n_bins,
                                method=method, maxiter=maxiter,
                                ftol=ftol, gtol=gtol, disp=disp,
                                _save=False)

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
        theta_all = np.array(
            [[float(r[0][k]) for k in kernel_keys] for r in results])
        self._autosave("acf_fit_results.npz",
                       theta_acf=best_theta, theta_all=theta_all)
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
            convention as ``fit_map``: None uses ``self.theta0``, a dict
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

        n_kernel = len(self.spot_model.param_keys)  # envelope-dependent
        to_phys = self._to_physical
        qn, qw = self._quad_nodes, self._quad_weights
        n_h, n_l, lr = self.n_harmonics, self.n_lat, self.lat_range
        r_gamma_fn = self.spot_model.get_r_gamma_func()
        lat_wt_fn = self.spot_model.get_lat_weight_func()
        w_acf = float(acf_weight)
        w_psd = float(psd_weight)

        x     = np.asarray(self.x)
        resid = np.asarray(self.y) - self.mean_val
        dt_med   = float(np.median(np.diff(x)))
        baseline = float(x[-1] - x[0])

        # --- Empirical ACF (skipped when acf_weight == 0) -------------------
        if w_acf != 0.0:
            if tlags is None:
                tlags = np.linspace(0, baseline / 2, n_bins + 1)
            lag_centers, acf_data = self.compute_acf(tlags, normalize=False)
            lag_jax = jnp.asarray(lag_centers, dtype=jnp.float64)
            acf_jax = jnp.asarray(acf_data,    dtype=jnp.float64)
            acf_rms = jnp.sqrt(jnp.mean(acf_jax ** 2)) + 1e-30
        else:
            lag_jax = acf_jax = acf_rms = None

        # --- Empirical PSD via Lomb-Scargle (skipped when psd_weight == 0) --
        if w_psd != 0.0:
            freq_min  = 1.0 / baseline
            freq_max  = 1.0 / (2.0 * dt_med)
            freqs     = np.linspace(freq_min, freq_max, n_freq)
            pgram     = lombscargle(x, resid, 2.0 * np.pi * freqs, normalize=False)
            df        = freqs[1] - freqs[0]
            psd_data_norm = pgram / (np.sum(pgram) * df)
            psd_jax   = jnp.asarray(psd_data_norm, dtype=jnp.float64)
            freqs_jax = jnp.asarray(freqs,         dtype=jnp.float64)
            if dt_kernel is None:
                dt_kernel = dt_med / 5.0
            tau_grid = np.arange(0.0, baseline, dt_kernel)
            tau_jax  = jnp.asarray(tau_grid, dtype=jnp.float64)
            cos_mat  = jnp.cos(2.0 * jnp.pi * tau_jax[:, None] * freqs_jax[None, :])
        else:
            psd_jax = freqs_jax = tau_jax = cos_mat = df = None

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

        @jax.jit
        def loss_u(u_arr):
            free_theta  = blo + u_arr * brange
            theta_samp  = self._theta_from_free(
                free_theta, free_idx, fixed_idx, fixed_vals)
            theta_phys  = to_phys(theta_samp)

            loss = 0.0

            if w_acf != 0.0:
                K_acf    = _kernel_eval(theta_phys, lag_jax, n_h, n_l, lr,
                                        quad_nodes=qn, quad_weights=qw,
                                        r_gamma_func=r_gamma_fn,
                                        lat_weight_func=lat_wt_fn)
                acf_loss = jnp.mean(((acf_jax - K_acf) / acf_rms) ** 2)
                loss = loss + w_acf * acf_loss

            if w_psd != 0.0:
                K_tau = _kernel_eval(theta_phys, tau_jax, n_h, n_l, lr,
                                     quad_nodes=qn, quad_weights=qw,
                                     r_gamma_func=r_gamma_fn,
                                     lat_weight_func=lat_wt_fn)
                psd_model = jnp.maximum(
                    dt_kernel * (K_tau[0] + 2.0 * jnp.dot(K_tau[1:], cos_mat[1:])),
                    0.0)
                psd_norm  = jnp.sum(psd_model) * df
                psd_model_norm = psd_model / (psd_norm + 1e-30)
                psd_loss = jnp.mean((psd_jax - psd_model_norm) ** 2)
                loss = loss + w_psd * psd_loss

            return loss

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
        """Convert a theta dict or array to a physical kernel parameter array.

        Handles sampling-space dicts that may contain ``log_``-prefixed keys
        (e.g. ``log_sigma_k``), converting them to physical values via
        ``10 ** value`` before building the array.  Plain arrays are returned
        as-is (assumed already physical).

        Parameters
        ----------
        theta : dict or array_like
            Kernel parameters, either as a dict (physical or log-space keys)
            or an array in spot_model.param_keys order.

        Returns
        -------
        theta_arr : jnp.ndarray
            Physical kernel parameters in spot_model.param_keys order.
        """
        kernel_keys = list(self.spot_model.param_keys)
        if isinstance(theta, dict):
            phys = {}
            for k, v in theta.items():
                if k.startswith("log_"):
                    phys[k[4:]] = 10.0 ** float(v)
                else:
                    phys[k] = float(v)
            return jnp.array([float(phys[k]) for k in kernel_keys],
                             dtype=jnp.float64)
        return jnp.asarray(theta, dtype=jnp.float64)

    def plot_acf(self, theta=None, tlags=None, n_bins=50, ax=None,
                 normalize=False, data_color="k", model_color="r",
                 show_legend=True, xlim=None, ylim=None, 
                 model_label="Analytic ACF", data_label="Data ACF"):
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
        xlim : tuple, optional
            Limits for the x-axis. If None, defaults to the data range.
        ylim : tuple, optional
            Limits for the y-axis. If None, defaults to the data range.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        if tlags is None:
            baseline = float(jnp.max(self.x) - jnp.min(self.x))
            tlags = np.linspace(0, baseline / 2, n_bins + 1)

        lag_centers, acf_data = self.compute_acf(tlags=tlags, n_bins=n_bins,
                                                    normalize=normalize)

        if ax is None:
            fig, ax = plt.subplots()

        ax.plot(lag_centers, acf_data, color=data_color, label=data_label)

        if theta is not None:
            theta_arr = self._theta_dict_to_phys_array(theta)
            lag_fine = np.linspace(0.0, float(tlags[-1]), 300)
            K_model = np.asarray(_kernel_eval(
                theta_arr, jnp.asarray(lag_fine),
                self.n_harmonics, self.n_lat, self.lat_range,
                quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
                r_gamma_func=self.spot_model.get_r_gamma_func(),
                lat_weight_func=self.spot_model.get_lat_weight_func()))
            if normalize:
                var = np.mean((np.asarray(self.y) - self.mean_val) ** 2)
                if var > 0:
                    K_model = K_model / var
            ax.plot(lag_fine, K_model, color=model_color, label=model_label)

        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(min(tlags), max(tlags))
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_xlabel("Time lag [days]", fontsize=22)
        ax.set_ylabel("ACF" if normalize else "Autocovariance", fontsize=22)
        if show_legend:
            ax.legend()

        return ax

    def plot_psd(self, theta=None, n_freq=500, dt_kernel=None, ax=None,
                 data_color="k", model_color="r", show_legend=True,
                 xlim=None, ylim=None, model_label="Analytic PSD", 
                 data_label="Data Lomb-Scargle"):
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
        xlim : tuple, optional
            Limits for the x-axis. If None, defaults to the data range.
        ylim : tuple, optional
            Limits for the y-axis. If None, defaults to the data range.

        Returns
        -------
        ax : matplotlib Axes
        """
        import matplotlib.pyplot as plt

        x = np.asarray(self.x)
        resid = np.asarray(self.y) - self.mean_val
        var = float(np.mean(resid ** 2))

        baseline = float(x[-1] - x[0])
        dt_med = float(np.median(np.diff(x)))
        freq_min = 1.0 / baseline
        freq_max = 1.0 / (2.0 * dt_med)

        # Data PSD via TimeSeriesData
        freqs, psd_data = self.data.compute_psd(
            normalization="psd", n_freq=n_freq,
            freq_min=freq_min, freq_max=freq_max)
        # Normalize so ∫PSD df = var(data)
        integral = np.trapezoid(psd_data, freqs)
        if integral > 0:
            psd_data = psd_data * var / integral

        if ax is None:
            fig, ax = plt.subplots()

        ax.semilogy(freqs, psd_data, color=data_color, lw=0.8, label=data_label)

        if theta is not None:
            theta_arr = self._theta_dict_to_phys_array(theta)

            if dt_kernel is None:
                dt_kernel = dt_med / 5.0
            tau_grid = np.arange(0.0, baseline, dt_kernel)
            K = np.asarray(_kernel_eval(
                theta_arr, jnp.asarray(tau_grid),
                self.n_harmonics, self.n_lat, self.lat_range,
                quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
                r_gamma_func=self.spot_model.get_r_gamma_func(),
                lat_weight_func=self.spot_model.get_lat_weight_func()))
            # Extend to two-sided symmetric sequence, then rfft → one-sided PSD
            K_twosided = np.concatenate([K[::-1], K[1:]])
            psd_model = np.abs(np.fft.rfft(K_twosided)) * dt_kernel
            freqs_model = np.fft.rfftfreq(len(K_twosided), d=dt_kernel)
            # Restrict to the data frequency range and skip DC
            mask = (freqs_model > 0) & (freqs_model <= freq_max)
            fm, pm = freqs_model[mask], psd_model[mask]
            # Normalize so ∫PSD df = var(data)
            pm = pm * var / np.trapezoid(pm, fm)
            ax.semilogy(fm, pm, color=model_color, lw=1.5, label=model_label)
            
        if xlim is not None:
            ax.set_xlim(xlim)
        else:
            ax.set_xlim(freq_min, freq_max)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel("Frequency [1/day]", fontsize=22)
        ax.set_ylabel("PSD", fontsize=22)
        if show_legend:
            ax.legend()

        return ax

    def plot_covariance_matrix(self, theta=None, ax=None, cmap="RdBu_r",
                               show_colorbar=True, vmax=None, nbins=50,
                               show=False, filename="covariance_matrix.png"):
        """
        Plot the GP covariance matrix K (signal only, no noise).

        Entries outside the banded support are set to zero, matching the
        ``cholesky_banded`` approximation.  The matrix is binned to
        ``nbins x nbins`` before plotting.  The bandwidth boundary is drawn
        as dashed lines, and band width plus matrix sparsity are annotated.

        Parameters
        ----------
        theta : dict or array_like, optional
            Kernel hyperparameters.  Accepts a physical dict with keys from
            ``param_keys``, a sampling-space dict with ``log_``-prefixed keys,
            or a raw array.  If None, uses the current ``self.hparam`` values.
        ax : matplotlib Axes, optional
            Axes to plot on.  If None, a new figure is created.
        cmap : str, optional
            Colormap name.  Defaults to ``"RdBu_r"`` (diverging, centred at
            zero).
        show_colorbar : bool, optional
            Whether to add a colorbar.  Default True.
        vmax : float, optional
            Symmetric color scale limit ``[-vmax, vmax]``.  If None, uses the
            maximum absolute value of the banded matrix.
        nbins : int, optional
            Bin the N x N matrix down to ``nbins x nbins`` by averaging
            non-overlapping blocks before plotting.  Default 50.
        show : bool, optional
            If True, call ``plt.show()``.  Default False.
        filename : str, optional
            Filename used when saving to ``save_dir``.
            Default ``"covariance_matrix.png"``.

        Returns
        -------
        ax : matplotlib Axes
        """
        import os
        import matplotlib.pyplot as plt

        if theta is not None:
            theta_arr = self._theta_dict_to_phys_array(theta)
        else:
            theta_arr = self._theta_dict_to_phys_array(self.hparam)

        N = self.N
        b = self.bandwidth
        dt = float(self.x[1] - self.x[0]) if N > 1 else 1.0
        band_days = b * dt

        # Build banded K: evaluate only the b+1 diagonals, zero elsewhere
        K = np.zeros((N, N))
        for d in range(b + 1):
            i_idx = np.arange(N - d)
            j_idx = i_idx + d
            lags = jnp.abs(self.x[i_idx] - self.x[j_idx])
            K_diag = np.asarray(_kernel_eval(
                theta_arr, lags,
                self.n_harmonics, self.n_lat, self.lat_range,
                quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
                r_gamma_func=self.spot_model.get_r_gamma_func(),
                lat_weight_func=self.spot_model.get_lat_weight_func()))
            K[i_idx, j_idx] = K_diag
            if d > 0:
                K[j_idx, i_idx] = K_diag

        # Sparsity: fraction of entries outside the band
        n_nonzero = int(N) * (2 * int(b) + 1) - int(b) * (int(b) + 1)
        n_nonzero = min(n_nonzero, N * N)
        sparsity = 100.0 * (1.0 - n_nonzero / (N * N))

        # Bin down to nbins x nbins by block-averaging
        n_plot = min(nbins, N)
        block = N // n_plot
        n_trim = block * n_plot
        K_bin = (K[:n_trim, :n_trim]
                 .reshape(n_plot, block, n_plot, block)
                 .mean(axis=(1, 3)))

        del K

        if vmax is None:
            vmax = float(np.max(np.abs(K_bin)))

        # Bandwidth in binned-matrix units
        b_bin = b / block

        if ax is None:
            fig, ax = plt.subplots()

        im = ax.imshow(K_bin, origin="upper", cmap=cmap,
                       vmin=-vmax, vmax=vmax, aspect="auto")
        if show_colorbar:
            plt.colorbar(im, ax=ax, label="Covariance")

        # Dashed lines marking the bandwidth boundary in binned coordinates
        M = n_plot
        diag_x = np.array([-0.5, M - 0.5])
        ax.plot(diag_x + b_bin, diag_x, color="k", lw=1, ls="--", alpha=0.6)
        ax.plot(diag_x - b_bin, diag_x, color="k", lw=1, ls="--", alpha=0.6)
        ax.set_xlim(-0.5, M - 0.5)
        ax.set_ylim(M - 0.5, -0.5)

        # Sparsity annotation in upper-right corner
        ax.text(0.98, 0.02, f"sparsity = {sparsity:.1f}%",
                ha="right", va="bottom", fontsize=11,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

        ax.set_title(f"bandwidth = {b} pts ({band_days:.1f} d)", fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])

        del K_bin

        if self.save_dir is not None:
            path = os.path.join(self.save_dir, filename)
            ax.figure.savefig(path, bbox_inches="tight")
            print(f"Saved {filename} → {path}")

        if show:
            plt.show()

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
        # Normalize user-supplied keys: accept both log_ and physical forms
        # by mapping to whichever name is in self.param_keys.
        pk_set = set(self.param_keys)
        normalized = []
        for k in keys:
            if k in pk_set:
                normalized.append(k)
            elif k.startswith("log_") and k[4:] in pk_set:
                normalized.append(k[4:])
            elif f"log_{k}" in pk_set:
                normalized.append(f"log_{k}")
            else:
                raise ValueError(
                    f"Unknown key '{k}'. Valid keys: {self.param_keys}")
        keys = normalized
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

    def _update_model_from_theta(self, theta_dict):
        """Update the SpotEvolutionModel components from a theta-style dict.

        The keys in *theta_dict* must be a subset of
        ``self.spot_model.param_keys`` (physical names like ``tau_em``,
        ``lat_min``, ``sigma_k``).  Each component (visibility, envelope,
        latitude distribution) is reconstructed with updated values.
        """
        model = self.spot_model
        # Visibility
        if model.visibility is not None:
            for k in model.visibility.param_keys:
                if k in theta_dict:
                    setattr(model.visibility, k, float(theta_dict[k]))
        # Envelope: reconstruct from updated param_dict to handle
        # varying internal storage (distributions vs plain floats)
        if model.envelope is not None:
            env_params = dict(model.envelope.param_dict)
            env_params.update({k: float(theta_dict[k])
                               for k in env_params if k in theta_dict})
            model.envelope = type(model.envelope)(**env_params)
        # Latitude distribution: reconstruct with updated values
        lat = model.latitude_distribution
        if lat.param_dict:
            lat_params = dict(lat.param_dict)
            lat_params.update({k: float(theta_dict[k])
                               for k in lat_params if k in theta_dict})
            # Convert from radians to degrees for constructor
            init_args = {}
            for k, v in lat_params.items():
                if "lat" in k:
                    init_args[k.replace("lat_", "") + "_lat_deg"] = float(np.rad2deg(v))
                else:
                    init_args[k] = v
            model.latitude_distribution = type(lat)(**init_args)
        if "sigma_k" in theta_dict:
            model.sigma_k = float(theta_dict["sigma_k"])

    def update_hparam(self, hparam):
        """Update hyperparameters and rebuild kernel and covariance.

        Accepts a SpotEvolutionModel, an hparam dict (legacy keys like
        ``lspot``), or a theta-style dict whose keys match
        ``self.spot_model.param_keys`` (e.g. ``tau_em``, ``lat_min``).
        """
        if isinstance(hparam, SpotEvolutionModel):
            self.spot_model = hparam
            self.hparam = hparam.to_hparam()
        elif set(hparam.keys()) <= set(self.spot_model.param_keys):
            # Theta-style dict: update the existing model components directly
            self._update_model_from_theta(hparam)
            self.hparam = self.spot_model.to_hparam()
        else:
            _validate_hparam(hparam)
            self.hparam = dict(hparam)
            self.spot_model = SpotEvolutionModel.from_hparam(self.hparam)
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
        _phys_theta0 = dict(zip(self.spot_model.param_keys,
                                self.spot_model.theta0))
        self.theta0 = jnp.array([
            np.log10(float(_phys_theta0.get(self._log_param_map[k], 0.0)))
            if k in self._log_param_map
            else float(_phys_theta0.get(k, 0.0))
            for k in self.param_keys
        ], dtype=jnp.float64)

    # =================================================================
    # MAP estimation
    # =================================================================

    def fit_map(self, theta0=None, keys=None, method="L-BFGS-B",
                 maxiter=500, ftol=0, gtol=1e-8, disp=False, nopt=1,
                 ncore=None, rng=None, _save=True):
        """
        Find the maximum a posteriori (MAP) estimate.

        Uses scipy.optimize.minimize with JAX-computed gradients.
        When ``nopt > 1``, delegates to ``fit_map_parallel`` which
        runs ``nopt`` independent trials from random starting points
        (drawn uniformly within the bounds) and returns the best result.

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

            Ignored when ``nopt > 1`` (starting points are randomised).
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
        nopt : int
            Number of independent optimisation trials (default 1).
            When > 1, ``fit_map_parallel`` is called and the best
            result across all trials is returned.
        ncore : int or None
            Number of parallel workers for multi-start runs. If None,
            uses ``nopt`` or the number of available CPUs, whichever is
            smaller. Only used when ``nopt > 1``.
        rng : numpy.random.Generator, optional
            RNG for random starting points. Only used when ``nopt > 1``.

        Returns
        -------
        theta_dict : dict
            Full dictionary of all hyperparameters (fixed + optimized).
        result : scipy OptimizeResult
            Full optimizer output.
        """
        if nopt > 1:
            return self.fit_map_parallel(
                nopt=nopt, ncore=ncore, keys=keys, method=method,
                maxiter=maxiter, ftol=ftol, gtol=gtol, disp=disp, rng=rng,
            )
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

        # Warm up the JIT-compiled function before the optimizer starts so
        # the CUDA kernel is compiled and timed accurately from the first call.
        jax.block_until_ready(vg_fn(jnp.array(u0, dtype=jnp.float64)))

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
        theta_dict = self._result_dict(theta_full)
        if _save:
            self._autosave("map_fit_results.npz", theta_map=theta_dict)
        return theta_dict, result

    def fit_map_parallel(self, nopt=10, ncore=None, keys=None,
                          method="nelder-mead", maxiter=500, ftol=0,
                          gtol=1e-8, disp=False, return_all=False,
                          rng=None):
        """
        Run ``fit_map`` from multiple random starting points in parallel.

        Starting points are drawn uniformly within the parameter bounds.

        Parameters
        ----------
        nopt : int
            Number of independent optimization trials (default 10).
        ncore : int or None
            Number of parallel workers. If None, uses ``nopt`` or the
            number of available CPUs, whichever is smaller.
        keys : list of str, optional
            Free parameters (forwarded to ``fit_map``).
        method : str
            Optimizer method (default "nelder-mead").
        maxiter, ftol, gtol, disp
            Forwarded to ``fit_map``.
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

        # Generate random starting points using independent child RNGs
        seeds = rng.integers(0, 2**31, size=nopt)
        starts = []
        for i in range(nopt):
            child_rng = np.random.default_rng(int(seeds[i]))
            theta0_dict = {}
            u = child_rng.uniform(size=len(free_keys))
            for j, k in enumerate(free_keys):
                theta0_dict[k] = float(blo[j] + u[j] * (bhi[j] - blo[j]))
            starts.append(theta0_dict)

        # Worker function (must be top-level-picklable for ProcessPool,
        # so we use ThreadPool which shares memory with JAX)
        def _run_one(theta0_dict):
            return self.fit_map(theta0=theta0_dict, keys=keys,
                                 method=method, maxiter=maxiter,
                                 ftol=ftol, gtol=gtol, disp=disp,
                                 _save=False)

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
        theta_all = np.array(
            [[float(r[0][k]) for k in self.param_keys] for r in results])
        self._autosave("map_fit_results.npz",
                       theta_map=best_theta, theta_all=theta_all)
        return best_theta, best_result

    # =================================================================
    # Mass matrix helpers
    # =================================================================

    def _build_neg_log_lik(self, force_dense=False):
        """Return a JIT-compiled negative log-likelihood function.

        Parameters
        ----------
        force_dense : bool
            If True, always use the dense ``_gp_log_likelihood`` regardless of
            ``self.matrix_solver``.  Required for second-order autodiff (Hessian):
            the banded compact storage format does not differentiate cleanly
            through JAX's AD and produces NaNs in the Hessian.
        """
        x, y, yerr = self.x, self.y, self.yerr
        mean_val = self.mean_val
        n_h, n_l, lr = self.n_harmonics, self.n_lat, self.lat_range
        fit_sn = self.fit_sigma_n
        qn, qw = self._quad_nodes, self._quad_weights
        to_phys = self._to_physical

        if self.matrix_solver == "cholesky_banded" and not force_dense:
            b = self.bandwidth

            @jax.jit
            def neg_log_lik(theta_arr):
                return -_gp_log_likelihood_banded(
                    to_phys(theta_arr), x, y, yerr, mean_val,
                    n_h, n_l, lr, fit_sn, b,
                    quad_nodes=qn, quad_weights=qw)
        else:
            @jax.jit
            def neg_log_lik(theta_arr):
                return -_gp_log_likelihood(
                    to_phys(theta_arr), x, y, yerr, mean_val,
                    n_h, n_l, lr, fit_sn,
                    quad_nodes=qn, quad_weights=qw)

        return neg_log_lik

    # =================================================================
    # Mass matrix estimation: Method 1 -- Hessian at MAP
    # =================================================================

    def mass_matrix_hessian_map(self, theta_map=None):
        """
        Estimate the inverse mass matrix from the Hessian of the
        negative log-likelihood at the MAP.

        ``M^{-1} = H^{-1}``  where  ``H = d^2(-log L)/d theta^2`` at the MAP.

        Parameters
        ----------
        theta_map : array_like, optional
            MAP estimate. If None, calls fit_map() first.

        Returns
        -------
        inv_mass_matrix : jnp.ndarray, shape (n_params, n_params)
        """
        if theta_map is None:
            if self.map_estimate is None:
                self.fit_map()
            theta_map = self.map_estimate
        else:
            theta_map = jnp.asarray(theta_map, dtype=jnp.float64)

        # Always use dense solver: banded compact storage does not differentiate
        # cleanly through JAX's second-order AD and produces NaNs in the Hessian.
        neg_log_lik = self._build_neg_log_lik(force_dense=True)

        hessian_fn = jax.jit(jax.hessian(neg_log_lik))
        H = jax.block_until_ready(hessian_fn(theta_map))

        # Regularize: ensure positive-definite
        eigvals, eigvecs = jnp.linalg.eigh(H)
        eigvals = jnp.maximum(eigvals, 1e-6)
        H_reg = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

        self.inverse_mass_matrix = jnp.linalg.inv(H_reg)
        self._hessian = H
        return self.inverse_mass_matrix

    # =================================================================
    # Mass matrix estimation: Method 2 -- Fisher information (analytic)
    # =================================================================

    def mass_matrix_fisher(self, theta_map=None, eigval_clip=1e-6, white_noise=1e-8):
        """
        Estimate the inverse mass matrix from the Fisher information.

        For the GP log-likelihood:

            I_{ij} = (1/2) tr(K^{-1} dK/dtheta_i  K^{-1} dK/dtheta_j)

        When ``matrix_solver="cholesky_full"``, the kernel derivatives
        dK/dtheta_i are computed via JAX forward-mode autodiff (jacfwd)
        on the full N×N covariance matrix.

        When ``matrix_solver="cholesky_banded"``, the exact Fisher requires
        the dense N×N kernel and its inverse, which would defeat the purpose
        of banded storage.  Instead, the Fisher is approximated by the
        Hessian of the banded negative log-likelihood at the MAP
        (Fisher ≈ observed information at the MLE).

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
                self.fit_map()
            theta_map = self.map_estimate
        else:
            theta_map = jnp.asarray(theta_map, dtype=jnp.float64)

        # Banded path: approximate Fisher via Hessian of dense log-likelihood.
        # Must use force_dense=True — banded compact storage does not differentiate
        # cleanly through JAX's second-order AD and produces NaNs in the Hessian.
        if self.matrix_solver == "cholesky_banded":
            neg_log_lik = self._build_neg_log_lik(force_dense=True)
            hessian_fn = jax.jit(jax.hessian(neg_log_lik))
            H = jax.block_until_ready(hessian_fn(theta_map))

            eigvals, eigvecs = jnp.linalg.eigh(H)
            eigvals = jnp.maximum(eigvals, eigval_clip)
            fisher_reg = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

            self.inverse_mass_matrix = jnp.linalg.inv(fisher_reg)
            self._fisher_matrix = H
            return self.inverse_mass_matrix

        # Dense path: exact Fisher via kernel Jacobian
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
                                  quad_nodes=qn, quad_weights=qw,
                                  r_gamma_func=self.spot_model.get_r_gamma_func(),
                                  lat_weight_func=self.spot_model.get_lat_weight_func())
            K = K_flat.reshape(N, N)
            noise_var = self.yerr ** 2 + sigma_n ** 2
            K_noise = K + jnp.diag(noise_var) + white_noise * jnp.eye(N)
            return K_noise.ravel()

        jacfwd_fn = jax.jit(jax.jacfwd(K_noise_flat_from_theta))
        dK_flat_dtheta = jax.block_until_ready(jacfwd_fn(theta_map))
        dK_dtheta = dK_flat_dtheta.reshape(N, N, n_params)

        K_noise_flat = K_noise_flat_from_theta(theta_map)
        K = K_noise_flat.reshape(N, N)
        K_inv = jnp.linalg.inv(K)

        K_inv_dK = jnp.einsum('ab,bcj->acj', K_inv, dK_dtheta)
        fisher = 0.5 * jnp.einsum('abi,baj->ij', K_inv_dK, K_inv_dK)

        # Regularize
        eigvals, eigvecs = jnp.linalg.eigh(fisher)
        eigvals = jnp.maximum(eigvals, eigval_clip)
        fisher_reg = eigvecs @ jnp.diag(eigvals) @ eigvecs.T

        self.inverse_mass_matrix = jnp.linalg.inv(fisher_reg)
        self._fisher_matrix = fisher
        return self.inverse_mass_matrix

    # =================================================================
    # Mass matrix estimation: Method 3 -- Laplace approximation
    # =================================================================

    def mass_matrix_laplace(self, theta_map=None, eigval_clip=1e-6):
        """
        Laplace approximation: inverse mass matrix = inverse Hessian
        of the negative log-likelihood at the MAP.

        The posterior is approximated as:

            p(theta | data) ~ N(theta_MAP, H^{-1})

        Parameters
        ----------
        theta_map : array_like, optional
            MAP estimate. If None, calls fit_map() first.

        Returns
        -------
        inv_mass_matrix : jnp.ndarray, shape (n_params, n_params)
        """
        if theta_map is None:
            if self.map_estimate is None:
                self.fit_map()
            theta_map = self.map_estimate
        else:
            theta_map = jnp.asarray(theta_map, dtype=jnp.float64)

        neg_log_lik = self._build_neg_log_lik()

        hessian_fn = jax.jit(jax.hessian(neg_log_lik))
        H = jax.block_until_ready(hessian_fn(theta_map))

        # Regularize
        eigvals, eigvecs = jnp.linalg.eigh(H)
        eigvals = jnp.maximum(eigvals, eigval_clip)
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
            hessian_fn = jax.jit(jax.hessian(self.neg_log_posterior))
            H = jax.block_until_ready(hessian_fn(theta_ref))
            diag = jnp.maximum(jnp.diag(H), 1e-6)
            self.inverse_mass_matrix = jnp.diag(1.0 / diag)
        else:
            raise ValueError(f"Unknown mass_matrix_method: {method}")

        return self.inverse_mass_matrix
