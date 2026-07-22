"""
JAX-accelerated Gaussian Process solver using the starspot analytic kernel.

Uses JAX for JIT-compiled covariance matrix construction, Cholesky
factorization, and GP operations (log-likelihood, prediction, MAP
estimation, and mass matrix computation for MCMC).
"""
import logging
import os
import warnings

import jax
jax.config.update("jax_enable_x64", True)
if "jax_platforms" not in os.environ.get("XLA_FLAGS", "") and "JAX_PLATFORMS" not in os.environ:
    jax.config.update("jax_platforms", "cpu")

import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np

from .params import (
    resolve_hparam, KERNEL_HPARAM_KEYS, HPARAM_KEYS_WITH_NOISE,
)
from .spot_model import (
    SpotEvolutionModel, VisibilityFunction, EdgeOnVisibilityFunction,
    _gauss_legendre_grid,
)
from .analytic_kernel import (
    AnalyticKernel, _kernel_eval, _kernel_eval_edgeon,
)
from .banded_cholesky import (banded_cholesky, banded_solve,
                               banded_cholesky_compact, banded_solve_compact)
from .fitting import FittingMixin
from .gp_plots import GPPlotsMixin
from .mass_matrix import MassMatrixMixin
from .validation import (
    validate_data, validate_data_vs_model, raise_cholesky_error,
)

__all__ = ["GPSolver"]

logger = logging.getLogger("spotgp")

# KERNEL_HPARAM_KEYS and HPARAM_KEYS_WITH_NOISE are imported from params.
# Re-export as lists for any callers that expect list type.
KERNEL_HPARAM_KEYS = list(KERNEL_HPARAM_KEYS)
HPARAM_KEYS_WITH_NOISE = list(HPARAM_KEYS_WITH_NOISE)


# _kernel_eval and _kernel_eval_edgeon are imported from analytic_kernel.py
# (the single source of truth for kernel math).


def _gp_log_likelihood(theta_full, x, y, yerr, mean_val,
                       n_harmonics, n_lat, lat_range,
                       fit_sigma_n, n_kernel=6,
                       r_gamma_func=None,
                       quad_nodes=None, quad_weights=None,
                       edgeon_cn_sq=None,
                       lat_weight_func=None,
                       cn_sq_func=None,
                       uniform_dt=None,
                       lag_table=None):
    """
    Pure-functional GP marginal log-likelihood.

    When ``uniform_dt`` is given (Toeplitz fast path), the stationary
    kernel takes only N distinct values over the whole matrix, so it is
    evaluated once per distinct lag and gathered: K[i, j] = K1d[|i - j|].
    ``lag_table`` generalizes this to uniform-cadence data *with gaps*
    (the common Kepler/TESS case): the kernel is evaluated once per
    distinct integer-cadence lag and gathered through a precomputed
    index table.  Otherwise the kernel is evaluated on the
    upper-triangular lags (N*(N+1)/2 instead of N^2), halving memory and
    compute relative to the naive full-matrix evaluation.

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
    uniform_dt : float or None
        Sample spacing when ``x`` is a uniform grid (compile-time
        constant). Enables the Toeplitz fast path; None disables it.
    lag_table : (jnp.ndarray, jnp.ndarray) or None
        ``(unique_lags, inv_idx)`` for uniform-cadence data with gaps:
        ``unique_lags`` holds the distinct pairwise lags and ``inv_idx``
        is an (N, N) integer map with ``|x_i - x_j| =
        unique_lags[inv_idx[i, j]]``.  Ignored when ``uniform_dt`` is
        given.

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

    if uniform_dt is not None:
        # Toeplitz fast path: N distinct lags instead of N*(N+1)/2.
        lag_unique = jnp.arange(N) * uniform_dt
        K1d = _kernel_eval(theta_kernel, lag_unique,
                           n_harmonics, n_lat, lat_range,
                           quad_nodes=quad_nodes, quad_weights=quad_weights,
                           r_gamma_func=r_gamma_func,
                           edgeon_cn_sq=edgeon_cn_sq,
                           lat_weight_func=lat_weight_func,
                           cn_sq_func=cn_sq_func)
        idx = jnp.abs(jnp.arange(N)[:, None] - jnp.arange(N)[None, :])
        K = K1d[idx]
    elif lag_table is not None:
        # Gappy-uniform fast path: distinct lags gathered by index table.
        lags_unique, inv_idx = lag_table
        K1d = _kernel_eval(theta_kernel, lags_unique,
                           n_harmonics, n_lat, lat_range,
                           quad_nodes=quad_nodes, quad_weights=quad_weights,
                           r_gamma_func=r_gamma_func,
                           edgeon_cn_sq=edgeon_cn_sq,
                           lat_weight_func=lat_weight_func,
                           cn_sq_func=cn_sq_func)
        K = K1d[inv_idx]
    else:
        # Upper-triangular indices (includes diagonal)
        row_idx, col_idx = jnp.triu_indices(N)
        lag_upper = jnp.abs(x[row_idx] - x[col_idx])

        K_upper = _kernel_eval(theta_kernel, lag_upper,
                               n_harmonics, n_lat, lat_range,
                               quad_nodes=quad_nodes, quad_weights=quad_weights,
                               r_gamma_func=r_gamma_func,
                               edgeon_cn_sq=edgeon_cn_sq,
                               lat_weight_func=lat_weight_func,
                               cn_sq_func=cn_sq_func)

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
                              lat_weight_func=None,
                              cn_sq_func=None,
                              uniform_dt=None,
                              band_lag_table=None):
    """
    Build the kernel covariance directly in compact (b+1, N) banded storage.

    Evaluates only the O(N*b) within-band lags instead of the full N^2
    matrix.  When ``uniform_dt`` is given (Toeplitz fast path), every
    diagonal d of the band holds the single value K(d*dt), so the kernel
    is evaluated at just b+1 distinct lags and broadcast across the band
    — O(b) kernel evaluations instead of O(N*b).  ``band_lag_table``
    generalizes this to uniform-cadence data with gaps: the kernel is
    evaluated once per distinct within-band lag and gathered through a
    precomputed index table.

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
    uniform_dt : float or None
        Sample spacing when ``x`` is a uniform grid (compile-time
        constant). Enables the Toeplitz fast path; None disables it.
    band_lag_table : (jnp.ndarray, jnp.ndarray) or None
        ``(unique_lags, inv_idx)`` for uniform-cadence data with gaps:
        ``unique_lags`` holds the distinct within-band lags and
        ``inv_idx`` is a (b+1, N) integer map with ``x[i+d] - x[i] =
        unique_lags[inv_idx[d, i]]``.  Ignored when ``uniform_dt`` is
        given.

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
    valid = j_idx < N

    if uniform_dt is not None:
        # Toeplitz fast path: b+1 distinct lags instead of (b+1)*N.
        lag_unique = jnp.arange(b + 1) * uniform_dt
        K_unique = _kernel_eval(theta_kernel, lag_unique,
                                n_harmonics, n_lat, lat_range,
                                quad_nodes=quad_nodes,
                                quad_weights=quad_weights,
                                r_gamma_func=r_gamma_func,
                                edgeon_cn_sq=edgeon_cn_sq,
                                lat_weight_func=lat_weight_func,
                                cn_sq_func=cn_sq_func)
        return jnp.where(valid, K_unique[:, None], 0.0)

    if band_lag_table is not None:
        # Gappy-uniform fast path: distinct within-band lags, gathered
        # through the precomputed index table.
        lags_unique, inv_idx = band_lag_table
        K_unique = _kernel_eval(theta_kernel, lags_unique,
                                n_harmonics, n_lat, lat_range,
                                quad_nodes=quad_nodes,
                                quad_weights=quad_weights,
                                r_gamma_func=r_gamma_func,
                                edgeon_cn_sq=edgeon_cn_sq,
                                lat_weight_func=lat_weight_func,
                                cn_sq_func=cn_sq_func)
        return jnp.where(valid, K_unique[inv_idx], 0.0)

    j_safe = jnp.minimum(j_idx, N - 1)
    lags_flat = jnp.abs(x[j_safe] - x[i_idx]).ravel()
    K_flat = _kernel_eval(theta_kernel, lags_flat,
                          n_harmonics, n_lat, lat_range,
                          quad_nodes=quad_nodes, quad_weights=quad_weights,
                          r_gamma_func=r_gamma_func,
                          edgeon_cn_sq=edgeon_cn_sq,
                          lat_weight_func=lat_weight_func,
                          cn_sq_func=cn_sq_func)
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
                               lat_weight_func=None,
                               cn_sq_func=None,
                               uniform_dt=None,
                               band_lag_table=None):
    """
    Pure-functional GP marginal log-likelihood using banded Cholesky.

    Builds the covariance directly in compact (b+1, N) banded storage,
    evaluating only the O(N*b) within-band lags — or, on a uniform grid
    (``uniform_dt`` given), only the b+1 distinct lags of the Toeplitz
    band.  The Cholesky factorization and solve operate entirely in
    compact storage — no (N, N) matrix is ever allocated.

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
    uniform_dt : float or None
        Sample spacing when ``x`` is a uniform grid (compile-time
        constant). Enables the Toeplitz fast path; None disables it.
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
                                   lat_weight_func=lat_weight_func,
                                   cn_sq_func=cn_sq_func,
                                   uniform_dt=uniform_dt,
                                   band_lag_table=band_lag_table)

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


def _detect_uniform_dt(x, rtol=1e-6):
    """
    Return the sample spacing if ``x`` is a uniform grid, else None.

    A uniform grid enables the Toeplitz fast path: the stationary kernel
    takes only one distinct value per diagonal of the covariance matrix,
    so it can be evaluated once per distinct lag instead of once per
    matrix entry.

    Parameters
    ----------
    x : array_like, shape (N,)
        Observation times, assumed sorted.
    rtol : float
        Maximum relative deviation of any spacing from the median
        spacing (default 1e-6).

    Returns
    -------
    dt : float or None
        The common spacing, or None if the grid is irregular.
        Uniform-cadence data *with gaps* also returns None here but is
        handled by the second detection tier
        (``_detect_cadence_offsets``).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return None
    diffs = np.diff(x)
    dt = float(np.median(diffs))
    if dt <= 0.0:
        return None
    if np.allclose(diffs, dt, rtol=rtol, atol=0.0):
        return dt
    return None


def _detect_cadence_offsets(x, rtol=1e-3):
    """
    Detect a uniform cadence with gaps and return integer lag offsets.

    Handles the common Kepler/TESS case: samples on a regular cadence
    ``dt`` with missing chunks.  Every pairwise lag is then an integer
    multiple of ``dt``, so the stationary kernel takes far fewer distinct
    values than there are matrix entries and can be evaluated once per
    distinct within-band lag (see ``GPSolver._band_lag_table``).

    Parameters
    ----------
    x : array_like, shape (N,)
        Observation times, assumed sorted.
    rtol : float
        Maximum deviation of any sample from its cadence grid point,
        relative to ``dt`` (default 1e-3).

    Returns
    -------
    dt : float or None
        The cadence, or None if ``x`` is not on a uniform cadence grid.
    offsets : np.ndarray of int64 or None
        Integer grid index of each sample, ``x ≈ x[0] + offsets * dt``.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return None, None
    diffs = np.diff(x)
    dt = float(np.median(diffs))
    if dt <= 0.0:
        return None, None
    k = np.round((x - x[0]) / dt)
    on_grid = np.max(np.abs(x - (x[0] + k * dt))) <= rtol * dt
    if on_grid and np.all(np.diff(k) >= 1):
        return dt, k.astype(np.int64)
    return None, None


def _make_lbfgsb_batch_runner(vg_fn, n_free, maxiter, gtol):
    """
    Build a compiled multi-start L-BFGS-B runner.

    The returned callable maps a ``(n_starts, n_free)`` batch of starting
    points in unit-cube coordinates to ``(u, fun, nit)`` arrays, running
    every restart inside a single ``jax.vmap``-ed ``jaxopt.LBFGSB``
    program instead of one scipy optimizer per start in a thread pool.

    Compilation of the vmapped solver is expensive (it traces the full
    while-loop optimizer around the covariance build), so callers cache
    the returned runner and reuse it across calls; only the first call
    per (objective, maxiter, gtol) pays the compile.

    Parameters
    ----------
    vg_fn : callable
        JAX-traceable objective mapping a length-``n_free`` vector in
        [0, 1] coordinates to ``(value, grad)``.
    n_free : int
        Number of free parameters.
    maxiter : int
        Maximum L-BFGS-B iterations per start.
    gtol : float
        Convergence tolerance passed to the solver.

    Raises
    ------
    ImportError
        If jaxopt is not installed.  Callers fall back to the
        thread-pool implementation.
    """
    from jaxopt import LBFGSB

    solver = LBFGSB(fun=vg_fn, value_and_grad=True,
                    maxiter=int(maxiter), tol=float(gtol))
    lo = jnp.zeros(n_free, dtype=jnp.float64)
    hi = jnp.ones(n_free, dtype=jnp.float64)

    def _run_one(u0):
        res = solver.run(u0, bounds=(lo, hi))
        return res.params, res.state.value, res.state.iter_num

    vmapped = jax.jit(jax.vmap(_run_one))

    def runner(u0_batch):
        u, fun, nit = vmapped(jnp.asarray(u0_batch, dtype=jnp.float64))
        return np.asarray(u), np.asarray(fun), np.asarray(nit)

    return runner


# =====================================================================
# GPSolver class
# =====================================================================

class GPSolver(FittingMixin, GPPlotsMixin, MassMatrixMixin):
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

        # ── Validate data quality ───────────────────────────────────────
        validate_data(self.data.x, self.data.y, self.data.yerr)

        # Detect uniform sampling.  On a uniform grid the covariance is
        # Toeplitz, so the kernel needs evaluating only once per distinct
        # lag (b+1 values for the banded solver, N for the full solver)
        # instead of once per matrix entry.  None for irregular grids.
        self.uniform_dt = _detect_uniform_dt(self.data.x)

        # Second tier: uniform cadence *with gaps* (Kepler/TESS-style).
        # Pairwise lags are then integer multiples of the cadence, so the
        # kernel is evaluated once per distinct within-band lag and
        # gathered through a precomputed index table (see
        # _band_lag_table / _full_lag_table).
        if self.uniform_dt is None:
            self._cadence_dt, self._cadence_offsets = \
                _detect_cadence_offsets(self.data.x)
        else:
            self._cadence_dt, self._cadence_offsets = self.uniform_dt, None
        self._band_lag_table_cache = None
        self._full_lag_table_cache = None

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

        # ── Validate data vs model configuration ───────────────────────
        validate_data_vs_model(self.data.x, self.bounds, self.param_keys)

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
            _toep = (", Toeplitz fast path (uniform grid)"
                     if self.uniform_dt is not None else "")
            logger.info("Banded Cholesky: bandwidth=%d, N=%d, sparsity=%.1f%%%s",
                       self.bandwidth, self.N, _sparsity, _toep)
            self._check_bandwidth_efficiency()

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
        logger.debug("Saved %s → %s", filename, path)

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

        Uses the detected uniform spacing when available, otherwise the
        median spacing (robust to gaps, unlike ``x[1] - x[0]``).
        """
        if self.N < 2:
            return self.N
        if self.uniform_dt is not None:
            dt = self.uniform_dt
        else:
            dt = float(np.median(np.diff(np.asarray(self.x))))
        support = self.spot_model.bandwidth_support(self.param_keys, self.bounds)
        b = int(np.ceil(support / dt))
        return min(b, self.N - 1)

    def _check_bandwidth_efficiency(self):
        """Warn when the band is so wide that banded loses to dense.

        The bandwidth is derived from the prior *upper bounds* of the
        envelope parameters, so a generous prior (e.g. ``lspot`` up to
        tens of days) combined with a fine cadence can silently push
        ``b`` toward ``N`` — where the O(N b^2) banded solver costs as
        much as dense O(N^3) Cholesky while adding overhead.
        """
        if 2 * self.bandwidth >= self.N:
            warnings.warn(
                f"Banded solver bandwidth b={self.bandwidth} is >= N/2 "
                f"(N={self.N}), so the banded Cholesky has (almost) no "
                "sparsity to exploit and is likely slower than the dense "
                "solver. The bandwidth is set by the prior upper bounds "
                "of the envelope parameters (kernel support ~ lspot + "
                "2*tau_spot) divided by the cadence; tighten those "
                "bounds, downsample the lightcurve, or switch to "
                "matrix_solver='cholesky_full'.",
                stacklevel=3)

    def _band_lag_table(self):
        """(unique_lags, inv_idx) for the banded build on gappy-uniform grids.

        Returns None when the grid is strictly uniform (the cheaper
        ``uniform_dt`` broadcast applies) or genuinely irregular.  The
        table is cached; ``inv_idx`` has shape (b+1, N) and satisfies
        ``x[min(i+d, N-1)] - x[i] = unique_lags[inv_idx[d, i]]``.
        """
        if self.uniform_dt is not None or self._cadence_offsets is None:
            return None
        if self._band_lag_table_cache is None:
            off = self._cadence_offsets
            b, N = self.bandwidth, self.N
            d = np.arange(b + 1)[:, None]
            i = np.arange(N)[None, :]
            j = np.minimum(i + d, N - 1)
            lag_int = off[j] - off[i]
            uniq, inv = np.unique(lag_int, return_inverse=True)
            self._band_lag_table_cache = (
                jnp.asarray(uniq * self._cadence_dt, dtype=jnp.float64),
                jnp.asarray(inv.reshape(b + 1, N).astype(np.int32)))
        return self._band_lag_table_cache

    def _full_lag_table(self):
        """(unique_lags, inv_idx) for the dense build on gappy-uniform grids.

        Returns None when the grid is strictly uniform or irregular.
        ``inv_idx`` has shape (N, N) with ``|x_i - x_j| =
        unique_lags[inv_idx[i, j]]``; the dense path is O(N^2) anyway,
        so the int32 index table does not change its memory scaling.
        """
        if self.uniform_dt is not None or self._cadence_offsets is None:
            return None
        if self._full_lag_table_cache is None:
            off = self._cadence_offsets
            lag_int = np.abs(off[:, None] - off[None, :])
            uniq, inv = np.unique(lag_int, return_inverse=True)
            self._full_lag_table_cache = (
                jnp.asarray(uniq * self._cadence_dt, dtype=jnp.float64),
                jnp.asarray(inv.reshape(self.N, self.N).astype(np.int32)))
        return self._full_lag_table_cache

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
            valid = j_idx < N

            band_tab = self._band_lag_table()
            if self.uniform_dt is not None:
                # Toeplitz fast path: b+1 distinct lags, broadcast per row
                K_unique = self._eval_kernel(
                    jnp.arange(b + 1) * self.uniform_dt)
                cb = jnp.where(valid, K_unique[:, None], 0.0)
            elif band_tab is not None:
                # Gappy-uniform fast path: distinct within-band lags,
                # gathered through the precomputed index table
                K_unique = self._eval_kernel(band_tab[0])
                cb = jnp.where(valid, K_unique[band_tab[1]], 0.0)
            else:
                j_safe = jnp.minimum(j_idx, N - 1)
                lags_flat = jnp.abs(self.x[j_safe] - self.x[i_idx]).ravel()
                K_flat = self._eval_kernel(lags_flat)
                cb = K_flat.reshape(b + 1, N)
                cb = jnp.where(valid, cb, 0.0)

            # Add noise to diagonal (row 0)
            cb = cb.at[0, :].add(self.yerr ** 2 + 1e-8)

            self.K = None
            self.K_noise = None
            try:
                self._Lc = banded_cholesky_compact(cb, self.bandwidth)
            except Exception as e:
                raise_cholesky_error(
                    e, theta=self.theta0 if hasattr(self, 'theta0') else None,
                    param_keys=self.param_keys if hasattr(self, 'param_keys') else None,
                    bounds=self.bounds if hasattr(self, 'bounds') else None,
                    context="initial covariance build (banded solver)")
            self._L = None
            self._alpha = banded_solve_compact(
                self._Lc, self._resid, self.bandwidth)
        else:
            N = self.N
            full_tab = self._full_lag_table()
            if self.uniform_dt is not None:
                # Toeplitz fast path: N distinct lags, gathered by |i - j|
                K1d = self._eval_kernel(jnp.arange(N) * self.uniform_dt)
                idx = jnp.abs(jnp.arange(N)[:, None] - jnp.arange(N)[None, :])
                K = K1d[idx]
            elif full_tab is not None:
                # Gappy-uniform fast path: distinct lags via index table
                K1d = self._eval_kernel(full_tab[0])
                K = K1d[full_tab[1]]
            else:
                row_idx, col_idx = jnp.triu_indices(N)
                lag_upper = jnp.abs(self.x[row_idx] - self.x[col_idx])
                K_upper = self._eval_kernel(lag_upper)

                K = jnp.zeros((N, N))
                K = K.at[row_idx, col_idx].set(K_upper)
                K = K + K.T - jnp.diag(jnp.diag(K))
            self.K = K
            self.K_noise = K + jnp.diag(self.yerr ** 2) + 1e-8 * jnp.eye(N)

            self._Lc = None
            try:
                self._L = jla.cholesky(self.K_noise, lower=True)
            except Exception as e:
                raise_cholesky_error(
                    e, theta=self.theta0 if hasattr(self, 'theta0') else None,
                    param_keys=self.param_keys if hasattr(self, 'param_keys') else None,
                    bounds=self.bounds if hasattr(self, 'bounds') else None,
                    context="initial covariance build (dense solver)")
            self._alpha = jla.cho_solve((self._L, True), self._resid)

    def _build_logposterior(self):
        """Build the JIT-compiled log-posterior and its value-and-gradient.

        A single raw log-posterior is compiled exactly twice: once
        value-only (``log_posterior``, the traceable log-density handed to
        samplers) and once wrapped in ``jax.value_and_grad``
        (``value_and_grad_log_posterior``, the workhorse for optimizers).
        ``neg_log_posterior``, ``grad_log_posterior`` and
        ``grad_neg_log_posterior`` are thin Python wrappers that negate or
        slice those two compiled functions outside XLA, so they trigger no
        additional compilation (previously each was a separate XLA
        compilation, quadrupling warm-up time).
        """
        bounds = self.bounds
        x, y, yerr = self.x, self.y, self.yerr
        mean_val = self.mean_val
        n_h, n_l, lr = self.n_harmonics, self.n_lat, self.lat_range
        custom_prior = self._custom_log_prior
        fit_sn = self.fit_sigma_n
        qn, qw = self._quad_nodes, self._quad_weights
        to_phys = self._to_physical  # sampling theta → physical theta
        u_dt = self.uniform_dt       # Toeplitz fast path (None = disabled)
        # Gappy-uniform lag tables (None on strictly-uniform or irregular
        # grids); captured as compile-time constants in the closures.
        band_tab = (self._band_lag_table()
                    if self.matrix_solver == "cholesky_banded" else None)
        full_tab = (self._full_lag_table()
                    if self.matrix_solver != "cholesky_banded" else None)

        # Envelope-specific R_Gamma function (JAX-traceable, captured in closure)
        r_gamma_fn = self.spot_model.get_r_gamma_func()
        # Latitude weight function (JAX-traceable, or None for static weights)
        lat_wt_fn = self.spot_model.get_lat_weight_func()
        # Visibility coefficient function (JAX-traceable, or None for the
        # closed-form analytic c_n used by the default visibility function)
        cn_sq_fn = self.spot_model.get_cn_sq_func(self.n_harmonics)
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

            def _log_likelihood_raw(theta_arr):
                return _gp_log_likelihood_banded(
                    to_phys(theta_arr), x, y, yerr, mean_val,
                    n_h, n_l, lr, fit_sn, b,
                    n_kernel=n_kernel, r_gamma_func=r_gamma_fn,
                    quad_nodes=qn, quad_weights=qw,
                    edgeon_cn_sq=eo_cn,
                    lat_weight_func=lat_wt_fn,
                    cn_sq_func=cn_sq_fn,
                    uniform_dt=u_dt,
                    band_lag_table=band_tab)
        else:
            def _log_likelihood_raw(theta_arr):
                return _gp_log_likelihood(
                    to_phys(theta_arr), x, y, yerr, mean_val,
                    n_h, n_l, lr, fit_sn,
                    n_kernel=n_kernel, r_gamma_func=r_gamma_fn,
                    quad_nodes=qn, quad_weights=qw,
                    edgeon_cn_sq=eo_cn,
                    lat_weight_func=lat_wt_fn,
                    cn_sq_func=cn_sq_fn,
                    uniform_dt=u_dt,
                    lag_table=full_tab)

        def _log_prior_raw(theta_arr):
            return (custom_prior(theta_arr) if custom_prior is not None
                    else _default_log_prior(theta_arr, bounds))

        def _log_posterior_raw(theta_arr):
            return _log_likelihood_raw(theta_arr) + _log_prior_raw(theta_arr)

        self.log_posterior = jax.jit(_log_posterior_raw)
        self.value_and_grad_log_posterior = jax.jit(
            jax.value_and_grad(_log_posterior_raw))

        vag = self.value_and_grad_log_posterior

        def neg_log_posterior(theta_arr):
            return -self.log_posterior(theta_arr)

        def grad_log_posterior(theta_arr):
            return vag(theta_arr)[1]

        def grad_neg_log_posterior(theta_arr):
            return -vag(theta_arr)[1]

        self.neg_log_posterior = neg_log_posterior
        self.grad_log_posterior = grad_log_posterior
        self.grad_neg_log_posterior = grad_neg_log_posterior

        # Separate log-prior and log-likelihood for SMC tempering.
        # Compiled lazily on first call (only run_smc uses them).
        self.log_prior_fn = jax.jit(_log_prior_raw)
        self.log_likelihood_fn = jax.jit(_log_likelihood_raw)

    # =================================================================
    # JAX compilation warmup
    # =================================================================

    def build_jax(self, recompute=True):
        """
        Pre-compile and warm up the solver's JAX JIT functions.

        Exactly two XLA compilations are triggered: the value-only
        ``log_posterior`` (the log-density handed to samplers) and
        ``value_and_grad_log_posterior`` (used by ``fit_map`` and the
        gradient accessors).  The remaining public functions
        (``neg_log_posterior``, ``grad_log_posterior``,
        ``grad_neg_log_posterior``) are Python wrappers around these two
        and need no compilation of their own.

        Call ``build_jax()`` once after constructing the solver to pay
        the compilation cost upfront instead of inside the first fit or
        MCMC step.

        Returns
        -------
        self : GPSolver
            Returns ``self`` so the call can be chained:
            ``gp = GPSolver(...).build_jax()``.
        """
        import time

        theta0 = self.theta0

        logger.info("Compiling JAX functions (one-time cost)...")
        t0 = time.time()
        try:
            jax.block_until_ready(self.log_posterior(theta0))
            jax.block_until_ready(self.value_and_grad_log_posterior(theta0))
        except Exception as e:
            raise_cholesky_error(
                e, theta=theta0, param_keys=self.param_keys,
                bounds=self.bounds,
                context="JIT compilation of log-posterior")
        elapsed = time.time() - t0
        logger.info("JAX GP solver compiled in %.2fs", elapsed)

        if recompute:
            t0 = time.time()
            jax.block_until_ready(self.log_posterior(theta0))
            jax.block_until_ready(self.value_and_grad_log_posterior(theta0))
            logger.info("JAX GP solver recompute in %.2fs", time.time() - t0)

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

    def sample_lightcurves(self, theta=None, xpred=None, n_samples=5,
                           n_points=2000, source="prior", rng=None):
        """
        Sample lightcurves from the GP prior or posterior.

        Parameters
        ----------
        theta : dict or array_like, optional
            Kernel parameters.  Accepts a physical dict with keys from
            ``param_keys``, a sampling-space dict with ``log_``-prefixed
            keys, or a flat array matching ``param_keys``.
            If None, uses the current internal hyperparameters.
        xpred : array_like, optional
            Times at which to evaluate the samples.  If None, uses
            ``n_points`` evenly spaced times spanning the data baseline.
        n_samples : int
            Number of lightcurve samples to draw (default 5).
        n_points : int
            Number of prediction points when ``xpred`` is None (default
            2000).
        source : {'prior', 'posterior'}
            Whether to sample from the GP prior or posterior (default
            'prior').
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        xpred : ndarray, shape (M,)
            Prediction times.
        samples : ndarray, shape (n_samples, M)
            Sampled lightcurves.
        """
        if theta is not None:
            phys = {}
            for k, v in (theta.items() if isinstance(theta, dict)
                         else zip(self.spot_model.param_keys, theta)):
                if isinstance(k, str) and k.startswith("log_"):
                    phys[k[4:]] = 10.0 ** float(v)
                else:
                    phys[k] = float(v)
            self.update_hparam(phys)

        if xpred is None:
            xpred = np.linspace(float(self.x[0]), float(self.x[-1]),
                                n_points)
        else:
            xpred = np.asarray(xpred)

        if source == "prior":
            samples = self.sample_prior(xpred, n_samples=n_samples, rng=rng)
        elif source == "posterior":
            samples = self.sample_posterior(xpred, n_samples=n_samples,
                                           rng=rng)
        else:
            raise ValueError(f"source must be 'prior' or 'posterior', "
                             f"got {source!r}")

        return xpred, samples


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
            y_full = getattr(self.data, '_y_full', self.data.y)
            var = np.var(y_full)
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
                self._band_lag_table_cache = None
                self._check_bandwidth_efficiency()
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
    # HDF5 save / load
    # =================================================================

    def save(self, path):
        """Save solver state to an HDF5 file.

        Writes data, model configuration, bounds, and any completed
        fit results (MAP, ACF, mass matrix).  Can be called repeatedly
        — each call overwrites only the groups whose data changed.

        Parameters
        ----------
        path : str
            File path (should end in ``.h5`` or ``.hdf5``).
        """
        from .io import save_gp
        save_gp(path, self)

    @classmethod
    def load(cls, path):
        """Reconstruct a GPSolver from an HDF5 file.

        The returned solver has fit results restored but is NOT
        JIT-compiled.  Call ``.build_jax()`` before running fits
        or sampling.

        Parameters
        ----------
        path : str
            Path to an HDF5 file previously written by :meth:`save`.

        Returns
        -------
        GPSolver
        """
        from .io import load_gp
        return load_gp(path)

