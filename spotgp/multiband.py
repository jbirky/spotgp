"""
multiband.py — Multi-band GP solver with wavelength-dependent spot contrast.

Extends the single-band spotgp kernel to multi-band photometry via the
factorized chromatic kernel:

    K(τ; λ_i, λ_j) = c(λ_i) · c(λ_j) · K_geom(τ)

where c(λ) = 1 - B_λ(T_spot) / B_λ(T_phot) is the blackbody contrast
factor and K_geom(τ) is the wavelength-independent geometric kernel from
the single-band analysis.

The factorized structure means the multi-band covariance matrix is a
rank-1 Kronecker product in band space, preserving the banded temporal
structure and O(Nb²) scaling of the single-band solver.
"""
import os
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", os.environ.get("JAX_PLATFORMS", "cpu"))

import jax.numpy as jnp
import numpy as np

from .contrast import contrast_factor as _default_contrast_factor, contrast_matrix
from .params import resolve_hparam, BASE_REQUIRED_KEYS
from .spot_model import SpotEvolutionModel
from .visibility import (
    EdgeOnVisibilityFunction, _cn_general_jax, _gauss_legendre_grid,
)
from .analytic_kernel import AnalyticKernel, _kernel_eval
from .gp_solver import (
    _build_banded_kernel_jax, _default_log_prior,
    _gp_log_likelihood, GPSolver,
)
from .banded_cholesky import banded_cholesky_compact, banded_solve_compact

__all__ = ["MultiBandData", "MultiBandGPSolver", "SpotFaculaeGPSolver"]


class MultiBandData:
    """
    Container for multi-band photometric observations.

    Merges observations from multiple photometric bands into a single
    time-sorted array with per-observation band labels.  Each band is
    independently median-normalized before merging.

    Parameters
    ----------
    bands : dict
        ``{name: {"x": times, "y": flux, "yerr": errors, "wavelength": λ}}``
        where ``wavelength`` is the effective wavelength in Angstroms.
    normalize : bool
        If True, normalize each band to median flux = 1.
    """

    def __init__(self, bands, normalize=True):
        xs, ys, yerrs, idxs, wls, names = [], [], [], [], [], []

        for i, (name, b) in enumerate(bands.items()):
            x = np.asarray(b["x"], dtype=float)
            y = np.asarray(b["y"], dtype=float)
            yerr = np.asarray(b["yerr"], dtype=float)
            yerr = np.broadcast_to(yerr, x.shape).copy()
            wl = float(b["wavelength"])

            mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
            x, y, yerr = x[mask], y[mask], yerr[mask]

            if normalize and len(y) > 0:
                med = np.median(y)
                if med != 0:
                    yerr /= np.abs(med)
                    y /= med

            xs.append(x)
            ys.append(y)
            yerrs.append(yerr)
            idxs.append(np.full(len(x), i, dtype=int))
            wls.append(wl)
            names.append(name)

        x_all = np.concatenate(xs)
        order = np.argsort(x_all)

        self.x = x_all[order]
        self.y = np.concatenate(ys)[order]
        self.yerr = np.concatenate(yerrs)[order]
        self.band_indices = np.concatenate(idxs)[order]
        self.band_wavelengths = np.array(wls)
        self.band_names = names
        self.n_bands = len(bands)

    @property
    def N(self):
        return len(self.x)

    @property
    def baseline(self):
        return float(self.x[-1] - self.x[0])

    @property
    def median_dt(self):
        return float(np.median(np.diff(self.x)))


# =====================================================================
# Pure-functional multi-band likelihood (module-level for JAX tracing)
# =====================================================================

def _multiband_log_likelihood_banded(
        theta_full, x, y, yerr, mean_val,
        band_indices, band_wavelengths, T_phot,
        harmonics, n_lat, lat_range,
        fit_sigma_n, bandwidth, n_kernel,
        r_gamma_func=None,
        quad_nodes=None, quad_weights=None,
        edgeon_cn_sq=None,
        lat_weight_func=None,
        contrast_fn=None):
    """
    Multi-band GP log-likelihood using banded Cholesky.

    Builds the geometric kernel in compact banded storage, scales each
    entry by the per-observation contrast factors, adds noise, and solves.
    """
    if contrast_fn is None:
        contrast_fn = _default_contrast_factor
    N = x.shape[0]

    # Split theta: [kernel_params..., T_spot, (sigma_n)]
    theta_kernel = theta_full[:n_kernel]
    T_spot = theta_full[n_kernel]
    if fit_sigma_n:
        sigma_n = theta_full[n_kernel + 1]
    else:
        sigma_n = 0.0

    # Contrast factors per band → per observation
    c_bands = contrast_fn(band_wavelengths, T_spot, T_phot)
    c_obs = c_bands[band_indices]

    # Build geometric kernel in compact banded storage
    cb = _build_banded_kernel_jax(
        theta_kernel, x, bandwidth,
        harmonics, n_lat, lat_range,
        r_gamma_func=r_gamma_func,
        quad_nodes=quad_nodes, quad_weights=quad_weights,
        edgeon_cn_sq=edgeon_cn_sq,
        lat_weight_func=lat_weight_func)

    # Scale by contrast: cb[d, i] corresponds to K(x[i+d], x[i])
    b = bandwidth
    d_idx = jnp.arange(b + 1)[:, None]
    i_idx = jnp.arange(N)[None, :]
    j_idx = jnp.minimum(i_idx + d_idx, N - 1)
    contrast_scale = c_obs[j_idx] * c_obs[i_idx]
    cb = cb * contrast_scale

    # Add noise to diagonal (row 0 of compact storage)
    noise_var = yerr ** 2 + sigma_n ** 2
    cb = cb.at[0, :].add(noise_var + 1e-8)

    # Cholesky factorize and solve
    Lc = banded_cholesky_compact(cb, bandwidth)
    resid = y - mean_val
    alpha = banded_solve_compact(Lc, resid, bandwidth)

    data_fit = resid @ alpha
    log_det = 2.0 * jnp.sum(jnp.log(Lc[0, :]))

    return -0.5 * (data_fit + log_det + N * jnp.log(2 * jnp.pi))


def _multiband_log_likelihood_full(
        theta_full, x, y, yerr, mean_val,
        band_indices, band_wavelengths, T_phot,
        harmonics, n_lat, lat_range,
        fit_sigma_n, n_kernel,
        r_gamma_func=None,
        quad_nodes=None, quad_weights=None,
        edgeon_cn_sq=None,
        lat_weight_func=None,
        contrast_fn=None):
    """
    Multi-band GP log-likelihood using full Cholesky (for small datasets).
    """
    if contrast_fn is None:
        contrast_fn = _default_contrast_factor
    N = x.shape[0]

    theta_kernel = theta_full[:n_kernel]
    T_spot = theta_full[n_kernel]
    if fit_sigma_n:
        sigma_n = theta_full[n_kernel + 1]
    else:
        sigma_n = 0.0

    c_bands = contrast_fn(band_wavelengths, T_spot, T_phot)
    c_obs = c_bands[band_indices]

    # Upper-triangular kernel evaluation
    row_idx, col_idx = jnp.triu_indices(N)
    lag_upper = jnp.abs(x[row_idx] - x[col_idx])
    K_upper = _kernel_eval(
        theta_kernel, lag_upper,
        harmonics, n_lat, lat_range,
        quad_nodes=quad_nodes, quad_weights=quad_weights,
        r_gamma_func=r_gamma_func,
        edgeon_cn_sq=edgeon_cn_sq,
        lat_weight_func=lat_weight_func)

    # Scale by contrast
    K_upper = K_upper * c_obs[row_idx] * c_obs[col_idx]

    # Reconstruct symmetric matrix
    K = jnp.zeros((N, N))
    K = K.at[row_idx, col_idx].set(K_upper)
    K = K + K.T - jnp.diag(jnp.diag(K))

    noise_var = yerr ** 2 + sigma_n ** 2
    K_noise = K + jnp.diag(noise_var) + 1e-8 * jnp.eye(N)

    L = jax.scipy.linalg.cholesky(K_noise, lower=True)
    resid = y - mean_val
    alpha = jax.scipy.linalg.cho_solve((L, True), resid)

    data_fit = resid @ alpha
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    return -0.5 * (data_fit + log_det + N * jnp.log(2 * jnp.pi))


# =====================================================================
# MultiBandGPSolver
# =====================================================================

class MultiBandGPSolver:
    """
    Multi-band GP solver for stellar photometry with chromatic spot contrast.

    Uses the factorized kernel K(τ; λ_i, λ_j) = c(λ_i)·c(λ_j)·K_geom(τ)
    where the contrast c(λ) = 1 - B_λ(T_spot)/B_λ(T_phot) comes from the
    blackbody ratio.  Adds one new free parameter (T_spot) to the standard
    spotgp kernel.

    Parameters
    ----------
    data : MultiBandData
        Multi-band observations.
    model_or_hparam : SpotEvolutionModel or dict
        Spot model for the geometric kernel.  Must NOT include T_spot
        (that is passed separately via ``T_spot_init``).
    T_phot : float
        Photospheric effective temperature [K].  Assumed known.
    T_spot_init : float
        Initial spot temperature [K] for optimization/sampling.
    fit_sigma_n : bool
        Whether to include white noise sigma_n as a free parameter.
    bounds : dict or None
        Parameter bounds.  Keys are the same as single-band GPSolver,
        plus ``"T_spot"`` (and optionally ``"log_T_spot"`` for log-space).
    log_prior : callable or None
        Custom log-prior f(theta_arr) -> scalar.
    matrix_solver : {"cholesky_banded", "cholesky_full"}
        Linear algebra backend.
    bandwidth : int or None
        Banded solver bandwidth (auto-computed from bounds if None).
    kernel_kwargs : dict
        Extra kwargs forwarded to AnalyticKernel.
    """

    DEFAULT_BOUNDS = {
        **GPSolver.DEFAULT_BOUNDS,
        "T_spot": (2500.0, 6500.0),
    }

    def __init__(self, data, model_or_hparam, T_phot, T_spot_init=None,
                 fit_sigma_n=False, bounds=None, log_prior=None,
                 matrix_solver="cholesky_banded", bandwidth=None,
                 contrast_model=None,
                 **kernel_kwargs):

        if not isinstance(data, MultiBandData):
            raise TypeError("data must be a MultiBandData instance")

        self.data = data
        self.T_phot = float(T_phot)
        self.contrast_model = contrast_model
        if contrast_model is not None:
            self._contrast_fn = contrast_model.contrast_factor
        else:
            self._contrast_fn = _default_contrast_factor
        # Deferred: _jit_contrast_fn set after band_wavelengths is known

        # JAX arrays for the merged data
        self.x = jnp.asarray(data.x, dtype=jnp.float64)
        self.y = jnp.asarray(data.y, dtype=jnp.float64)
        self.yerr = jnp.asarray(data.yerr, dtype=jnp.float64)
        self.N = data.N
        self._band_indices = jnp.asarray(data.band_indices, dtype=jnp.int32)
        self._band_wavelengths = jnp.asarray(
            data.band_wavelengths, dtype=jnp.float64)

        # Parse spot model (standard params, no T_spot)
        if isinstance(model_or_hparam, SpotEvolutionModel):
            self.spot_model = model_or_hparam
            self.hparam = model_or_hparam.to_hparam()
        else:
            hparam = dict(model_or_hparam)
            hparam.pop("T_spot", None)
            resolve_hparam(hparam)
            self.hparam = hparam
            self.spot_model = SpotEvolutionModel.from_hparam(hparam)

        # T_spot initial value
        if T_spot_init is None:
            T_spot_init = self.T_phot - 500.0
        self.T_spot_init = float(T_spot_init)

        # Mean (matches GPSolver convention)
        self.mean_val = float(jnp.mean(self.y))

        # Matrix solver
        self.matrix_solver = matrix_solver
        self.fit_sigma_n = fit_sigma_n

        # Build kernel (for config: harmonics, n_lat, lat_range, etc.)
        self.kernel = AnalyticKernel(self.spot_model, **kernel_kwargs)

        # Parameter keys: standard kernel + T_spot + optional sigma_n
        _model_keys = self.spot_model.param_keys
        self._n_kernel = len(_model_keys)
        _base_keys = _model_keys + ("T_spot",)
        if fit_sigma_n:
            _base_keys = _base_keys + ("sigma_n",)

        # Log-space parameter detection
        self._log_param_map = {}
        if isinstance(bounds, dict):
            for k in bounds:
                if k.startswith("log_"):
                    self._log_param_map[k] = k[4:]

        _phys_to_log = {v: k for k, v in self._log_param_map.items()}
        self.param_keys = tuple(
            _phys_to_log.get(k, k) for k in _base_keys)
        self.n_params = len(self.param_keys)

        # Parse bounds
        if bounds is None:
            self.bounds = jnp.array(
                [self.DEFAULT_BOUNDS[k] for k in _base_keys],
                dtype=jnp.float64)
        elif isinstance(bounds, dict):
            self.bounds = jnp.array(
                [bounds.get(_pk, self.DEFAULT_BOUNDS.get(_bk, (-1e10, 1e10)))
                 for _pk, _bk in zip(self.param_keys, _base_keys)],
                dtype=jnp.float64)
        else:
            self.bounds = jnp.asarray(bounds, dtype=jnp.float64)

        # Kernel config
        self.harmonics = self.kernel.harmonics
        self.n_lat = self.kernel.n_lat
        if self.spot_model.latitude_distribution.param_dict:
            self.lat_range = (-np.pi / 2, np.pi / 2)
        else:
            self.lat_range = self.kernel.lat_range

        # Quadrature nodes
        if self.kernel.quadrature == "gauss-legendre":
            if self.spot_model.latitude_distribution.param_dict:
                gl_nodes, gl_weights = _gauss_legendre_grid(
                    self.n_lat, -np.pi / 2, np.pi / 2)
                _norm = float(jnp.sum(gl_weights))
                self._quad_nodes = gl_nodes
                self._quad_weights = gl_weights / _norm
            else:
                raw_w = self.kernel._quad_weights
                _norm = float(jnp.sum(raw_w))
                self._quad_nodes = self.kernel._quad_nodes
                self._quad_weights = raw_w / _norm
        else:
            self._quad_nodes = None
            self._quad_weights = None

        # Bandwidth
        if matrix_solver == "cholesky_banded":
            if bandwidth is not None:
                self.bandwidth = min(int(bandwidth), self.N - 1)
            else:
                self.bandwidth = self._compute_bandwidth()
            _n_banded = (self.bandwidth + 1) * self.N
            _n_full = self.N * self.N
            _sparsity = 100.0 * (1.0 - _n_banded / _n_full) if _n_full > 0 else 0
            print(f"MultiBand banded Cholesky: bandwidth={self.bandwidth}, "
                  f"N={self.N}, n_bands={data.n_bands}, "
                  f"sparsity={_sparsity:.1f}%")

        # Build theta0
        _phys_theta0 = dict(
            zip(self.spot_model.param_keys, self.spot_model.theta0))
        _phys_theta0["T_spot"] = self.T_spot_init
        if fit_sigma_n:
            _phys_theta0["sigma_n"] = float(
                self.hparam.get("sigma_n",
                                self.DEFAULT_BOUNDS["sigma_n"][0]))
        self.theta0 = jnp.array([
            np.log10(float(_phys_theta0.get(self._log_param_map[k], 0.0)))
            if k in self._log_param_map
            else float(_phys_theta0.get(k, 0.0))
            for k in self.param_keys
        ], dtype=jnp.float64)

        # Prior
        self._custom_log_prior = log_prior

        # Build log-param transform
        self._build_transform()

        # Build JIT-compiled log-posterior
        self._build_logposterior()

        # Storage
        self.map_result = None

    @property
    def n_harmonics(self):
        """Largest harmonic order — the int summary of :attr:`harmonics`."""
        return max(self.harmonics)

    def _compute_bandwidth(self):
        if self.N < 2:
            return self.N
        diffs = np.diff(np.asarray(self.x))
        dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
        support = self.spot_model.bandwidth_support(
            self.spot_model.param_keys,
            jnp.array([self.DEFAULT_BOUNDS.get(k, (0, 10))
                        for k in self.spot_model.param_keys]))
        b = int(np.ceil(support / dt))
        return min(b, self.N - 1)

    def _build_transform(self):
        if not self._log_param_map:
            self._to_physical = lambda x: x
            return
        keys = list(self.param_keys)
        log_indices = jnp.array(
            [keys.index(k) for k in self._log_param_map], dtype=jnp.int32)

        @jax.jit
        def to_physical(theta_arr):
            return theta_arr.at[log_indices].set(
                10.0 ** theta_arr[log_indices])
        self._to_physical = to_physical

    def _build_logposterior(self):
        """Build JIT-compiled log-posterior with multi-band likelihood."""
        bounds = self.bounds
        x, y, yerr = self.x, self.y, self.yerr
        mean_val = self.mean_val
        n_h, n_l, lr = self.harmonics, self.n_lat, self.lat_range
        custom_prior = self._custom_log_prior
        fit_sn = self.fit_sigma_n
        qn, qw = self._quad_nodes, self._quad_weights
        to_phys = self._to_physical
        n_kernel = self._n_kernel
        bi = self._band_indices
        bw = self._band_wavelengths
        T_phot = self.T_phot
        # For JIT: use make_contrast_fn (eagerly resolved band indices)
        if self.contrast_model is not None:
            cfn = self.contrast_model.make_contrast_fn(
                np.asarray(self.data.band_wavelengths))
        else:
            cfn = _default_contrast_factor

        r_gamma_fn = self.spot_model.get_r_gamma_func()
        lat_wt_fn = self.spot_model.get_lat_weight_func()

        if isinstance(self.spot_model.visibility, EdgeOnVisibilityFunction):
            eo_cn = jnp.array(
                self.spot_model.visibility.cn_squared(0.0, self.harmonics))
        else:
            eo_cn = None

        if self.matrix_solver == "cholesky_banded":
            b = self.bandwidth

            @jax.jit
            def log_posterior(theta_arr):
                theta_phys = to_phys(theta_arr)
                ll = _multiband_log_likelihood_banded(
                    theta_phys, x, y, yerr, mean_val,
                    bi, bw, T_phot,
                    n_h, n_l, lr, fit_sn, b, n_kernel,
                    r_gamma_func=r_gamma_fn,
                    quad_nodes=qn, quad_weights=qw,
                    edgeon_cn_sq=eo_cn,
                    lat_weight_func=lat_wt_fn,
                    contrast_fn=cfn)
                lp = (custom_prior(theta_arr) if custom_prior is not None
                      else _default_log_prior(theta_arr, bounds))
                return ll + lp
        else:
            @jax.jit
            def log_posterior(theta_arr):
                theta_phys = to_phys(theta_arr)
                ll = _multiband_log_likelihood_full(
                    theta_phys, x, y, yerr, mean_val,
                    bi, bw, T_phot,
                    n_h, n_l, lr, fit_sn, n_kernel,
                    r_gamma_func=r_gamma_fn,
                    quad_nodes=qn, quad_weights=qw,
                    edgeon_cn_sq=eo_cn,
                    lat_weight_func=lat_wt_fn,
                    contrast_fn=cfn)
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

        # Separate prior/likelihood for SMC tempering
        @jax.jit
        def _log_prior_fn(theta_arr):
            return (custom_prior(theta_arr) if custom_prior is not None
                    else _default_log_prior(theta_arr, bounds))

        if self.matrix_solver == "cholesky_banded":
            @jax.jit
            def _log_likelihood_fn(theta_arr):
                return _multiband_log_likelihood_banded(
                    to_phys(theta_arr), x, y, yerr, mean_val,
                    bi, bw, T_phot,
                    n_h, n_l, lr, fit_sn, b, n_kernel,
                    r_gamma_func=r_gamma_fn,
                    quad_nodes=qn, quad_weights=qw,
                    edgeon_cn_sq=eo_cn,
                    lat_weight_func=lat_wt_fn,
                    contrast_fn=cfn)
        else:
            @jax.jit
            def _log_likelihood_fn(theta_arr):
                return _multiband_log_likelihood_full(
                    to_phys(theta_arr), x, y, yerr, mean_val,
                    bi, bw, T_phot,
                    n_h, n_l, lr, fit_sn, n_kernel,
                    r_gamma_func=r_gamma_fn,
                    quad_nodes=qn, quad_weights=qw,
                    edgeon_cn_sq=eo_cn,
                    lat_weight_func=lat_wt_fn,
                    contrast_fn=cfn)

        self.log_prior_fn = _log_prior_fn
        self.log_likelihood_fn = _log_likelihood_fn

    def build_jax(self):
        """Pre-compile and warm up JIT functions."""
        _ = self.log_posterior(self.theta0).block_until_ready()
        _ = self.grad_log_posterior(self.theta0).block_until_ready()
        return self

    def log_likelihood_at(self, theta_arr):
        """Evaluate log-likelihood at given parameters."""
        return float(self.log_likelihood_fn(jnp.asarray(theta_arr)))

    def predict(self, xpred, theta=None, band_wavelength=None):
        """
        Predictive mean and variance at new times.

        Parameters
        ----------
        xpred : array_like, shape (M,)
            Prediction times.
        theta : dict or array_like, optional
            Parameters to use. If None, uses theta0.
        band_wavelength : float, optional
            Wavelength [Angstroms] of the prediction band.
            If None, returns the geometric (contrast-free) prediction.

        Returns
        -------
        mu_pred : ndarray, shape (M,)
        var_pred : ndarray, shape (M,)
        """
        xpred = jnp.asarray(xpred, dtype=jnp.float64)

        if theta is not None:
            if isinstance(theta, dict):
                theta_arr = jnp.array([
                    float(theta.get(k, 0.0)) for k in self.param_keys])
            else:
                theta_arr = jnp.asarray(theta)
            theta_phys = self._to_physical(theta_arr)
        else:
            theta_phys = self._to_physical(self.theta0)

        theta_kernel = theta_phys[:self._n_kernel]
        T_spot = float(theta_phys[self._n_kernel])

        # Contrast factors for training data
        c_bands = self._contrast_fn(self._band_wavelengths, T_spot, self.T_phot)
        c_obs = c_bands[self._band_indices]

        # Prediction band contrast
        if band_wavelength is not None:
            c_pred = float(self._contrast_fn(
                jnp.array(band_wavelength), T_spot, self.T_phot))
        else:
            c_pred = 1.0

        # Build training covariance (full matrix for prediction)
        N = self.N
        row_idx, col_idx = jnp.triu_indices(N)
        lag_train = jnp.abs(self.x[row_idx] - self.x[col_idx])
        K_upper = _kernel_eval(
            theta_kernel, lag_train,
            self.harmonics, self.n_lat, self.lat_range,
            quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
            r_gamma_func=self.spot_model.get_r_gamma_func())
        K_upper = K_upper * c_obs[row_idx] * c_obs[col_idx]

        K = jnp.zeros((N, N))
        K = K.at[row_idx, col_idx].set(K_upper)
        K = K + K.T - jnp.diag(jnp.diag(K))

        sigma_n = 0.0
        if self.fit_sigma_n:
            sigma_n = float(theta_phys[self._n_kernel + 1])
        noise_var = self.yerr ** 2 + sigma_n ** 2
        K_noise = K + jnp.diag(noise_var) + 1e-8 * jnp.eye(N)

        L = jax.scipy.linalg.cholesky(K_noise, lower=True)
        resid = self.y - self.mean_val
        alpha = jax.scipy.linalg.cho_solve((L, True), resid)

        # Cross-covariance: K(xpred, x_train) with prediction band contrast
        lag_cross = jnp.abs(xpred[:, None] - self.x[None, :])
        Ks_geom = _kernel_eval(
            theta_kernel, lag_cross.ravel(),
            self.harmonics, self.n_lat, self.lat_range,
            quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
            r_gamma_func=self.spot_model.get_r_gamma_func()
        ).reshape(lag_cross.shape)
        Ks = c_pred * c_obs[None, :] * Ks_geom

        mu_pred = self.mean_val + Ks @ alpha

        V = jax.scipy.linalg.cho_solve((L, True), Ks.T)
        k0 = float(_kernel_eval(
            theta_kernel, jnp.zeros(1),
            self.harmonics, self.n_lat, self.lat_range,
            quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
            r_gamma_func=self.spot_model.get_r_gamma_func())[0])
        var_pred = c_pred ** 2 * k0 - jnp.einsum('ij,ji->i', Ks, V)

        return np.asarray(mu_pred), np.asarray(var_pred)

    def amplitude_ratio(self, lam1, lam2, theta=None):
        """
        Predicted variability amplitude ratio between two bands.

        Returns σ(λ1) / σ(λ2) = c(λ1) / c(λ2), which depends only on
        T_spot / T_phot.

        Parameters
        ----------
        lam1, lam2 : float
            Wavelengths in Angstroms.
        theta : dict or array_like, optional
            Parameters (needs T_spot).  If None, uses T_spot_init.
        """
        if theta is not None:
            if isinstance(theta, dict):
                T_spot = theta.get("T_spot", self.T_spot_init)
            else:
                theta_arr = jnp.asarray(theta)
                T_spot = float(self._to_physical(theta_arr)[self._n_kernel])
        else:
            T_spot = self.T_spot_init
        c1 = self._contrast_fn(jnp.array(lam1), T_spot, self.T_phot)
        c2 = self._contrast_fn(jnp.array(lam2), T_spot, self.T_phot)
        return float(c1 / c2)

    def plot_pgm(self, **kwargs):
        """Plot the probabilistic graphical model for this multi-band GP.

        Parameters
        ----------
        **kwargs
            Forwarded to :meth:`PGModelVis.render` (``dpi``, ``node_scale``,
            ``font_size``).

        Returns
        -------
        fig : matplotlib.figure.Figure
        """
        from .pgm import PGModelVis
        return PGModelVis(self).render(**kwargs)


# =====================================================================
# Rank-2 (spots + faculae) likelihood functions
# =====================================================================

def _spotfac_log_likelihood_banded(
        theta_full, x, y, yerr, mean_val,
        band_indices, band_wavelengths, T_phot,
        harmonics, n_lat, lat_range,
        fit_sigma_n, bandwidth, n_kernel,
        r_gamma_func=None,
        quad_nodes=None, quad_weights=None,
        edgeon_cn_sq=None,
        lat_weight_func=None,
        contrast_fn=None):
    """
    Spots + faculae GP log-likelihood using banded Cholesky.

    Rank-2 contrast matrix:
        C(lambda_i, lambda_j) = c_s(lambda_i)*c_s(lambda_j)
                              + w_fac * c_f(lambda_i)*c_f(lambda_j)

    theta_full layout: [kernel_params..., T_spot, T_fac, w_fac, (sigma_n)]
    """
    if contrast_fn is None:
        contrast_fn = _default_contrast_factor
    N = x.shape[0]

    theta_kernel = theta_full[:n_kernel]
    T_spot = theta_full[n_kernel]
    T_fac = theta_full[n_kernel + 1]
    w_fac = theta_full[n_kernel + 2]
    if fit_sigma_n:
        sigma_n = theta_full[n_kernel + 3]
    else:
        sigma_n = 0.0

    c_spot_bands = contrast_fn(band_wavelengths, T_spot, T_phot)
    c_fac_bands = contrast_fn(band_wavelengths, T_fac, T_phot)
    c_spot_obs = c_spot_bands[band_indices]
    c_fac_obs = c_fac_bands[band_indices]

    cb = _build_banded_kernel_jax(
        theta_kernel, x, bandwidth,
        harmonics, n_lat, lat_range,
        r_gamma_func=r_gamma_func,
        quad_nodes=quad_nodes, quad_weights=quad_weights,
        edgeon_cn_sq=edgeon_cn_sq,
        lat_weight_func=lat_weight_func)

    b = bandwidth
    d_idx = jnp.arange(b + 1)[:, None]
    i_idx = jnp.arange(N)[None, :]
    j_idx = jnp.minimum(i_idx + d_idx, N - 1)
    contrast_scale = (c_spot_obs[j_idx] * c_spot_obs[i_idx]
                      + w_fac * c_fac_obs[j_idx] * c_fac_obs[i_idx])
    cb = cb * contrast_scale

    noise_var = yerr ** 2 + sigma_n ** 2
    cb = cb.at[0, :].add(noise_var + 1e-8)

    Lc = banded_cholesky_compact(cb, bandwidth)
    resid = y - mean_val
    alpha = banded_solve_compact(Lc, resid, bandwidth)

    data_fit = resid @ alpha
    log_det = 2.0 * jnp.sum(jnp.log(Lc[0, :]))

    return -0.5 * (data_fit + log_det + N * jnp.log(2 * jnp.pi))


def _spotfac_log_likelihood_full(
        theta_full, x, y, yerr, mean_val,
        band_indices, band_wavelengths, T_phot,
        harmonics, n_lat, lat_range,
        fit_sigma_n, n_kernel,
        r_gamma_func=None,
        quad_nodes=None, quad_weights=None,
        edgeon_cn_sq=None,
        lat_weight_func=None,
        contrast_fn=None):
    """
    Spots + faculae GP log-likelihood using full Cholesky.

    theta_full layout: [kernel_params..., T_spot, T_fac, w_fac, (sigma_n)]
    """
    if contrast_fn is None:
        contrast_fn = _default_contrast_factor
    N = x.shape[0]

    theta_kernel = theta_full[:n_kernel]
    T_spot = theta_full[n_kernel]
    T_fac = theta_full[n_kernel + 1]
    w_fac = theta_full[n_kernel + 2]
    if fit_sigma_n:
        sigma_n = theta_full[n_kernel + 3]
    else:
        sigma_n = 0.0

    c_spot_bands = contrast_fn(band_wavelengths, T_spot, T_phot)
    c_fac_bands = contrast_fn(band_wavelengths, T_fac, T_phot)
    c_spot_obs = c_spot_bands[band_indices]
    c_fac_obs = c_fac_bands[band_indices]

    row_idx, col_idx = jnp.triu_indices(N)
    lag_upper = jnp.abs(x[row_idx] - x[col_idx])
    K_upper = _kernel_eval(
        theta_kernel, lag_upper,
        harmonics, n_lat, lat_range,
        quad_nodes=quad_nodes, quad_weights=quad_weights,
        r_gamma_func=r_gamma_func,
        edgeon_cn_sq=edgeon_cn_sq,
        lat_weight_func=lat_weight_func)

    K_upper = K_upper * (c_spot_obs[row_idx] * c_spot_obs[col_idx]
                         + w_fac * c_fac_obs[row_idx] * c_fac_obs[col_idx])

    K = jnp.zeros((N, N))
    K = K.at[row_idx, col_idx].set(K_upper)
    K = K + K.T - jnp.diag(jnp.diag(K))

    noise_var = yerr ** 2 + sigma_n ** 2
    K_noise = K + jnp.diag(noise_var) + 1e-8 * jnp.eye(N)

    L = jax.scipy.linalg.cholesky(K_noise, lower=True)
    resid = y - mean_val
    alpha = jax.scipy.linalg.cho_solve((L, True), resid)

    data_fit = resid @ alpha
    log_det = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))

    return -0.5 * (data_fit + log_det + N * jnp.log(2 * jnp.pi))


# =====================================================================
# SpotFaculaeGPSolver
# =====================================================================

class SpotFaculaeGPSolver:
    """
    Multi-band GP solver with rank-2 contrast matrix for spots and faculae.

    Extends the chromatic kernel to include both cool spots and hot faculae:

        K(tau; lambda_i, lambda_j) = [c_s(lambda_i)*c_s(lambda_j)
            + w_fac * c_f(lambda_i)*c_f(lambda_j)] * K_geom(tau)

    where c_s = 1 - B_lambda(T_spot)/B_lambda(T_phot) > 0 (spots) and
    c_f = 1 - B_lambda(T_fac)/B_lambda(T_phot) < 0 (faculae). The spot
    weight is implicitly 1 (absorbed into sigma_k); w_fac is the relative
    facular weight.

    New parameters beyond MultiBandGPSolver: T_fac, w_fac.

    Parameters
    ----------
    data : MultiBandData
        Multi-band observations.
    model_or_hparam : SpotEvolutionModel or dict
        Spot model for the geometric kernel.
    T_phot : float
        Photospheric effective temperature [K].
    T_spot_init : float
        Initial spot temperature [K].
    T_fac_init : float
        Initial facular temperature [K].  Default: T_phot + 300.
    w_fac_init : float
        Initial facular weight.  Default: 0.5.
    fit_sigma_n : bool
        Whether to include white noise sigma_n as a free parameter.
    bounds : dict or None
        Parameter bounds.  Standard keys plus ``"T_spot"``, ``"T_fac"``,
        ``"w_fac"`` (and log-space variants).
    log_prior : callable or None
        Custom log-prior f(theta_arr) -> scalar.
    matrix_solver : {"cholesky_banded", "cholesky_full"}
        Linear algebra backend.
    bandwidth : int or None
        Banded solver bandwidth (auto-computed if None).
    contrast_model : object or None
        Custom contrast model with ``.contrast_factor(lam, T, T_phot)``
        and ``.make_contrast_fn(band_wavelengths)`` methods.
    """

    DEFAULT_BOUNDS = {
        **GPSolver.DEFAULT_BOUNDS,
        "T_spot": (2500.0, 6500.0),
        "T_fac":  (4000.0, 10000.0),
        "w_fac":  (0.0, 10.0),
    }

    def __init__(self, data, model_or_hparam, T_phot, T_spot_init=None,
                 T_fac_init=None, w_fac_init=0.5,
                 fit_sigma_n=False, bounds=None, log_prior=None,
                 matrix_solver="cholesky_banded", bandwidth=None,
                 contrast_model=None,
                 **kernel_kwargs):

        if not isinstance(data, MultiBandData):
            raise TypeError("data must be a MultiBandData instance")

        self.data = data
        self.T_phot = float(T_phot)
        self.contrast_model = contrast_model
        if contrast_model is not None:
            self._contrast_fn = contrast_model.contrast_factor
        else:
            self._contrast_fn = _default_contrast_factor

        self.x = jnp.asarray(data.x, dtype=jnp.float64)
        self.y = jnp.asarray(data.y, dtype=jnp.float64)
        self.yerr = jnp.asarray(data.yerr, dtype=jnp.float64)
        self.N = data.N
        self._band_indices = jnp.asarray(data.band_indices, dtype=jnp.int32)
        self._band_wavelengths = jnp.asarray(
            data.band_wavelengths, dtype=jnp.float64)

        if isinstance(model_or_hparam, SpotEvolutionModel):
            self.spot_model = model_or_hparam
            self.hparam = model_or_hparam.to_hparam()
        else:
            hparam = dict(model_or_hparam)
            for k in ("T_spot", "T_fac", "w_fac"):
                hparam.pop(k, None)
            resolve_hparam(hparam)
            self.hparam = hparam
            self.spot_model = SpotEvolutionModel.from_hparam(hparam)

        if T_spot_init is None:
            T_spot_init = self.T_phot - 500.0
        self.T_spot_init = float(T_spot_init)

        if T_fac_init is None:
            T_fac_init = self.T_phot + 300.0
        self.T_fac_init = float(T_fac_init)
        self.w_fac_init = float(w_fac_init)

        self.mean_val = float(jnp.mean(self.y))
        self.matrix_solver = matrix_solver
        self.fit_sigma_n = fit_sigma_n

        self.kernel = AnalyticKernel(self.spot_model, **kernel_kwargs)

        _model_keys = self.spot_model.param_keys
        self._n_kernel = len(_model_keys)
        _base_keys = _model_keys + ("T_spot", "T_fac", "w_fac")
        if fit_sigma_n:
            _base_keys = _base_keys + ("sigma_n",)

        self._log_param_map = {}
        if isinstance(bounds, dict):
            for k in bounds:
                if k.startswith("log_"):
                    self._log_param_map[k] = k[4:]

        _phys_to_log = {v: k for k, v in self._log_param_map.items()}
        self.param_keys = tuple(
            _phys_to_log.get(k, k) for k in _base_keys)
        self.n_params = len(self.param_keys)

        if bounds is None:
            self.bounds = jnp.array(
                [self.DEFAULT_BOUNDS[k] for k in _base_keys],
                dtype=jnp.float64)
        elif isinstance(bounds, dict):
            self.bounds = jnp.array(
                [bounds.get(_pk, self.DEFAULT_BOUNDS.get(_bk, (-1e10, 1e10)))
                 for _pk, _bk in zip(self.param_keys, _base_keys)],
                dtype=jnp.float64)
        else:
            self.bounds = jnp.asarray(bounds, dtype=jnp.float64)

        self.harmonics = self.kernel.harmonics
        self.n_lat = self.kernel.n_lat
        if self.spot_model.latitude_distribution.param_dict:
            self.lat_range = (-np.pi / 2, np.pi / 2)
        else:
            self.lat_range = self.kernel.lat_range

        if self.kernel.quadrature == "gauss-legendre":
            if self.spot_model.latitude_distribution.param_dict:
                gl_nodes, gl_weights = _gauss_legendre_grid(
                    self.n_lat, -np.pi / 2, np.pi / 2)
                _norm = float(jnp.sum(gl_weights))
                self._quad_nodes = gl_nodes
                self._quad_weights = gl_weights / _norm
            else:
                raw_w = self.kernel._quad_weights
                _norm = float(jnp.sum(raw_w))
                self._quad_nodes = self.kernel._quad_nodes
                self._quad_weights = raw_w / _norm
        else:
            self._quad_nodes = None
            self._quad_weights = None

        if matrix_solver == "cholesky_banded":
            if bandwidth is not None:
                self.bandwidth = min(int(bandwidth), self.N - 1)
            else:
                self.bandwidth = self._compute_bandwidth()
            _n_banded = (self.bandwidth + 1) * self.N
            _n_full = self.N * self.N
            _sparsity = 100.0 * (1.0 - _n_banded / _n_full) if _n_full > 0 else 0
            print(f"SpotFaculae banded Cholesky: bandwidth={self.bandwidth}, "
                  f"N={self.N}, n_bands={data.n_bands}, "
                  f"sparsity={_sparsity:.1f}%")

        _phys_theta0 = dict(
            zip(self.spot_model.param_keys, self.spot_model.theta0))
        _phys_theta0["T_spot"] = self.T_spot_init
        _phys_theta0["T_fac"] = self.T_fac_init
        _phys_theta0["w_fac"] = self.w_fac_init
        if fit_sigma_n:
            _phys_theta0["sigma_n"] = float(
                self.hparam.get("sigma_n",
                                self.DEFAULT_BOUNDS["sigma_n"][0]))
        self.theta0 = jnp.array([
            np.log10(float(_phys_theta0.get(self._log_param_map[k], 0.0)))
            if k in self._log_param_map
            else float(_phys_theta0.get(k, 0.0))
            for k in self.param_keys
        ], dtype=jnp.float64)

        self._custom_log_prior = log_prior
        self._build_transform()
        self._build_logposterior()
        self.map_result = None

    @property
    def n_harmonics(self):
        """Largest harmonic order — the int summary of :attr:`harmonics`."""
        return max(self.harmonics)

    def _compute_bandwidth(self):
        if self.N < 2:
            return self.N
        diffs = np.diff(np.asarray(self.x))
        dt = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 1.0
        support = self.spot_model.bandwidth_support(
            self.spot_model.param_keys,
            jnp.array([self.DEFAULT_BOUNDS.get(k, (0, 10))
                        for k in self.spot_model.param_keys]))
        b = int(np.ceil(support / dt))
        return min(b, self.N - 1)

    def _build_transform(self):
        if not self._log_param_map:
            self._to_physical = lambda x: x
            return
        keys = list(self.param_keys)
        log_indices = jnp.array(
            [keys.index(k) for k in self._log_param_map], dtype=jnp.int32)

        @jax.jit
        def to_physical(theta_arr):
            return theta_arr.at[log_indices].set(
                10.0 ** theta_arr[log_indices])
        self._to_physical = to_physical

    def _build_logposterior(self):
        bounds = self.bounds
        x, y, yerr = self.x, self.y, self.yerr
        mean_val = self.mean_val
        n_h, n_l, lr = self.harmonics, self.n_lat, self.lat_range
        custom_prior = self._custom_log_prior
        fit_sn = self.fit_sigma_n
        qn, qw = self._quad_nodes, self._quad_weights
        to_phys = self._to_physical
        n_kernel = self._n_kernel
        bi = self._band_indices
        bw = self._band_wavelengths
        T_phot = self.T_phot

        if self.contrast_model is not None:
            cfn = self.contrast_model.make_contrast_fn(
                np.asarray(self.data.band_wavelengths))
        else:
            cfn = _default_contrast_factor

        r_gamma_fn = self.spot_model.get_r_gamma_func()
        lat_wt_fn = self.spot_model.get_lat_weight_func()

        if isinstance(self.spot_model.visibility, EdgeOnVisibilityFunction):
            eo_cn = jnp.array(
                self.spot_model.visibility.cn_squared(0.0, self.harmonics))
        else:
            eo_cn = None

        if self.matrix_solver == "cholesky_banded":
            b = self.bandwidth

            @jax.jit
            def log_posterior(theta_arr):
                theta_phys = to_phys(theta_arr)
                ll = _spotfac_log_likelihood_banded(
                    theta_phys, x, y, yerr, mean_val,
                    bi, bw, T_phot,
                    n_h, n_l, lr, fit_sn, b, n_kernel,
                    r_gamma_func=r_gamma_fn,
                    quad_nodes=qn, quad_weights=qw,
                    edgeon_cn_sq=eo_cn,
                    lat_weight_func=lat_wt_fn,
                    contrast_fn=cfn)
                lp = (custom_prior(theta_arr) if custom_prior is not None
                      else _default_log_prior(theta_arr, bounds))
                return ll + lp
        else:
            @jax.jit
            def log_posterior(theta_arr):
                theta_phys = to_phys(theta_arr)
                ll = _spotfac_log_likelihood_full(
                    theta_phys, x, y, yerr, mean_val,
                    bi, bw, T_phot,
                    n_h, n_l, lr, fit_sn, n_kernel,
                    r_gamma_func=r_gamma_fn,
                    quad_nodes=qn, quad_weights=qw,
                    edgeon_cn_sq=eo_cn,
                    lat_weight_func=lat_wt_fn,
                    contrast_fn=cfn)
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

        @jax.jit
        def _log_prior_fn(theta_arr):
            return (custom_prior(theta_arr) if custom_prior is not None
                    else _default_log_prior(theta_arr, bounds))

        if self.matrix_solver == "cholesky_banded":
            @jax.jit
            def _log_likelihood_fn(theta_arr):
                return _spotfac_log_likelihood_banded(
                    to_phys(theta_arr), x, y, yerr, mean_val,
                    bi, bw, T_phot,
                    n_h, n_l, lr, fit_sn, b, n_kernel,
                    r_gamma_func=r_gamma_fn,
                    quad_nodes=qn, quad_weights=qw,
                    edgeon_cn_sq=eo_cn,
                    lat_weight_func=lat_wt_fn,
                    contrast_fn=cfn)
        else:
            @jax.jit
            def _log_likelihood_fn(theta_arr):
                return _spotfac_log_likelihood_full(
                    to_phys(theta_arr), x, y, yerr, mean_val,
                    bi, bw, T_phot,
                    n_h, n_l, lr, fit_sn, n_kernel,
                    r_gamma_func=r_gamma_fn,
                    quad_nodes=qn, quad_weights=qw,
                    edgeon_cn_sq=eo_cn,
                    lat_weight_func=lat_wt_fn,
                    contrast_fn=cfn)

        self.log_prior_fn = _log_prior_fn
        self.log_likelihood_fn = _log_likelihood_fn

    def build_jax(self):
        _ = self.log_posterior(self.theta0).block_until_ready()
        _ = self.grad_log_posterior(self.theta0).block_until_ready()
        return self

    def log_likelihood_at(self, theta_arr):
        return float(self.log_likelihood_fn(jnp.asarray(theta_arr)))

    def predict(self, xpred, theta=None, band_wavelength=None):
        """
        Predictive mean and variance at new times.

        Parameters
        ----------
        xpred : array_like, shape (M,)
            Prediction times.
        theta : dict or array_like, optional
            Parameters to use.  If None, uses theta0.
        band_wavelength : float, optional
            Wavelength [Angstroms] of the prediction band.
            If None, returns the geometric (contrast-free) prediction.

        Returns
        -------
        mu_pred : ndarray, shape (M,)
        var_pred : ndarray, shape (M,)
        """
        xpred = jnp.asarray(xpred, dtype=jnp.float64)

        if theta is not None:
            if isinstance(theta, dict):
                theta_arr = jnp.array([
                    float(theta.get(k, 0.0)) for k in self.param_keys])
            else:
                theta_arr = jnp.asarray(theta)
            theta_phys = self._to_physical(theta_arr)
        else:
            theta_phys = self._to_physical(self.theta0)

        theta_kernel = theta_phys[:self._n_kernel]
        T_spot = float(theta_phys[self._n_kernel])
        T_fac = float(theta_phys[self._n_kernel + 1])
        w_fac = float(theta_phys[self._n_kernel + 2])

        c_spot_bands = self._contrast_fn(
            self._band_wavelengths, T_spot, self.T_phot)
        c_fac_bands = self._contrast_fn(
            self._band_wavelengths, T_fac, self.T_phot)
        c_spot_obs = c_spot_bands[self._band_indices]
        c_fac_obs = c_fac_bands[self._band_indices]

        if band_wavelength is not None:
            lam_arr = jnp.array(band_wavelength)
            c_spot_pred = float(self._contrast_fn(lam_arr, T_spot, self.T_phot))
            c_fac_pred = float(self._contrast_fn(lam_arr, T_fac, self.T_phot))
        else:
            c_spot_pred = 1.0
            c_fac_pred = 0.0

        r_gamma_fn = self.spot_model.get_r_gamma_func()

        N = self.N
        row_idx, col_idx = jnp.triu_indices(N)
        lag_train = jnp.abs(self.x[row_idx] - self.x[col_idx])
        K_upper = _kernel_eval(
            theta_kernel, lag_train,
            self.harmonics, self.n_lat, self.lat_range,
            quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
            r_gamma_func=r_gamma_fn)
        K_upper = K_upper * (c_spot_obs[row_idx] * c_spot_obs[col_idx]
                             + w_fac * c_fac_obs[row_idx] * c_fac_obs[col_idx])

        K = jnp.zeros((N, N))
        K = K.at[row_idx, col_idx].set(K_upper)
        K = K + K.T - jnp.diag(jnp.diag(K))

        sigma_n = 0.0
        if self.fit_sigma_n:
            sigma_n = float(theta_phys[self._n_kernel + 3])
        noise_var = self.yerr ** 2 + sigma_n ** 2
        K_noise = K + jnp.diag(noise_var) + 1e-8 * jnp.eye(N)

        L = jax.scipy.linalg.cholesky(K_noise, lower=True)
        resid = self.y - self.mean_val
        alpha = jax.scipy.linalg.cho_solve((L, True), resid)

        lag_cross = jnp.abs(xpred[:, None] - self.x[None, :])
        Ks_geom = _kernel_eval(
            theta_kernel, lag_cross.ravel(),
            self.harmonics, self.n_lat, self.lat_range,
            quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
            r_gamma_func=r_gamma_fn
        ).reshape(lag_cross.shape)
        Ks = (c_spot_pred * c_spot_obs[None, :]
              + w_fac * c_fac_pred * c_fac_obs[None, :]) * Ks_geom

        mu_pred = self.mean_val + Ks @ alpha

        V = jax.scipy.linalg.cho_solve((L, True), Ks.T)
        k0 = float(_kernel_eval(
            theta_kernel, jnp.zeros(1),
            self.harmonics, self.n_lat, self.lat_range,
            quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
            r_gamma_func=r_gamma_fn)[0])
        var_prior = (c_spot_pred ** 2 + w_fac * c_fac_pred ** 2) * k0
        var_pred = var_prior - jnp.einsum('ij,ji->i', Ks, V)

        return np.asarray(mu_pred), np.asarray(var_pred)

    def amplitude_ratio(self, lam1, lam2, theta=None):
        """
        Predicted variability amplitude ratio between two bands.

        For the rank-2 model:
            sigma(lam1) / sigma(lam2) = sqrt(C(lam1,lam1) / C(lam2,lam2))

        where C(lam,lam) = c_s(lam)^2 + w_fac * c_f(lam)^2.
        """
        if theta is not None:
            if isinstance(theta, dict):
                T_spot = theta.get("T_spot", self.T_spot_init)
                T_fac = theta.get("T_fac", self.T_fac_init)
                w_fac = theta.get("w_fac", self.w_fac_init)
            else:
                theta_arr = jnp.asarray(theta)
                theta_phys = self._to_physical(theta_arr)
                T_spot = float(theta_phys[self._n_kernel])
                T_fac = float(theta_phys[self._n_kernel + 1])
                w_fac = float(theta_phys[self._n_kernel + 2])
        else:
            T_spot = self.T_spot_init
            T_fac = self.T_fac_init
            w_fac = self.w_fac_init

        cs1 = float(self._contrast_fn(jnp.array(lam1), T_spot, self.T_phot))
        cs2 = float(self._contrast_fn(jnp.array(lam2), T_spot, self.T_phot))
        cf1 = float(self._contrast_fn(jnp.array(lam1), T_fac, self.T_phot))
        cf2 = float(self._contrast_fn(jnp.array(lam2), T_fac, self.T_phot))
        var1 = cs1 ** 2 + w_fac * cf1 ** 2
        var2 = cs2 ** 2 + w_fac * cf2 ** 2
        return float(jnp.sqrt(var1 / var2))

    def contrast_matrix_at(self, theta=None):
        """
        Contrast matrix C_lambda at given parameters.

        Returns the N_lambda x N_lambda rank-2 matrix for diagnostic
        eigenvalue analysis.
        """
        if theta is not None:
            if isinstance(theta, dict):
                T_spot = theta.get("T_spot", self.T_spot_init)
                T_fac = theta.get("T_fac", self.T_fac_init)
                w_fac = theta.get("w_fac", self.w_fac_init)
            else:
                theta_arr = jnp.asarray(theta)
                theta_phys = self._to_physical(theta_arr)
                T_spot = float(theta_phys[self._n_kernel])
                T_fac = float(theta_phys[self._n_kernel + 1])
                w_fac = float(theta_phys[self._n_kernel + 2])
        else:
            T_spot = self.T_spot_init
            T_fac = self.T_fac_init
            w_fac = self.w_fac_init

        return contrast_matrix(
            self._band_wavelengths,
            jnp.array([T_spot, T_fac]),
            self.T_phot,
            weights=jnp.array([1.0, w_fac]))
