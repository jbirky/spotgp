import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

try:
    from .params import resolve_hparam
    from .envelope import (
        EnvelopeFunction,
        TrapezoidAsymmetricEnvelope,
        SkewedGaussianEnvelope,
        ExponentialEnvelope,
        compute_R_Gamma_numerical,
    )
    from .spot_model import (
        VisibilityFunction, EdgeOnVisibilityFunction, SpotEvolutionModel,
        _cn_squared_coefficients_jax, _gauss_legendre_grid,
    )
except ImportError:
    from params import resolve_hparam
    from envelope import (
        EnvelopeFunction,
        TrapezoidAsymmetricEnvelope,
        SkewedGaussianEnvelope,
        ExponentialEnvelope,
        compute_R_Gamma_numerical,
    )
    from spot_model import (
        VisibilityFunction, EdgeOnVisibilityFunction, SpotEvolutionModel,
        _cn_squared_coefficients_jax, _gauss_legendre_grid,
    )

__all__ = ["AnalyticKernel", "NonstationaryAnalyticKernel",
           "compute_R_Gamma_numerical"]


class AnalyticKernel:
    """
    JAX-accelerated analytic GP kernel for stellar rotation variability.

    Parameters
    ----------
    model_or_hparam : SpotEvolutionModel or dict
        Either a SpotEvolutionModel instance (new API) or a raw hparam dict
        (backward-compatible old API).
    n_harmonics : int
        Number of Fourier harmonics for the visibility function (default 3).
    n_lat : int
        Number of latitude quadrature points (default 64).
    lat_range : tuple
        (min, max) latitude in radians (default (-pi/2, pi/2)).
    quadrature : str
        Latitude integration method: "trapezoid" or "gauss-legendre".
    """

    def __init__(self, model_or_hparam, n_harmonics=3, n_lat=64,
                 lat_range=None, quadrature="trapezoid"):

        # ── Accept SpotEvolutionModel or legacy hparam dict ────────────────
        if isinstance(model_or_hparam, SpotEvolutionModel):
            self.spot_model = model_or_hparam
            self.hparam = model_or_hparam.to_hparam()
        else:
            # Backward compat: dict input
            self.hparam = resolve_hparam(model_or_hparam)
            self.spot_model = SpotEvolutionModel.from_hparam(self.hparam)

        # ── Unpack commonly-used params ────────────────────────────────────
        self.envelope   = self.spot_model.envelope
        self.visibility = self.spot_model.visibility

        self.peq     = self.spot_model.peq
        self.kappa   = self.spot_model.kappa
        self.inc     = self.spot_model.inc
        self.lspot   = self.spot_model.lspot
        self.sigma_k = self.spot_model.sigma_k
        self.tau_spot = self.spot_model.tau_spot

        # ── Envelope-type attributes (backward compat) ────────────────────
        if isinstance(self.envelope, SkewedGaussianEnvelope):
            self.envelope_type = "skew_normal"
            self.sigma_sn  = self.envelope.sigma_sn
            self.n_sn      = self.envelope.n_sn
            self.tau_em    = self.tau_spot
            self.tau_dec   = self.tau_spot
            self.asymmetric = False
            # Re-use grids from the envelope object
            self._R_Gamma_lag_grid = self.envelope._R_lag_grid
            self._R_Gamma_vals     = self.envelope._R_vals
            self._Gh_sq_omega_grid = self.envelope._Gh_omega_grid
            self._Gh_sq_vals       = self.envelope._Gh_sq_vals

        elif isinstance(self.envelope, TrapezoidAsymmetricEnvelope):
            self.envelope_type = "trapezoid_asymmetric"
            self.asymmetric = True
            self.tau_em  = self.envelope.tau_em
            self.tau_dec = self.envelope.tau_dec
            self._te = min(self.tau_em, self.tau_dec)
            self._td = max(self.tau_em, self.tau_dec)

        elif isinstance(self.envelope, ExponentialEnvelope):
            self.envelope_type = "exponential"
            self.asymmetric = False
            self.tau_em  = self.tau_spot
            self.tau_dec = self.tau_spot

        else:
            # Default: symmetric trapezoid (or any other future type)
            self.envelope_type = "trapezoid_symmetric"
            self.asymmetric = False
            self.tau_em  = self.tau_spot
            self.tau_dec = self.tau_spot

        # ── Kernel config ──────────────────────────────────────────────────
        self.n_harmonics = n_harmonics
        self.n_lat       = n_lat
        self.lat_range   = (lat_range if lat_range is not None
                            else self.spot_model.latitude_distribution.lat_range)
        self.quadrature  = quadrature

        if quadrature == "gauss-legendre":
            self._quad_nodes, self._quad_weights = _gauss_legendre_grid(
                n_lat, lat_range[0], lat_range[1])
        elif quadrature == "trapezoid":
            self._quad_nodes   = None
            self._quad_weights = None
        else:
            raise ValueError(
                f"Unknown quadrature method: {quadrature!r}. "
                "Use 'trapezoid' or 'gauss-legendre'.")

    # ── Core kernel helpers ─────────────────────────────────────────────────

    def omega0(self, phi):
        """Latitude-dependent rotation angular frequency [rad/day]."""
        return self.visibility.omega0(phi)

    def R_Gamma(self, lag):
        """Autocorrelation of the squared envelope (delegates to envelope)."""
        return self.envelope.R_Gamma(jnp.asarray(lag))

    def cn_squared(self, phi):
        """Squared Fourier visibility coefficients at latitude phi."""
        return self.visibility.cn_squared(phi, self.n_harmonics)

    # ── Single-latitude kernel ──────────────────────────────────────────────

    def kernel_single_latitude(self, lag, phi):
        """Single-spot kernel at a fixed latitude."""
        lag = jnp.asarray(lag, dtype=float).ravel()
        R = self.R_Gamma(lag)
        cn_sq = self.cn_squared(phi)
        w0 = self.omega0(phi)

        ns = jnp.arange(1, len(cn_sq))
        cosine_terms = jnp.sum(
            cn_sq[1:] * jnp.cos(ns * w0 * lag[:, None]), axis=1)
        return R * (cn_sq[0] + 2 * cosine_terms)

    # ── Stationary kernel (without sigma_k² scaling) ─────────────────────

    def _kernel_stationary(self, lag, lat_dist=None):
        """
        Stationary kernel *without* the σ_k² prefactor.

        Returns R_Γ(τ) · Σ_n w_n |c_n|² cos(n·ω₀·τ), averaged over latitude.
        """
        lag = jnp.asarray(lag, dtype=float)
        orig_shape = lag.shape
        lag_flat = lag.ravel()

        if isinstance(self.visibility, EdgeOnVisibilityFunction):
            R = self.R_Gamma(lag_flat)
            cn_sq = self.visibility.cn_squared(0.0, self.n_harmonics)
            w0 = self.visibility.omega0(0.0)
            ns = jnp.arange(1, self.n_harmonics + 1)
            cosine_terms = jnp.sum(
                cn_sq[1:] * jnp.cos(ns * w0 * lag_flat[:, None]), axis=1)
            K = R * (cn_sq[0] + 2 * cosine_terms)
            return K.reshape(orig_shape)

        if lat_dist is None:
            lat_dist = self.spot_model.latitude_distribution

        R = self.R_Gamma(lag_flat)
        n_harmonics = self.n_harmonics

        def _lat_contribution(phi):
            cn_sq = self.cn_squared(phi)
            w0 = self.omega0(phi)
            ns = jnp.arange(1, n_harmonics + 1)
            cosine_terms = jnp.sum(
                cn_sq[1:] * jnp.cos(ns * w0 * lag_flat[:, None]), axis=1)
            return cn_sq[0] + 2 * cosine_terms

        if self.quadrature == "gauss-legendre":
            phi_grid   = self._quad_nodes
            quad_weights = self._quad_weights
            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            weights = user_weights * quad_weights
            norm = jnp.sum(weights)
        else:
            phi_min, phi_max = self.lat_range
            phi_grid = jnp.linspace(phi_min, phi_max, self.n_lat)
            dphi = phi_grid[1] - phi_grid[0]
            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            weights = user_weights * dphi
            norm = jnp.trapezoid(user_weights, phi_grid)

        def _scan_body(K_acc, idx):
            phi = phi_grid[idx]
            w   = weights[idx]
            return K_acc + w * _lat_contribution(phi), None

        K, _ = jax.lax.scan(
            _scan_body, jnp.zeros_like(lag_flat), jnp.arange(len(phi_grid)))
        K = R * K / norm

        return K.reshape(orig_shape)

    # ── Full kernel (latitude-averaged) ────────────────────────────────────

    def kernel(self, lag, lat_dist=None):
        """
        Full GP kernel averaged over latitude.

        Uses jax.lax.scan for memory-efficient accumulation: only one
        lag-sized buffer is live at a time — O(M) instead of O(n_lat·M).

        When the visibility function is an EdgeOnVisibilityFunction, the
        latitude-averaged ``|c_n|^2`` are known constants and the latitude
        loop is bypassed entirely.

        Parameters
        ----------
        lag : array_like
            Time lags [days]. Can be 1D or 2D.
        lat_dist : callable or None
            Latitude probability density. If None, uniform.

        Returns
        -------
        K : ndarray, same shape as lag input.
        """
        K = self._kernel_stationary(lag, lat_dist=lat_dist)
        return np.asarray(self.sigma_k ** 2 * K)

    def kernel_solid_body(self, lag, lat_dist=None):
        """Kernel for solid-body rotation (kappa=0)."""
        lag = jnp.asarray(lag, dtype=float)

        if lat_dist is None:
            lat_dist = self.spot_model.latitude_distribution

        if self.quadrature == "gauss-legendre":
            phi_grid   = self._quad_nodes
            quad_weights = self._quad_weights
            all_cn_sq  = jax.vmap(
                lambda phi: _cn_squared_coefficients_jax(
                    self.inc, phi, self.n_harmonics))(phi_grid)
            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            norm = jnp.sum(user_weights * quad_weights)
            cn_sq_avg = jnp.sum(
                user_weights[:, None] * quad_weights[:, None] * all_cn_sq,
                axis=0) / norm
        else:
            phi_min, phi_max = self.lat_range
            phi_grid = jnp.linspace(phi_min, phi_max, self.n_lat)
            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            norm = jnp.trapezoid(user_weights, phi_grid)
            all_cn_sq = jax.vmap(
                lambda phi: _cn_squared_coefficients_jax(
                    self.inc, phi, self.n_harmonics))(phi_grid)
            cn_sq_avg = jnp.sum(
                user_weights[:, None] * all_cn_sq, axis=0
            ) * (phi_grid[1] - phi_grid[0]) / norm

        w0 = 2 * jnp.pi / self.peq
        R  = self.R_Gamma(lag)
        ns = jnp.arange(1, len(cn_sq_avg))
        cosine_terms = jnp.sum(
            cn_sq_avg[1:] * jnp.cos(ns * w0 * lag[:, None]), axis=1)
        return np.asarray(R * (cn_sq_avg[0] + 2 * cosine_terms) * self.sigma_k ** 2)

    # ── Power spectral density ──────────────────────────────────────────────

    def compute_psd(self, omega, lat_dist=None):
        """
        Analytic power spectral density.

        Parameters
        ----------
        omega : array_like
            Angular frequencies [rad/day].
        lat_dist : callable or None
            Latitude probability density.

        Returns
        -------
        freq : ndarray   [cycles/day]
        power : ndarray
        """
        omega = jnp.asarray(omega, dtype=float)

        if lat_dist is None:
            lat_dist = self.spot_model.latitude_distribution

        # Build the per-latitude PSD contribution based on envelope type
        if isinstance(self.envelope, (SkewedGaussianEnvelope, ExponentialEnvelope)):
            # Use envelope's Gamma_hat_sq directly
            def _psd_at_lat(phi):
                cn_sq = self.cn_squared(phi)
                w0 = self.omega0(phi)

                contrib = cn_sq[0] * self.envelope.Gamma_hat_sq(omega)

                def _harmonic(n):
                    return cn_sq[n] * (
                        self.envelope.Gamma_hat_sq(omega - n * w0)
                        + self.envelope.Gamma_hat_sq(omega + n * w0))

                ns = jnp.arange(1, len(cn_sq))
                harmonic_contribs = jax.vmap(lambda n: _harmonic(n))(ns)
                return contrib + jnp.sum(harmonic_contribs, axis=0)

        else:
            # Trapezoid types use the closed-form _Gamma_hat
            def _psd_at_lat(phi):
                cn_sq = self.cn_squared(phi)
                w0 = self.omega0(phi)

                Gh_0 = self.envelope.Gamma_hat(omega)
                contrib = cn_sq[0] * Gh_0 ** 2

                def _harmonic(n):
                    Gh_p = self.envelope.Gamma_hat(omega - n * w0)
                    Gh_m = self.envelope.Gamma_hat(omega + n * w0)
                    return cn_sq[n] * (Gh_p ** 2 + Gh_m ** 2)

                ns = jnp.arange(1, len(cn_sq))
                harmonic_contribs = jax.vmap(lambda n: _harmonic(n))(ns)
                return contrib + jnp.sum(harmonic_contribs, axis=0)

        if self.quadrature == "gauss-legendre":
            phi_grid    = self._quad_nodes
            quad_weights = self._quad_weights
            all_contribs = jax.vmap(_psd_at_lat)(phi_grid)
            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            norm = jnp.sum(user_weights * quad_weights)
            psd  = jnp.sum(
                user_weights[:, None] * quad_weights[:, None]
                * all_contribs, axis=0) / norm
        else:
            phi_min, phi_max = self.lat_range
            phi_grid = jnp.linspace(phi_min, phi_max, self.n_lat)
            dphi = phi_grid[1] - phi_grid[0]
            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            norm = jnp.trapezoid(user_weights, phi_grid)
            all_contribs = jax.vmap(_psd_at_lat)(phi_grid)
            psd = jnp.sum(user_weights[:, None] * all_contribs, axis=0) * dphi / norm

        psd = psd * self.sigma_k ** 2

        self.psd_omega = np.asarray(omega)
        self.psd_freq  = np.asarray(omega / (2 * jnp.pi))
        self.psd_power = np.asarray(psd)

        return self.psd_freq, self.psd_power

    def build_jax(self, n_lag=256):
        """
        Pre-compile and warm up JAX JIT computation for this kernel.

        ``jax.lax.scan`` (used inside ``kernel()``) triggers XLA compilation
        on its first call for a given array shape.  That compilation can take
        several seconds and is easy to mistake for slow runtime.  Call
        ``build_jax()`` once after constructing the kernel to pay that cost
        upfront — subsequent calls to ``kernel()`` and ``compute_psd()`` with
        the same shape will be fast.

        Parameters
        ----------
        n_lag : int
            Length of the dummy lag array used to drive compilation (default
            256).  The actual value does not matter as long as it is
            representative of the sizes you will use at runtime.

        Returns
        -------
        self : AnalyticKernel
            Returns ``self`` so the call can be chained:
            ``ak = AnalyticKernel(model).build_jax()``.
        """
        import time

        dummy_lag = jnp.linspace(0.0, float(self.peq) * 3.0, n_lag)
        dummy_omega = jnp.linspace(0.0, 4.0 * float(np.pi / self.peq), n_lag)

        t0 = time.time()
        jax.block_until_ready(self.kernel(dummy_lag))
        jax.block_until_ready(self.compute_psd(dummy_omega))
        print(f"JAX kernel compiled in {np.round(time.time() - t0, 2)}s")

        t0 = time.time()
        jax.block_until_ready(self.kernel(dummy_lag))
        jax.block_until_ready(self.compute_psd(dummy_omega))
        print(f"JAX kernel recompute in {np.round(time.time() - t0, 2)}s")

        return self

    def __call__(self, lag, **kwargs):
        """Evaluate the kernel at the given lags."""
        return self.kernel(lag, **kwargs)


class NonstationaryAnalyticKernel(AnalyticKernel):
    """
    Non-stationary extension of AnalyticKernel with time-dependent σ_k.

    The covariance between times t1 and t2 is:

        K(t1, t2) = σ_k(t1) · σ_k(t2) · K_stationary(|t1 - t2|)

    where K_stationary is the latitude-averaged kernel without the σ_k²
    prefactor.  This factorization guarantees positive semi-definiteness
    for any non-negative σ_k(t).

    Parameters
    ----------
    model_or_hparam : SpotEvolutionModel or dict
        Same as AnalyticKernel.
    sigma_k_func : callable
        Function mapping time (scalar or array) to σ_k values.
        Signature: ``sigma_k_func(t) -> array_like``.
    **kwargs
        Forwarded to AnalyticKernel (n_harmonics, n_lat, etc.).

    Examples
    --------
    >>> def activity_cycle(t, sigma0=0.01, amp=0.5, period=365.0):
    ...     return sigma0 * (1 + amp * jnp.sin(2 * jnp.pi * t / period))
    >>> nsk = NonstationaryAnalyticKernel(model, sigma_k_func=activity_cycle)
    >>> K = nsk.kernel_matrix(t_obs)
    """

    def __init__(self, model_or_hparam, sigma_k_func, **kwargs):
        super().__init__(model_or_hparam, **kwargs)
        self.sigma_k_func = sigma_k_func

    def kernel_matrix(self, t, lat_dist=None):
        """
        Build the full N×N covariance matrix for observation times.

        Parameters
        ----------
        t : array_like, shape (N,)
            Observation times [days].
        lat_dist : callable or None
            Latitude probability density (forwarded to parent).

        Returns
        -------
        K : ndarray, shape (N, N)
        """
        t = jnp.asarray(t, dtype=float).ravel()
        lag = jnp.abs(t[:, None] - t[None, :])

        K_stat = self._kernel_stationary(lag, lat_dist=lat_dist)

        sk = jnp.asarray(self.sigma_k_func(t))
        K = sk[:, None] * sk[None, :] * K_stat

        return np.asarray(K)

    def kernel(self, lag, lat_dist=None):
        """
        Stationary kernel using the constant ``self.sigma_k``.

        Provided for backward compatibility — use ``kernel_matrix(t)``
        for the non-stationary covariance.
        """
        return super().kernel(lag, lat_dist=lat_dist)

    def __call__(self, t, **kwargs):
        """Evaluate the non-stationary kernel matrix at observation times."""
        return self.kernel_matrix(t, **kwargs)
