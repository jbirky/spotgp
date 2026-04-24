"""
spectral_temporal_kernel.py — 2D spectral-temporal GP kernel K_S(τ, Δv).

Extends the photometric kernel (AnalyticKernel) to velocity-resolved
spectroscopy.  The scalar |c_n(Φ)|² coefficients are replaced by
velocity-dependent harmonic amplitudes A_n(Δv; Φ), producing a kernel
that is a function of both time lag τ and velocity separation Δv.

Central equation (Paper III, Eq. 22):

    K_S(τ, Δv) = σ_S² R_Γ(τ) ∫ p(Φ) Σ_n A_n(Δv; Φ) cos(n ω(Φ) τ) dΦ

The photometric kernel is recovered by integrating over Δv:

    K(τ) = ∫ K_S(τ, Δv) dΔv

Reference: Birky et al. Paper III, Sections 5–7.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

try:
    from .spot_model import SpotEvolutionModel, _gauss_legendre_grid
    from .params import resolve_hparam
    from .envelope import (
        EnvelopeFunction,
        SkewedGaussianEnvelope,
        TrapezoidAsymmetricEnvelope,
        ExponentialEnvelope,
    )
    from .visibility import EdgeOnVisibilityFunction
    from .harmonic_amplitudes import compute_harmonic_amplitudes
except ImportError:
    from spot_model import SpotEvolutionModel, _gauss_legendre_grid
    from params import resolve_hparam
    from envelope import (
        EnvelopeFunction,
        SkewedGaussianEnvelope,
        TrapezoidAsymmetricEnvelope,
        ExponentialEnvelope,
    )
    from visibility import EdgeOnVisibilityFunction
    from harmonic_amplitudes import compute_harmonic_amplitudes

__all__ = ["SpectralTemporalKernel"]


class SpectralTemporalKernel:
    """
    2D spectral-temporal GP kernel K_S(τ, Δv) for starspot variability.

    Parameters
    ----------
    model_or_hparam : SpotEvolutionModel or dict
        Spot model instance or hyperparameter dict.
    dv_grid : array_like
        Velocity-lag grid on which to evaluate A_n [km/s].
    sigma_H : float
        Intrinsic local line profile width [km/s].
    vsini : float
        Projected equatorial rotation velocity [km/s].
    sigma_S : float or None
        Spectroscopic amplitude.  If None, uses sigma_k from the model.
    n_harmonics : int
        Number of rotational harmonics (default 3).
    n_lat : int
        Number of latitude quadrature points (default 64).
    lat_range : tuple or None
        (min, max) latitude in radians.
    quadrature : str
        Latitude integration method: "trapezoid" or "gauss-legendre".
    n_theta : int
        θ-grid resolution for h_m Fourier decomposition.
    n_v_internal : int
        Internal velocity-grid points for A_n FFT computation.
    v_pad_factor : float
        Internal v-grid extends to ±v_pad_factor × (|a| + 3σ_H).
    """

    def __init__(
        self,
        model_or_hparam,
        dv_grid,
        sigma_H,
        vsini,
        sigma_S=None,
        n_harmonics=3,
        n_lat=64,
        lat_range=None,
        quadrature="trapezoid",
        n_theta=256,
        n_v_internal=1024,
        v_pad_factor=5.0,
    ):
        # ── Parse model / hparam (same pattern as AnalyticKernel) ─────────
        if isinstance(model_or_hparam, SpotEvolutionModel):
            self.spot_model = model_or_hparam
            self.hparam = model_or_hparam.to_hparam()
        else:
            self.hparam = resolve_hparam(model_or_hparam)
            self.spot_model = SpotEvolutionModel.from_hparam(self.hparam)

        self.envelope = self.spot_model.envelope
        self.visibility = self.spot_model.visibility

        self.peq = self.spot_model.peq
        self.kappa = self.spot_model.kappa
        self.inc = self.spot_model.inc
        self.lspot = self.spot_model.lspot
        self.tau_spot = self.spot_model.tau_spot

        # ── Spectroscopic parameters ──────────────────────────────────────
        self.dv_grid = np.asarray(dv_grid, dtype=float)
        self.sigma_H = float(sigma_H)
        self.vsini = float(vsini)
        self.sigma_S = float(sigma_S) if sigma_S is not None else self.spot_model.sigma_k

        # ── Kernel configuration ──────────────────────────────────────────
        self.n_harmonics = n_harmonics
        self.n_lat = n_lat
        self.lat_range = (
            lat_range if lat_range is not None
            else self.spot_model.latitude_distribution.lat_range
        )
        self.quadrature = quadrature

        if quadrature == "gauss-legendre":
            self._quad_nodes, self._quad_weights = _gauss_legendre_grid(
                n_lat, self.lat_range[0], self.lat_range[1])
        elif quadrature == "trapezoid":
            self._quad_nodes = None
            self._quad_weights = None
        else:
            raise ValueError(
                f"Unknown quadrature method: {quadrature!r}. "
                "Use 'trapezoid' or 'gauss-legendre'.")

        # ── Precompute harmonic amplitudes at all quadrature latitudes ────
        self._n_theta = n_theta
        self._n_v_internal = n_v_internal
        self._v_pad_factor = v_pad_factor
        self._precompute_amplitudes()

    # ── Precomputation ─────────────────────────────────────────────────────

    def _precompute_amplitudes(self):
        """Compute A_n(Δv; Φ) and ω₀(Φ) at every quadrature latitude."""
        phi_grid = self._get_phi_grid()
        self._phi_grid = jnp.array(phi_grid)

        A_n_all = []
        omega0_all = []
        for phi in phi_grid:
            A_n = compute_harmonic_amplitudes(
                self.dv_grid,
                float(phi),
                float(self.inc),
                self.sigma_H,
                self.vsini,
                n_harmonics=self.n_harmonics,
                n_theta=self._n_theta,
                n_v_internal=self._n_v_internal,
                v_pad_factor=self._v_pad_factor,
            )
            A_n_all.append(A_n)
            omega0_all.append(float(self.visibility.omega0(float(phi))))

        # shape: (n_lat, n_harmonics+1, N_dv)
        self._A_n_precomputed = jnp.array(np.array(A_n_all))
        self._omega0_precomputed = jnp.array(omega0_all)

    def _get_phi_grid(self):
        if self.quadrature == "gauss-legendre":
            return self._quad_nodes
        phi_min, phi_max = self.lat_range
        return np.linspace(phi_min, phi_max, self.n_lat)

    # ── Core kernel helpers ────────────────────────────────────────────────

    def omega0(self, phi):
        return self.visibility.omega0(phi)

    def R_Gamma(self, lag):
        return self.envelope.R_Gamma(jnp.asarray(lag))

    # ── Single-latitude kernel ─────────────────────────────────────────────

    def kernel_single_latitude(self, lag, phi_idx):
        """
        K_S(τ, Δv) at a single latitude quadrature point.

        Parameters
        ----------
        lag : array_like, shape (M,)
        phi_idx : int
            Index into the precomputed latitude grid.

        Returns
        -------
        K_S : ndarray, shape (M, N_dv)
        """
        lag = jnp.asarray(lag, dtype=float).ravel()
        R = self.R_Gamma(lag)                               # (M,)
        A_n = self._A_n_precomputed[phi_idx]                # (n_harmonics+1, N_dv)
        w0 = self._omega0_precomputed[phi_idx]

        ns = jnp.arange(1, self.n_harmonics + 1)
        # cos(n * w0 * tau) -> shape (n_harmonics, M)
        cosines = jnp.cos(ns[:, None] * w0 * lag[None, :])

        # A_0(Δv) + 2 Σ_{n≥1} A_n(Δv) cos(n ω₀ τ)
        # A_n[1:] shape: (n_harmonics, N_dv)
        # cosines shape: (n_harmonics, M)
        # einsum 'nd,nM->Md' gives (M, N_dv)
        harmonic_sum = A_n[0][None, :] + 2.0 * jnp.einsum(
            'nd,nm->md', A_n[1:], cosines)

        return self.sigma_S ** 2 * R[:, None] * harmonic_sum

    # ── Latitude-averaged kernel ───────────────────────────────────────────

    def _kernel_stationary(self, lag, lat_dist=None):
        """
        Stationary kernel without the σ_S² prefactor.

        Returns R_Γ(τ) · Σ_n A_n(Δv) cos(nω₀τ), averaged over latitude.
        Shape: (M, N_dv).
        """
        lag = jnp.asarray(lag, dtype=float).ravel()
        M = len(lag)
        N_dv = len(self.dv_grid)

        if lat_dist is None:
            lat_dist = self.spot_model.latitude_distribution

        R = self.R_Gamma(lag)  # (M,)
        n_harmonics = self.n_harmonics
        phi_grid = self._phi_grid
        A_n_all = self._A_n_precomputed  # (n_lat, n_harmonics+1, N_dv)
        omega0_all = self._omega0_precomputed  # (n_lat,)

        # Build quadrature weights
        if self.quadrature == "gauss-legendre":
            quad_weights = jnp.array(self._quad_weights)
            user_weights = jnp.array([lat_dist(float(p)) for p in np.asarray(phi_grid)])
            weights = user_weights * quad_weights
            norm = jnp.sum(weights)
        else:
            dphi = phi_grid[1] - phi_grid[0]
            user_weights = jnp.array([lat_dist(float(p)) for p in np.asarray(phi_grid)])
            weights = user_weights * dphi
            norm = jnp.trapezoid(user_weights, phi_grid)

        ns = jnp.arange(1, n_harmonics + 1)

        def _scan_body(K_acc, idx):
            A_n = A_n_all[idx]                              # (n_harmonics+1, N_dv)
            w0 = omega0_all[idx]
            w = weights[idx]

            cosines = jnp.cos(ns[:, None] * w0 * lag[None, :])  # (n_harmonics, M)
            contribution = A_n[0][None, :] + 2.0 * jnp.einsum(
                'nd,nm->md', A_n[1:], cosines)              # (M, N_dv)

            return K_acc + w * contribution, None

        K, _ = jax.lax.scan(
            _scan_body, jnp.zeros((M, N_dv)), jnp.arange(len(phi_grid)))
        K = R[:, None] * K / norm

        return K

    def kernel(self, lag, lat_dist=None):
        """
        Full 2D kernel K_S(τ, Δv) averaged over latitude.

        Parameters
        ----------
        lag : array_like
            Time lags [days].
        lat_dist : callable or None
            Latitude probability density.

        Returns
        -------
        K_S : ndarray, shape (len(lag), N_dv)
        """
        K = self._kernel_stationary(lag, lat_dist=lat_dist)
        return np.asarray(self.sigma_S ** 2 * K)

    # ── Derived quantities ─────────────────────────────────────────────────

    def photometric_kernel(self, lag, lat_dist=None):
        """
        Recover the photometric kernel by integrating over Δv.

        K(τ) = ∫ K_S(τ, Δv) dΔv

        Returns
        -------
        K : ndarray, shape (len(lag),)
        """
        K_S = self.kernel(lag, lat_dist=lat_dist)
        return np.asarray(jnp.trapezoid(K_S, self.dv_grid, axis=1))

    def rv_autocovariance(self, lag, lat_dist=None):
        """
        RV autocovariance as the second velocity moment of the kernel.

        Cov[RV(t), RV(t+τ)] = (1/W₀²) ∫∫ v v' K_S(τ, v'-v) dv dv'

        For a stationary kernel in v, this simplifies to a weighted integral
        over Δv of K_S(τ, Δv) with a triangular moment kernel.

        Returns
        -------
        C_RV : ndarray, shape (len(lag),)
        """
        K_S = self.kernel(lag, lat_dist=lat_dist)
        dv = self.dv_grid
        # Second moment: ∫ Δv² K_S(τ, Δv) dΔv (leading-order approximation)
        return np.asarray(jnp.trapezoid(K_S * dv[None, :] ** 2, dv, axis=1))

    def ccf_covariance(self, lag, sum_w_sq=1.0, lat_dist=None):
        """
        CCF covariance = (Σ w_ℓ²) K_S(τ, Δv).

        Parameters
        ----------
        lag : array_like
        sum_w_sq : float
            Sum of squared line weights.

        Returns
        -------
        C_CCF : ndarray, shape (len(lag), N_dv)
        """
        return sum_w_sq * self.kernel(lag, lat_dist=lat_dist)

    def cross_covariance_phot_spec(self, lag, lat_dist=None):
        """
        Cross-covariance between photometric flux and spectral perturbation.

        K_×(τ, v) = Cov[δF(t), δS(v, t+τ)]
                   = σ_S σ_k ∫ K_S(τ, Δv) dΔv evaluated at each Δv.

        Since photometric flux is the velocity-integrated spectral signal,
        the cross-covariance is the marginal of K_S over one velocity index:

            K_×(τ, Δv) = σ_S · σ_k · R_Γ(τ) ∫ p(Φ) Σ_n A_n^(×)(Δv; Φ) cos(nωτ) dΦ

        In the leading-order approximation (shared amplitude scaling),
        this is proportional to K_S itself, weighted by the photometric
        kernel ratio:

            K_×(τ, Δv) ≈ sqrt(K_phot(τ) / ∫K_S dΔv) · K_S(τ, Δv)

        For practical use, we provide the simpler factored form.

        Parameters
        ----------
        lag : array_like
            Time lags [days].

        Returns
        -------
        K_cross : ndarray, shape (len(lag), N_dv)
        """
        K_S = self._kernel_stationary(lag, lat_dist=lat_dist)
        sigma_k = self.spot_model.sigma_k
        return np.asarray(self.sigma_S * sigma_k * K_S)
