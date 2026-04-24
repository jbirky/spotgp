"""
harmonic_amplitudes.py — Velocity-dependent harmonic amplitudes for the
spectral-temporal starspot kernel.

The harmonic amplitudes A_n(Δv; Φ) encode how each rotational harmonic of
the stellar variability is distributed across the line profile.  They extend
the photometric kernel (Paper I) to velocity-resolved spectroscopy (Paper III).

The photometric kernel uses scalar |c_n|² coefficients; the spectral kernel
replaces these with A_n(Δv; Φ), which depend on the velocity lag Δv and the
local line profile H(v).

Central identity (Parseval):
    ∫ A_n(Δv) dΔv = |c_n|² · (∫ H(v) dv)²

Reference: Paper III, Equations 16–18, 312–328.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from .visibility import _cn_general_jax

__all__ = [
    "compute_harmonic_amplitudes",
    "compute_harmonic_amplitudes_direct",
    "compute_h_m",
    "compute_g_n",
]


# ── Fourier coefficients of Doppler-shifted profile ──────────────────────────

def compute_h_m(v_grid, a, sigma_H, n_theta=256):
    """
    Fourier coefficients h_m(v; Φ) of the Doppler-shifted local profile.

        h_m(v) = (1/2π) ∫₀²π H(v − a sin θ) e^{−imθ} dθ

    For Gaussian H(v) = exp(−v²/2σ_H²).

    Parameters
    ----------
    v_grid : (N_v,) array
        Velocity grid [km/s].
    a : float
        Projected velocity semi-amplitude: a = v sin i · cos Φ [km/s].
    sigma_H : float
        Intrinsic local line profile width [km/s].
    n_theta : int
        Number of θ-grid points for the DFT.

    Returns
    -------
    h_m : (N_v, n_theta) complex array
        Fourier coefficients.  Index m corresponds to FFT index:
        m = 0, 1, ..., n_theta/2, −n_theta/2+1, ..., −1.

    Notes
    -----
    h_m is real for even m and purely imaginary for odd m (consequence of
    H being even and the sin θ Doppler shift).  h_{−m} = h_m* (reality
    condition for real H).
    """
    v_grid = jnp.asarray(v_grid)
    theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)

    shifts = a * jnp.sin(theta)                       # (n_theta,)
    H_grid = jnp.exp(
        -(v_grid[:, None] - shifts[None, :]) ** 2
        / (2 * sigma_H ** 2)
    )                                                  # (N_v, n_theta)

    h_m = jnp.fft.fft(H_grid, axis=1) / n_theta       # DFT over θ
    return h_m


# ── Combined visibility–profile coefficient g_N ──────────────────────────────

def compute_g_n(h_m_all, cn_array, N):
    """
    Combined visibility–profile Fourier coefficient.

        g_N(v) = Σ_α  c_α  h_{N−α}(v)

    Parameters
    ----------
    h_m_all : (N_v, n_theta) complex array
        Full h_m array from :func:`compute_h_m`.
    cn_array : (n_cn,) array
        Visibility coefficients c_0, c_1, ..., c_{n_cn−1}.
        Negative indices use c_{−k} = c_k (real μ).
    N : int
        Harmonic index.

    Returns
    -------
    g_N : (N_v,) complex array
    """
    n_theta = h_m_all.shape[1]
    n_cn = len(cn_array)
    g = jnp.zeros(h_m_all.shape[0], dtype=complex)

    for alpha in range(-n_cn + 1, n_cn):
        c_alpha = cn_array[abs(alpha)]
        m = N - alpha
        m_idx = m % n_theta
        g = g + c_alpha * h_m_all[:, m_idx]

    return g


# ── Harmonic amplitudes via g_N autocorrelation ──────────────────────────────

def compute_harmonic_amplitudes(
    dv_grid,
    phi,
    inc,
    sigma_H,
    vsini,
    n_harmonics=3,
    n_theta=256,
    n_v_internal=1024,
    v_pad_factor=5.0,
):
    """
    Compute velocity-dependent harmonic amplitudes A_n(Δv; Φ).

    A_n(Δv) = ∫ g_n(v)* g_n(v + Δv) dv

    where g_n(v) = Σ_α c_α h_{n−α}(v).

    Parameters
    ----------
    dv_grid : (N_dv,) array
        Velocity-lag grid on which to evaluate A_n [km/s].
    phi : float
        Latitude [radians].
    inc : float
        Stellar inclination [radians].
    sigma_H : float
        Intrinsic local line profile width [km/s].
    vsini : float
        Projected equatorial rotation velocity [km/s].
    n_harmonics : int
        Number of harmonics (n = 0, 1, ..., n_harmonics).
    n_theta : int
        θ-grid resolution for Fourier decomposition of H.
    n_v_internal : int
        Number of internal velocity-grid points.
    v_pad_factor : float
        Internal v-grid extends to ±v_pad_factor × (|a| + 3σ_H).

    Returns
    -------
    A_n : (n_harmonics + 1, N_dv) complex array
        Harmonic amplitudes at each velocity lag.
    """
    dv_grid = np.asarray(dv_grid, dtype=float)
    a = vsini * np.cos(phi)

    v_max = v_pad_factor * (abs(a) + 3.0 * sigma_H)
    v_internal = jnp.linspace(-v_max, v_max, n_v_internal)
    dv_step = float(v_internal[1] - v_internal[0])

    h_m_all = compute_h_m(v_internal, a, sigma_H, n_theta=n_theta)

    n_cn = n_harmonics + n_theta // 2
    cn_array = np.array([
        float(_cn_general_jax(n, inc, phi)) for n in range(n_cn)
    ])

    N_pad = 2 * n_v_internal
    A_n_list = []

    for N in range(n_harmonics + 1):
        g_N = compute_g_n(h_m_all, cn_array, N)

        G = jnp.fft.fft(g_N, n=N_pad)
        power = G * jnp.conj(G)
        autocorr = jnp.fft.ifft(power).real * dv_step  # A_N on internal Δv grid

        lags = jnp.fft.fftfreq(N_pad, d=1.0) * N_pad * dv_step
        lags = np.asarray(jnp.fft.fftshift(lags))
        autocorr_shifted = np.asarray(jnp.fft.fftshift(autocorr))

        A_n_interp = np.interp(dv_grid, lags, autocorr_shifted)
        A_n_list.append(A_n_interp)

    return np.array(A_n_list)


# ── Direct computation via θ-integral (reference / verification) ─────────────

def compute_harmonic_amplitudes_direct(
    dv_grid,
    phi,
    inc,
    sigma_H,
    vsini,
    n_harmonics=3,
    n_theta=512,
    n_omega_tau=256,
):
    """
    Compute A_n(Δv) by direct numerical integration of the v-integrated
    visibility–Doppler covariance and harmonic extraction.

    For Gaussian H, the v-integral has a closed form:

        ∫ H(v − s₁) H(v + Δv − s₂) dv = σ_H √π exp(−(Δv − Δs)² / 4σ_H²)

    where Δs = s₂ − s₁ = a[sin(θ+ωτ) − sin θ].  The remaining θ-integral
    and harmonic extraction are done numerically.

    Parameters
    ----------
    dv_grid : (N_dv,) array
        Velocity-lag grid [km/s].
    phi, inc, sigma_H, vsini : float
        As in :func:`compute_harmonic_amplitudes`.
    n_harmonics : int
        Number of harmonics.
    n_theta : int
        θ-grid resolution.
    n_omega_tau : int
        Number of ωτ grid points.

    Returns
    -------
    A_n : (n_harmonics + 1, N_dv) real array
        Harmonic amplitudes at each velocity lag.
    """
    dv_grid = np.asarray(dv_grid, dtype=float)
    a = vsini * np.cos(phi)

    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    omega_tau_grid = np.linspace(0, 2 * np.pi, n_omega_tau, endpoint=False)

    n_cn_max = n_harmonics + n_theta // 4
    cn = np.array([float(_cn_general_jax(n, inc, phi)) for n in range(n_cn_max)])

    mu_theta = np.zeros(n_theta)
    for n_idx in range(n_cn_max):
        weight = 1.0 if n_idx == 0 else 2.0
        mu_theta += weight * cn[n_idx] * np.cos(n_idx * theta)

    prefactor = sigma_H * np.sqrt(np.pi)
    H_grid = np.zeros((n_omega_tau, len(dv_grid)))

    for i, wt in enumerate(omega_tau_grid):
        theta_shifted = theta + wt
        mu2 = np.zeros(n_theta)
        for n_idx in range(n_cn_max):
            weight = 1.0 if n_idx == 0 else 2.0
            mu2 += weight * cn[n_idx] * np.cos(n_idx * theta_shifted)

        delta_s = a * (np.sin(theta_shifted) - np.sin(theta))

        for j, dv_val in enumerate(dv_grid):
            gauss = np.exp(-(dv_val - delta_s) ** 2 / (4 * sigma_H ** 2))
            integrand = mu_theta * mu2 * gauss * prefactor
            H_grid[i, j] = np.mean(integrand)

    A_n = np.zeros((n_harmonics + 1, len(dv_grid)))

    for n in range(n_harmonics + 1):
        cos_n = np.cos(n * omega_tau_grid)
        A_n[n] = np.mean(H_grid * cos_n[:, None], axis=0)

    return A_n
