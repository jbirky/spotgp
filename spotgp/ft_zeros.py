"""
ft_zeros.py — Fourier transform zeros method for rotational broadening.

Implements the FTM diagnostic (Reiners & Schmitt 2002, 2003) as a derived
quantity from the spectral-temporal kernel K_S(τ, Δv):

    K_S(τ=0, Δv) → FT_{Δv} → zeros q_n → q_2/q_1 ratio

The q_2/q_1 ratio diagnoses differential rotation:
  - Rigid rotation:      1.72 < q_2/q_1 < 1.83
  - Solar-like DR (κ>0): q_2/q_1 < 1.72
  - Anti-solar DR (κ<0): q_2/q_1 > 1.83

This module also provides the classical rotational broadening function
FT and zero-finding for direct comparison.

Reference: Reiners & Schmitt (2002, 2003); Paper III, Section 7.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

try:
    from .line_profile import rotational_broadening_kernel
except ImportError:
    from line_profile import rotational_broadening_kernel

__all__ = [
    "broadening_function_ft",
    "find_ft_zeros",
    "q_ratio",
    "q_ratio_from_kernel",
    "rigid_rotation_q_ratio",
]


def broadening_function_ft(v, G):
    """
    Fourier transform of a broadening function G(v).

    Parameters
    ----------
    v : array_like, shape (N,)
        Uniformly spaced velocity grid [km/s].
    G : array_like, shape (N,)
        Broadening function on v.

    Returns
    -------
    sigma : ndarray
        Fourier conjugate variable [s/km].
    G_hat : ndarray
        |FT{G}|, normalized so G_hat[0] = 1.
    """
    v = np.asarray(v, dtype=float)
    G = np.asarray(G, dtype=float)
    dv = v[1] - v[0]
    N = len(v)

    G_fft = np.fft.fft(G) * dv
    G_fft = np.fft.fftshift(G_fft)
    sigma = np.fft.fftshift(np.fft.fftfreq(N, d=dv))

    G_hat = np.abs(G_fft)
    if G_hat.max() > 0:
        G_hat /= G_hat.max()

    mid = N // 2
    return sigma[mid:], G_hat[mid:]


def find_ft_zeros(sigma, G_hat, n_zeros=5, threshold=0.01):
    """
    Find the zero-crossing positions of |FT{G}|(σ).

    Parameters
    ----------
    sigma : array_like
        Fourier variable (positive frequencies only).
    G_hat : array_like
        Normalized |FT{G}|.
    n_zeros : int
        Maximum number of zeros to find.
    threshold : float
        Ignore zeros where the local amplitude is above this fraction
        of the peak.

    Returns
    -------
    q : ndarray
        Positions of the first n_zeros zero crossings in σ-space.
        NaN entries indicate zeros that could not be found.
    """
    sigma = np.asarray(sigma)
    G_hat = np.asarray(G_hat)

    zeros_found = []
    for i in range(1, len(G_hat) - 1):
        if len(zeros_found) >= n_zeros:
            break
        if G_hat[i] <= G_hat[i - 1] and G_hat[i] <= G_hat[i + 1]:
            if G_hat[i] < threshold:
                s_interp = sigma[i]
                zeros_found.append(s_interp)

    result = np.full(n_zeros, np.nan)
    result[:len(zeros_found)] = zeros_found
    return result


def q_ratio(sigma, G_hat, n_zeros=3):
    """
    Compute the q_2/q_1 ratio from the FT of a broadening function.

    Parameters
    ----------
    sigma : array_like
        Fourier variable (positive only).
    G_hat : array_like
        Normalized |FT|.
    n_zeros : int
        Number of zeros to find (need at least 2).

    Returns
    -------
    ratio : float
        q_2 / q_1.  NaN if fewer than 2 zeros found.
    """
    q = find_ft_zeros(sigma, G_hat, n_zeros=n_zeros)
    if np.isnan(q[0]) or np.isnan(q[1]):
        return np.nan
    return float(q[1] / q[0])


def q_ratio_from_kernel(kernel, n_v=2048, v_pad_factor=3.0, n_zeros=3):
    """
    Compute q_2/q_1 from a SpectralTemporalKernel via its τ=0 slice.

    K_S(τ=0, Δv) is the equal-time spectral covariance.  Its Fourier
    transform recovers the broadening function power spectrum, whose
    zeros encode vsini and differential rotation.

    Parameters
    ----------
    kernel : SpectralTemporalKernel
        Kernel instance.
    n_v : int
        Number of velocity points for the FT grid.
    v_pad_factor : float
        Velocity grid extends to ±v_pad_factor × vsini.
    n_zeros : int
        Number of FT zeros to find.

    Returns
    -------
    ratio : float
        q_2/q_1 ratio.
    sigma : ndarray
        Fourier variable.
    G_hat : ndarray
        Normalized |FT| of the τ=0 slice.
    """
    K0 = kernel.kernel(np.array([0.0]))  # (1, N_dv)
    K0_slice = np.array(K0[0])
    dv_grid = np.array(kernel.dv_grid)

    v_fine = np.linspace(dv_grid[0], dv_grid[-1], n_v)
    K0_fine = np.interp(v_fine, dv_grid, K0_slice)

    sigma, G_hat = broadening_function_ft(v_fine, K0_fine)
    ratio = q_ratio(sigma, G_hat, n_zeros=n_zeros)

    return ratio, sigma, G_hat


def rigid_rotation_q_ratio(vsini, epsilon=0.6, n_v=4096):
    """
    Theoretical q_2/q_1 for rigid rotation with linear limb darkening.

    This is the classical Reiners & Schmitt (2002) result, computed
    numerically from the Gray (2005) broadening kernel.

    Parameters
    ----------
    vsini : float
        Projected rotation velocity [km/s].
    epsilon : float
        Linear limb-darkening coefficient.
    n_v : int
        Number of velocity grid points.

    Returns
    -------
    ratio : float
        q_2/q_1 for rigid rotation.
    """
    v = np.linspace(-3 * vsini, 3 * vsini, n_v)
    G = np.array(rotational_broadening_kernel(v, vsini, epsilon=epsilon))
    sigma, G_hat = broadening_function_ft(v, G)
    return q_ratio(sigma, G_hat)
