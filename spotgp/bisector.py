"""
bisector.py — Bisector velocity span (BVS) from line profiles and K_S.

The bisector of a spectral line is the locus of midpoints between the
blue and red wings at each flux level.  The bisector velocity span (BVS)
is the difference in bisector velocity between two flux levels, and is
a standard activity indicator correlated with starspot-induced RV jitter.

This module provides:
  1. Direct bisector computation from an observed line profile.
  2. Linearized BVS covariance derived from the spectral-temporal
     kernel K_S(τ, Δv), following the hierarchy in Fig. 1 of Paper III.

Reference: Gray (2005) Ch. 21; Queloz et al. (2001) for BVS--RV correlation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

__all__ = [
    "compute_bisector",
    "bisector_velocity_span",
    "bvs_sensitivity_kernel",
    "bvs_covariance",
]


def compute_bisector(v, profile, n_levels=20, flux_range=(0.2, 0.9)):
    """
    Compute the bisector of a spectral line profile.

    At each flux level F between flux_range[0] and flux_range[1] (as a
    fraction of the continuum-to-core depth), find the blue and red wing
    velocities where profile(v) = F, and return their midpoint.

    Parameters
    ----------
    v : array_like, shape (N,)
        Velocity grid [km/s].
    profile : array_like, shape (N,)
        Line profile (absorption; 1 = continuum, min = core).
    n_levels : int
        Number of flux levels.
    flux_range : tuple of float
        (lower, upper) fraction of the line depth at which to evaluate
        the bisector.  0 = core, 1 = continuum.

    Returns
    -------
    flux_levels : ndarray, shape (n_levels,)
        Absolute flux values at which the bisector was evaluated.
    bisector : ndarray, shape (n_levels,)
        Bisector velocities [km/s] at each flux level.
    """
    v = np.asarray(v, dtype=float)
    profile = np.asarray(profile, dtype=float)

    core_flux = np.min(profile)
    cont_flux = np.max(profile)
    depth = cont_flux - core_flux

    frac_levels = np.linspace(flux_range[0], flux_range[1], n_levels)
    flux_levels = core_flux + frac_levels * depth

    core_idx = np.argmin(profile)
    v_blue = v[:core_idx + 1]
    p_blue = profile[:core_idx + 1]
    v_red = v[core_idx:]
    p_red = profile[core_idx:]

    bisector = np.zeros(n_levels)
    for i, F in enumerate(flux_levels):
        v_b = np.interp(F, p_blue[::-1], v_blue[::-1])
        v_r = np.interp(F, p_red, v_red)
        bisector[i] = 0.5 * (v_b + v_r)

    return flux_levels, bisector


def bisector_velocity_span(v, profile, top=0.8, bottom=0.3, n_levels=50):
    """
    Bisector velocity span: BVS = v_bis(top) - v_bis(bottom).

    Parameters
    ----------
    v : array_like
        Velocity grid [km/s].
    profile : array_like
        Line profile (absorption).
    top : float
        Upper flux fraction (closer to continuum), in (0, 1).
    bottom : float
        Lower flux fraction (closer to core), in (0, 1).
    n_levels : int
        Number of bisector levels for interpolation.

    Returns
    -------
    bvs : float
        Bisector velocity span [km/s].
    """
    flux_levels, bisector = compute_bisector(
        v, profile, n_levels=n_levels,
        flux_range=(min(bottom, top) - 0.05, max(bottom, top) + 0.05),
    )
    depth = np.max(profile) - np.min(profile)
    abs_top = np.min(profile) + top * depth
    abs_bottom = np.min(profile) + bottom * depth

    v_top = np.interp(abs_top, flux_levels, bisector)
    v_bottom = np.interp(abs_bottom, flux_levels, bisector)
    return float(v_top - v_bottom)


def bvs_sensitivity_kernel(v, profile, top=0.8, bottom=0.3, dI=None):
    """
    Linearized BVS sensitivity kernel: ∂BVS/∂I(v).

    Computes how a small perturbation δI(v) at each velocity changes the
    BVS, by finite difference.  This connects the spectral-temporal
    kernel to the BVS covariance:

        Cov[BVS, BVS'] = Σ_i Σ_j  W(v_i) W(v_j) K_S(τ, v_j - v_i)

    where W(v) = ∂BVS/∂I(v).

    Parameters
    ----------
    v : array_like, shape (N,)
        Velocity grid [km/s].
    profile : array_like, shape (N,)
        Quiescent line profile.
    top, bottom : float
        BVS flux fractions.
    dI : float
        Perturbation amplitude for finite difference.  If None,
        uses 1e-3 × line depth.

    Returns
    -------
    W : ndarray, shape (N,)
        Sensitivity kernel [km/s per unit flux perturbation].
    """
    v = np.asarray(v, dtype=float)
    profile = np.asarray(profile, dtype=float)
    N = len(v)

    if dI is None:
        depth = np.max(profile) - np.min(profile)
        dI = max(1e-3 * depth, 1e-6)

    bvs0 = bisector_velocity_span(v, profile, top=top, bottom=bottom)
    W = np.zeros(N)

    for k in range(N):
        perturbed = profile.copy()
        perturbed[k] += dI
        bvs_k = bisector_velocity_span(v, perturbed, top=top, bottom=bottom)
        W[k] = (bvs_k - bvs0) / dI

    return W


def bvs_covariance(lag, kernel, v_grid, profile, top=0.8, bottom=0.3):
    """
    BVS autocovariance from the spectral-temporal kernel K_S.

    Cov[BVS(t), BVS(t+τ)] = Σ_i Σ_j W(v_i) W(v_j) K_S(τ, v_j - v_i) dv²

    where W = ∂BVS/∂I is the sensitivity kernel evaluated on the
    quiescent profile.

    Parameters
    ----------
    lag : array_like
        Time lags [days].
    kernel : SpectralTemporalKernel
        Spectral-temporal kernel instance.
    v_grid : array_like
        Velocity grid on which the quiescent profile and K_S are defined.
    profile : array_like
        Quiescent line profile on v_grid.
    top, bottom : float
        BVS flux fractions.

    Returns
    -------
    C_BVS : ndarray, shape (len(lag),)
        BVS autocovariance.
    """
    v_grid = np.asarray(v_grid, dtype=float)
    dv = v_grid[1] - v_grid[0]
    lag = np.atleast_1d(lag)

    W = bvs_sensitivity_kernel(v_grid, profile, top=top, bottom=bottom)

    K_S = kernel.kernel(lag)  # (M, N_dv_kernel)
    dv_kernel = kernel.dv_grid

    dv_matrix = v_grid[None, :] - v_grid[:, None]  # (N_v, N_v)
    dv_flat = dv_matrix.ravel()

    C_BVS = np.zeros(len(lag))
    for t_idx in range(len(lag)):
        K_row = np.array(K_S[t_idx])  # (N_dv_kernel,)
        K_flat = np.interp(dv_flat, dv_kernel, K_row)
        K_matrix = K_flat.reshape(dv_matrix.shape)
        C_BVS[t_idx] = W @ K_matrix @ W * dv ** 2

    return C_BVS
