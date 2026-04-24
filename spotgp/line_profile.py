"""
line_profile.py — Rotational broadening and quiescent line profile model.

Provides the rotationally broadened line profile G(v) following Gray (2005),
and the quiescent (spot-free) disk-integrated absorption line I_0(v) obtained
by convolving an intrinsic Gaussian local profile H(v) with G(v).

These are the building blocks for the bisector velocity span (bisector.py)
and Fourier transform zeros (ft_zeros.py) diagnostics derived from the
spectral-temporal kernel K_S(τ, Δv).

Reference: Gray, "The Observation and Analysis of Stellar Photospheres",
3rd ed., Chapter 18.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "rotational_broadening_kernel",
    "quiescent_line_profile",
    "local_line_profile",
    "convolve_profiles",
]


def rotational_broadening_kernel(v, vsini, epsilon=0.6):
    """
    Rotational broadening kernel G(v) with linear limb darkening.

    Gray (2005) Eq. 18.14:

        G(v) ∝ (2(1-ε)√(1-x²) + πε/2 (1-x²)) / (π v sin i (1 - ε/3))

    where x = v / (v sin i).

    Parameters
    ----------
    v : array_like
        Velocity grid [km/s].
    vsini : float
        Projected equatorial rotation velocity [km/s].
    epsilon : float
        Linear limb-darkening coefficient (default 0.6).

    Returns
    -------
    G : ndarray
        Broadening kernel, normalized so ∫ G dv = 1.
    """
    v = jnp.asarray(v, dtype=float)
    x = v / vsini
    x2 = x ** 2
    inside = jnp.abs(x) < 1.0

    term1 = 2.0 * (1.0 - epsilon) * jnp.sqrt(jnp.maximum(1.0 - x2, 0.0))
    term2 = 0.5 * jnp.pi * epsilon * jnp.maximum(1.0 - x2, 0.0)
    G_unnorm = jnp.where(inside, term1 + term2, 0.0)

    norm = jnp.pi * vsini * (1.0 - epsilon / 3.0)
    return G_unnorm / norm


def local_line_profile(v, sigma_H, depth=1.0):
    """
    Intrinsic local line profile H(v) — Gaussian absorption.

    H(v) = depth × exp(-v² / 2σ_H²)

    Parameters
    ----------
    v : array_like
        Velocity grid [km/s].
    sigma_H : float
        Intrinsic line width [km/s] (thermal + microturbulent).
    depth : float
        Central line depth (0 to 1).

    Returns
    -------
    H : ndarray
        Local line profile (emission-like; positive bump).
    """
    v = jnp.asarray(v, dtype=float)
    return depth * jnp.exp(-v ** 2 / (2.0 * sigma_H ** 2))


def convolve_profiles(v, f, g):
    """
    Convolve two profiles on the same velocity grid via FFT.

    Parameters
    ----------
    v : array_like
        Uniformly spaced velocity grid, assumed symmetric about 0.
    f, g : array_like
        Profiles to convolve (same length as v).

    Returns
    -------
    convolved : ndarray
        (f * g)(v), normalized by the grid spacing.
    """
    v = jnp.asarray(v)
    f = jnp.asarray(f)
    g = jnp.asarray(g)
    dv = v[1] - v[0]
    n = len(v)
    n_pad = 2 * n - 1

    F = jnp.fft.fft(f, n=n_pad)
    G = jnp.fft.fft(g, n=n_pad)
    full_conv = jnp.fft.ifft(F * G).real * dv

    start = n // 2
    return full_conv[start:start + n]


def quiescent_line_profile(v, vsini, sigma_H, epsilon=0.6, depth=1.0):
    """
    Quiescent (spot-free) disk-integrated absorption line profile.

    I_0(v) = 1 - (H * G)(v)

    where H is the intrinsic Gaussian local profile and G is the
    rotational broadening kernel. The result is an absorption line
    (minimum at line center, continuum = 1).

    Parameters
    ----------
    v : array_like
        Velocity grid [km/s]. Should be uniformly spaced and extend
        well beyond ±(vsini + 3σ_H).
    vsini : float
        Projected equatorial rotation velocity [km/s].
    sigma_H : float
        Intrinsic line width [km/s].
    epsilon : float
        Linear limb-darkening coefficient.
    depth : float
        Central depth of the local (unbroadened) line.

    Returns
    -------
    I0 : ndarray
        Disk-integrated line profile (absorption; values in [0, 1]).
    """
    v = jnp.asarray(v, dtype=float)
    H = local_line_profile(v, sigma_H, depth=depth)
    G = rotational_broadening_kernel(v, vsini, epsilon=epsilon)
    broadened = convolve_profiles(v, H, G)
    return 1.0 - broadened
