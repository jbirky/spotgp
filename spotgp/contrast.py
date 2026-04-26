"""
contrast.py — Wavelength-dependent spot contrast from blackbody ratio.

Provides JAX-compatible functions for computing the spot-to-photosphere
flux ratio f_spot(λ) = B_λ(T_spot) / B_λ(T_phot) and the resulting
contrast factor c(λ) = 1 - f_spot(λ) used by the multi-band kernel.
"""
import jax.numpy as jnp

__all__ = ["spot_contrast", "contrast_factor", "contrast_matrix"]

# CGS constants
_h = 6.62607015e-27    # Planck constant [erg·s]
_c = 2.99792458e10     # speed of light [cm/s]
_kB = 1.380649e-16     # Boltzmann constant [erg/K]


def spot_contrast(lam_angstrom, T_spot, T_phot):
    """
    Spot-to-photosphere flux ratio f_spot(λ) = B_λ(T_spot) / B_λ(T_phot).

    Uses expm1 ratio for numerical stability (avoids overflow in the
    Planck exponential for short wavelengths / low temperatures).

    Parameters
    ----------
    lam_angstrom : array_like
        Wavelength(s) in Angstroms.
    T_spot : float
        Spot temperature [K].
    T_phot : float
        Photosphere temperature [K].

    Returns
    -------
    f_spot : array_like
        Flux ratio B_λ(T_spot) / B_λ(T_phot).  Approaches 0 in the
        Wien limit (short λ) and 1 in the Rayleigh-Jeans limit (long λ).
    """
    lam_cm = lam_angstrom * 1e-8
    x_spot = _h * _c / (lam_cm * _kB * T_spot)
    x_phot = _h * _c / (lam_cm * _kB * T_phot)
    return jnp.expm1(x_phot) / jnp.expm1(x_spot)


def contrast_factor(lam_angstrom, T_spot, T_phot):
    """
    Spot contrast factor c(λ) = 1 - f_spot(λ).

    This is the fractional flux deficit per unit projected spot area at
    wavelength λ.  Enters the multi-band kernel as a multiplicative
    scaling: K(τ; λ_i, λ_j) = c(λ_i) · c(λ_j) · K_geom(τ).

    Parameters
    ----------
    lam_angstrom : array_like
        Wavelength(s) in Angstroms.
    T_spot : float
        Spot temperature [K].
    T_phot : float
        Photosphere temperature [K].

    Returns
    -------
    c : array_like
        Contrast factor.  Near 1 for short λ (spots are dark), near 0
        for long λ (spots vanish in Rayleigh-Jeans tail).
    """
    return 1.0 - spot_contrast(lam_angstrom, T_spot, T_phot)


def contrast_matrix(band_wavelengths, temperatures, T_phot, weights=None):
    """
    Contrast matrix for M temperature components.

    C[p,q] = sum_m w_m * c(lambda_p, T_m) * c(lambda_q, T_m)

    For M=1 (spot-only), this is the rank-1 outer product c*c^T.
    For M=2 (spots + faculae), this is a rank-2 matrix with
    opposite-sign contributions from cool spots and hot faculae.

    Parameters
    ----------
    band_wavelengths : array_like, shape (N_lambda,)
        Effective wavelengths in Angstroms.
    temperatures : array_like, shape (M,)
        Component temperatures [K].
    T_phot : float
        Photospheric effective temperature [K].
    weights : array_like, shape (M,), optional
        Component weights.  Default: all ones.

    Returns
    -------
    C : jnp.ndarray, shape (N_lambda, N_lambda)
        Contrast matrix.
    """
    band_wavelengths = jnp.atleast_1d(jnp.asarray(band_wavelengths, dtype=jnp.float64))
    temperatures = jnp.atleast_1d(jnp.asarray(temperatures, dtype=jnp.float64))
    M = temperatures.shape[0]
    if weights is None:
        weights = jnp.ones(M, dtype=jnp.float64)
    else:
        weights = jnp.asarray(weights, dtype=jnp.float64)

    c_vectors = jnp.stack([
        contrast_factor(band_wavelengths, temperatures[m], T_phot)
        for m in range(M)
    ])
    return jnp.einsum('m,mi,mj->ij', weights, c_vectors, c_vectors)
