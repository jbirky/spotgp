"""
joint_gp.py — Joint photometric-spectroscopic GP solver.

Constructs the block-structured covariance matrix for simultaneous
inference on photometric lightcurve and spectral time-series data,
using shared starspot hyperparameters.

The joint covariance has block structure:

    K = [ K_phot,phot    K_phot,spec  ]
        [ K_spec,phot    K_spec,spec  ]

where all blocks share the same underlying kernel hyperparameters
θ = {P_eq, κ, I, ℓ_spot, τ_spot, ...}.

Reference: Paper III, Section 8 (Joint Photometric-Spectroscopic Inference).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

try:
    from .analytic_kernel import AnalyticKernel
    from .spectral_temporal_kernel import SpectralTemporalKernel
    from .params import resolve_hparam
except ImportError:
    from analytic_kernel import AnalyticKernel
    from spectral_temporal_kernel import SpectralTemporalKernel
    from params import resolve_hparam

__all__ = ["JointCovarianceBuilder"]


class JointCovarianceBuilder:
    """
    Build the joint photometric-spectroscopic covariance matrix.

    Given photometric times t_phot and spectroscopic (time, velocity)
    pairs, constructs the full block covariance from a shared
    SpectralTemporalKernel.

    Parameters
    ----------
    spectral_kernel : SpectralTemporalKernel
        Spectral-temporal kernel instance (carries all hyperparameters).
    photometric_kernel : AnalyticKernel or None
        Photometric kernel.  If None, derived from the spectral kernel
        by integrating over velocity.
    """

    def __init__(self, spectral_kernel, photometric_kernel=None):
        self.spectral_kernel = spectral_kernel
        if photometric_kernel is None:
            self.photometric_kernel = AnalyticKernel(
                spectral_kernel.hparam,
                n_harmonics=spectral_kernel.n_harmonics,
                n_lat=spectral_kernel.n_lat,
            )
        else:
            self.photometric_kernel = photometric_kernel

    def build_phot_phot(self, t_phot, sigma_phot=None):
        """
        Photometric auto-covariance block.

        Parameters
        ----------
        t_phot : array_like, shape (N_p,)
            Photometric observation times.
        sigma_phot : array_like or None
            Photometric noise (added to diagonal).

        Returns
        -------
        K_pp : ndarray, shape (N_p, N_p)
        """
        t_phot = np.asarray(t_phot)
        lag_matrix = np.abs(np.subtract.outer(t_phot, t_phot))
        K_pp = np.array(self.photometric_kernel.kernel(lag_matrix))

        if sigma_phot is not None:
            sigma_phot = np.asarray(sigma_phot)
            K_pp += np.diag(sigma_phot ** 2)

        return K_pp

    def build_spec_spec(self, t_spec, v_indices=None, sigma_spec=None):
        """
        Spectroscopic auto-covariance block.

        For spectral data at times t_spec and velocity bins given by
        the kernel's dv_grid, the covariance between observation
        (t_i, v_a) and (t_j, v_b) is K_S(|t_i - t_j|, v_b - v_a).

        Parameters
        ----------
        t_spec : array_like, shape (N_s,)
            Spectroscopic observation times.
        v_indices : array_like or None
            Indices into dv_grid to use.  If None, uses all.
        sigma_spec : float or array_like or None
            Spectroscopic noise (scalar or per-observation).

        Returns
        -------
        K_ss : ndarray, shape (N_s * N_v, N_s * N_v)
        """
        t_spec = np.asarray(t_spec)
        N_s = len(t_spec)
        dv_grid = self.spectral_kernel.dv_grid

        if v_indices is not None:
            N_v = len(v_indices)
            dv_sub = dv_grid[v_indices]
        else:
            N_v = len(dv_grid)
            dv_sub = dv_grid

        lag_unique = np.unique(np.abs(np.subtract.outer(t_spec, t_spec)))
        K_S_cache = {}
        for lag_val in lag_unique:
            K_row = self.spectral_kernel.kernel(np.array([lag_val]))[0]
            K_S_cache[lag_val] = np.array(K_row)

        dim = N_s * N_v
        K_ss = np.zeros((dim, dim))

        for i in range(N_s):
            for j in range(N_s):
                lag_val = abs(t_spec[i] - t_spec[j])
                K_row = K_S_cache.get(lag_val)
                if K_row is None:
                    K_row = np.array(
                        self.spectral_kernel.kernel(np.array([lag_val]))[0])

                for a in range(N_v):
                    for b in range(N_v):
                        delta_v = dv_sub[b] - dv_sub[a]
                        K_val = np.interp(delta_v, dv_grid, K_row)
                        K_ss[i * N_v + a, j * N_v + b] = K_val

        if sigma_spec is not None:
            sigma_spec = np.atleast_1d(sigma_spec)
            if sigma_spec.size == 1:
                K_ss += sigma_spec[0] ** 2 * np.eye(dim)
            else:
                K_ss += np.diag(np.asarray(sigma_spec).ravel() ** 2)

        return K_ss

    def build_phot_spec(self, t_phot, t_spec, v_indices=None):
        """
        Cross-covariance block: photometry × spectroscopy.

        K_×(τ, v) = Cov[δF(t_phot), δS(v, t_spec)]

        Parameters
        ----------
        t_phot : array_like, shape (N_p,)
        t_spec : array_like, shape (N_s,)
        v_indices : array_like or None

        Returns
        -------
        K_ps : ndarray, shape (N_p, N_s * N_v)
        """
        t_phot = np.asarray(t_phot)
        t_spec = np.asarray(t_spec)
        N_p = len(t_phot)
        N_s = len(t_spec)
        dv_grid = self.spectral_kernel.dv_grid

        if v_indices is not None:
            N_v = len(v_indices)
            dv_sub = dv_grid[v_indices]
        else:
            N_v = len(dv_grid)
            dv_sub = dv_grid

        K_ps = np.zeros((N_p, N_s * N_v))

        for i in range(N_p):
            for j in range(N_s):
                lag = np.array([abs(t_phot[i] - t_spec[j])])
                K_cross = self.spectral_kernel.cross_covariance_phot_spec(lag)
                K_row = np.array(K_cross[0])

                for b in range(N_v):
                    K_ps[i, j * N_v + b] = np.interp(dv_sub[b], dv_grid, K_row)

        return K_ps

    def build_joint_covariance(self, t_phot, t_spec,
                                sigma_phot=None, sigma_spec=None,
                                v_indices=None, jitter=1e-10):
        """
        Full joint covariance matrix.

        Parameters
        ----------
        t_phot : array_like, shape (N_p,)
        t_spec : array_like, shape (N_s,)
        sigma_phot : array_like or None
        sigma_spec : float or array_like or None
        v_indices : array_like or None
        jitter : float
            Small diagonal regularization for numerical stability.

        Returns
        -------
        K : ndarray, shape (N_p + N_s*N_v, N_p + N_s*N_v)
        """
        K_pp = self.build_phot_phot(t_phot, sigma_phot)
        K_ss = self.build_spec_spec(t_spec, v_indices, sigma_spec)
        K_ps = self.build_phot_spec(t_phot, t_spec, v_indices)

        K = np.block([
            [K_pp, K_ps],
            [K_ps.T, K_ss],
        ])

        if jitter > 0:
            K += jitter * np.eye(K.shape[0])

        return K

    def joint_log_likelihood(self, t_phot, y_phot, yerr_phot,
                              t_spec, y_spec, yerr_spec,
                              v_indices=None, mean_phot=1.0, mean_spec=0.0):
        """
        Joint GP log-likelihood for photometric + spectroscopic data.

        Parameters
        ----------
        t_phot : array_like, shape (N_p,)
        y_phot : array_like, shape (N_p,)
        yerr_phot : array_like, shape (N_p,)
        t_spec : array_like, shape (N_s,)
        y_spec : array_like, shape (N_s * N_v,)
            Vectorized spectral residuals.
        yerr_spec : float or array_like
        v_indices : array_like or None
        mean_phot : float
        mean_spec : float

        Returns
        -------
        logL : float
        """
        K = self.build_joint_covariance(
            t_phot, t_spec,
            sigma_phot=yerr_phot,
            sigma_spec=yerr_spec,
            v_indices=v_indices,
        )

        y = np.concatenate([
            np.asarray(y_phot) - mean_phot,
            np.asarray(y_spec) - mean_spec,
        ])

        N = len(y)
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            logdet = 2.0 * np.sum(np.log(np.diag(L)))
            logL = -0.5 * (y @ alpha + logdet + N * np.log(2 * np.pi))
        except np.linalg.LinAlgError:
            logL = -np.inf

        return float(logL)
