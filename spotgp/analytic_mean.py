"""
Analytic GP mean function for stellar spot variability.

Computes E[F] = 1 - nspot_rate * c_spot * (integral Gamma dt) * <V_bar>_Phi

where c_spot = (1 - f_spot) * alpha_max^2 is the spot contrast-area product,
V_bar(Phi, I) = c_0(Phi) is the time-averaged visibility (zeroth Fourier
coefficient), and the latitude average <V_bar>_Phi integrates over p(Phi).
"""

import jax.numpy as jnp
import numpy as np

from .spot_model import SpotEvolutionModel
from .visibility import (
    VisibilityFunction, EdgeOnVisibilityFunction,
    _cn_general_jax, _gauss_legendre_grid,
)

__all__ = ["AnalyticMean"]


class AnalyticMean:
    """
    Analytic GP mean function for stellar spot variability.

    Parameters
    ----------
    model_or_hparam : SpotEvolutionModel or dict
        Spot evolution model (same interface as AnalyticKernel).
        Must be constructed with ``(nspot_rate, c_spot)`` or
        ``(nspot_rate, fspot, alpha_max)`` so that the physical
        amplitude parameters are available.
    n_lat : int
        Number of latitude quadrature points (default 64).
    quadrature : str
        ``"trapezoid"`` or ``"gauss-legendre"``.
    """

    def __init__(self, model_or_hparam, n_lat=64, quadrature="trapezoid"):
        if isinstance(model_or_hparam, SpotEvolutionModel):
            self.spot_model = model_or_hparam
        else:
            self.spot_model = SpotEvolutionModel.from_hparam(model_or_hparam)

        self.envelope = self.spot_model.envelope
        self.visibility = self.spot_model.visibility
        self.n_lat = n_lat
        self.quadrature = quadrature

        if self.spot_model.nspot_rate is None or self.spot_model.c_spot is None:
            raise ValueError(
                "AnalyticMean requires (nspot_rate, c_spot) or "
                "(nspot_rate, fspot, alpha_max) parameterization. "
                "sigma_k alone is insufficient for the mean function.")

        self.nspot_rate = self.spot_model.nspot_rate
        self.c_spot = self.spot_model.c_spot
        self.Gamma_integral = self.envelope.Gamma_integral()
        self.mean_visibility = self._compute_mean_visibility()
        self._mean_flux = self._compute_mean_flux()

    def _compute_mean_visibility(self):
        """Compute latitude-averaged mean visibility <V_bar>_Phi."""
        lat_dist = self.spot_model.latitude_distribution

        if isinstance(self.visibility, EdgeOnVisibilityFunction):
            return 2.0 / (np.pi ** 2)

        lat_range = lat_dist.lat_range

        if self.quadrature == "gauss-legendre":
            phi_grid, quad_weights = _gauss_legendre_grid(
                self.n_lat, lat_range[0], lat_range[1])
            phi_grid = jnp.array(phi_grid)
            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            weights = user_weights * jnp.array(quad_weights)
        else:
            phi_grid = jnp.linspace(lat_range[0], lat_range[1], self.n_lat)
            dphi = float(phi_grid[1] - phi_grid[0])
            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            weights = user_weights * dphi

        norm = float(jnp.sum(weights))

        c0_vals = jnp.array([
            float(_cn_general_jax(0, self.visibility.inc, float(phi)))
            for phi in phi_grid
        ])

        return float(jnp.sum(weights * c0_vals) / norm)

    def _compute_mean_flux(self):
        """E[F] = 1 - nspot_rate * c_spot * Gamma_integral * <V_bar>_Phi."""
        return 1.0 - (self.nspot_rate * self.c_spot
                       * self.Gamma_integral * self.mean_visibility)

    @property
    def mean_flux(self):
        """The expected flux E[F] as a scalar."""
        return self._mean_flux

    @property
    def mean_deficit(self):
        """The expected flux deficit 1 - E[F] (positive)."""
        return 1.0 - self._mean_flux

    def __call__(self, t=None):
        """Evaluate mean function. Returns scalar (constant in time)."""
        return self._mean_flux

    def __repr__(self):
        return (
            f"AnalyticMean(\n"
            f"  nspot_rate={self.nspot_rate},\n"
            f"  c_spot={self.c_spot},\n"
            f"  Gamma_integral={self.Gamma_integral:.4f},\n"
            f"  mean_visibility={self.mean_visibility:.6f},\n"
            f"  E[F]={self._mean_flux:.6f}\n)")
