"""Tests for GPSolver accepting TimeSeriesData."""

import numpy as np
import pytest

from spotgp.gp_solver import GPSolver
from spotgp.observations import TimeSeriesData


class TestGPSolverTimeSeriesData:
    def test_from_timeseries_data(self, default_hparam):
        rng = np.random.default_rng(42)
        data = TimeSeriesData(
            np.linspace(0, 20, 30),
            1.0 + 0.005 * rng.standard_normal(30),
            0.001,
        )
        gp = GPSolver(data, default_hparam)
        assert gp.N == data.N
        assert gp.data is data

    def test_log_likelihood_matches_legacy(self, default_hparam):
        rng = np.random.default_rng(42)
        x = np.linspace(0, 20, 30)
        y = 1.0 + 0.005 * rng.standard_normal(30)
        yerr = np.full(30, 0.001)

        # Both paths normalize by default, so results should match
        data = TimeSeriesData(x, y, yerr)
        gp_new = GPSolver(data, default_hparam)
        gp_old = GPSolver(x, y, yerr, default_hparam)

        ll_new = float(gp_new.log_likelihood())
        ll_old = float(gp_old.log_likelihood())
        np.testing.assert_allclose(ll_new, ll_old, rtol=1e-8)

    def test_compute_acf_delegates(self, default_hparam):
        rng = np.random.default_rng(42)
        data = TimeSeriesData(
            np.linspace(0, 20, 50),
            1.0 + 0.005 * rng.standard_normal(50),
            0.001,
        )
        gp = GPSolver(data, default_hparam)
        lags, acf = gp.compute_acf(n_bins=20)
        assert len(lags) == 20
        assert len(acf) == 20

    def test_compute_acf_unnormalized(self, default_hparam):
        rng = np.random.default_rng(42)
        data = TimeSeriesData(
            np.linspace(0, 20, 50),
            1.0 + 0.005 * rng.standard_normal(50),
            0.001,
        )
        gp = GPSolver(data, default_hparam)
        _, acf_norm = gp.compute_acf(n_bins=20, normalize=True)
        _, acf_unnorm = gp.compute_acf(n_bins=20, normalize=False)
        # Unnormalized should have larger absolute values
        assert np.max(np.abs(acf_unnorm)) != np.max(np.abs(acf_norm))
