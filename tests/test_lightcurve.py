"""Tests for src.lightcurve — LightcurveModel simulation."""

import warnings

import numpy as np
import pytest

from spotgp.lightcurve import LightcurveModel, compute_sigmak
from spotgp.envelope import TrapezoidSymmetricEnvelope
from spotgp.spot_model import SpotEvolutionModel, VisibilityFunction


class TestComputeSigmak:
    def test_basic(self):
        sk = compute_sigmak(nspot_rate=4.0, alpha_max=0.1, fspot=0.0)
        expected = np.sqrt(4.0) * 1.0 * 0.01
        np.testing.assert_allclose(sk, expected)

    def test_with_fspot(self):
        sk = compute_sigmak(nspot_rate=4.0, alpha_max=0.1, fspot=0.5)
        expected = np.sqrt(4.0) * 0.5 * 0.01
        np.testing.assert_allclose(sk, expected)

    def test_zero_rate(self):
        sk = compute_sigmak(nspot_rate=0.0, alpha_max=0.1)
        assert sk == 0.0


class TestLightcurveModel:
    def test_basic_init(self):
        np.random.seed(42)
        lc = LightcurveModel(
            peq=10.0, kappa=0.0, inc=np.pi / 2, nspot=3,
            tau_spot=1.0, tem=0.5, tdec=1.0, alpha_max=0.1,
            fspot=0.0, lspot=5.0, tsim=20, tsamp=0.5,
        )
        assert hasattr(lc, "flux")
        assert len(lc.flux) > 0
        assert len(lc.t) == len(lc.flux)

    def test_flux_close_to_one_for_small_spots(self):
        np.random.seed(42)
        lc = LightcurveModel(
            peq=10.0, kappa=0.0, inc=np.pi / 2, nspot=1,
            tau_spot=0.5, tem=0.5, tdec=0.5, alpha_max=0.01,
            fspot=0.0, lspot=3.0, tsim=10, tsamp=0.5,
        )
        assert np.all(np.abs(lc.flux - 1.0) < 0.1)

    def test_from_spot_model(self):
        np.random.seed(42)
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        model = SpotEvolutionModel(
            envelope=env, visibility=vis,
            nspot_rate=0.5, alpha_max=0.1, fspot=0.0,
        )
        lc = LightcurveModel.from_spot_model(model, nspot=5, tsim=20, tsamp=0.5)
        assert hasattr(lc, "flux")
        assert len(lc.flux) > 0

    def test_from_spot_model_warns_when_sigma_k_ignored(self):
        np.random.seed(42)
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        # alpha_max falls back to 0.1, so the data have sqrt(0.35) * 0.01
        model = SpotEvolutionModel(envelope=env, visibility=vis, sigma_k=0.005)
        with pytest.warns(UserWarning, match="does not use sigma_k"):
            LightcurveModel.from_spot_model(model, nspot_rate=0.35, tsim=20, tsamp=0.5)

    def test_from_spot_model_consistent_physical_params_no_warning(self):
        np.random.seed(42)
        env = TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=1.0)
        vis = VisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        model = SpotEvolutionModel(envelope=env, visibility=vis,
                                   nspot_rate=0.35, alpha_max=0.08, fspot=0.2)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lc = LightcurveModel.from_spot_model(model, nspot_rate=0.35,
                                                 tsim=20, tsamp=0.5)
        assert (lc.alpha_max, lc.fspot) == (0.08, 0.2)

    def test_nspot_rate_counts_over_emergence_window(self):
        np.random.seed(42)
        lc = LightcurveModel(nspot_rate=0.4, tsim=100, lspot=10.0, tem=2.0,
                             tdec=3.0, tsamp=1.0)
        lo, hi = lc._tmax_window()
        assert hi - lo == pytest.approx(100 + 10 + 2 + 3)
        assert lc.nspot == 46
        assert np.all((lc.tmax >= lo) & (lc.tmax <= hi))

    def test_simulated_variance_matches_kernel(self):
        """Ensemble variance equals the analytic K(0) at sigma_k =
        sqrt(nspot_rate) alpha_max^2, less the fixed-count term mu^2 / N.

        tsim = 60 against a 30-day spot lifetime, so counting only
        rate * tsim spots would come out near 2/3 of the kernel.
        """
        from spotgp import AnalyticKernel
        model = SpotEvolutionModel(
            envelope=TrapezoidSymmetricEnvelope(lspot=20.0, tau_spot=5.0),
            visibility=VisibilityFunction(peq=5.0, kappa=0.0, inc=np.pi / 2,
                                          harmonics=tuple(range(11))),
            nspot_rate=0.5, fspot=0.0, alpha_max=0.1)
        K0 = float(np.asarray(AnalyticKernel(
            model, n_lat=64, quadrature="gauss-legendre").kernel(np.array([0.0])))[0])
        xs = []
        for seed in range(400):
            np.random.seed(seed)
            lc = LightcurveModel.from_spot_model(model, nspot_rate=0.5,
                                                 tsim=60.0, tsamp=1.0)
            xs.append(-np.sum(lc.dspots, axis=0))
        xs = np.array(xs)
        mu = xs.mean()
        var = np.mean((xs - mu) ** 2)
        assert var / (K0 - mu ** 2 / lc.nspot) == pytest.approx(1.0, abs=0.1)

    def test_from_hparam(self, default_hparam):
        np.random.seed(42)
        lc = LightcurveModel.from_hparam(
            default_hparam, nspot=3, tsim=20, tsamp=0.5,
        )
        assert hasattr(lc, "flux")
        assert len(lc.flux) > 0

    def test_flux_length_matches_time_array(self):
        np.random.seed(42)
        lc = LightcurveModel(
            peq=5.0, kappa=0.0, inc=np.pi / 2, nspot=2,
            tau_spot=1.0, alpha_max=0.05, lspot=3.0,
            tsim=10, tsamp=0.25,
        )
        expected_len = len(np.arange(0, 10, 0.25))
        assert len(lc.t) == expected_len
        assert len(lc.flux) == expected_len
