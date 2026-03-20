"""Tests for src.lightcurve — LightcurveModel simulation."""

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
