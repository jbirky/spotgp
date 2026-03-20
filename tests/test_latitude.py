"""Tests for spotgp.latitude — LatitudeDistributionFunction."""

import numpy as np
import pytest

from spotgp.latitude import LatitudeDistributionFunction


class TestLatitudeDistributionFunction:
    def test_default_range(self):
        lat = LatitudeDistributionFunction()
        assert lat.lat_range == (-np.pi / 2, np.pi / 2)

    def test_uniform_value(self):
        lat = LatitudeDistributionFunction()
        assert lat(0.0) == 1.0
        assert lat(1.0) == 1.0

    def test_repr(self):
        lat = LatitudeDistributionFunction()
        r = repr(lat)
        assert "LatitudeDistributionFunction" in r
        assert "lat_range" in r


class TestCustomLatitude:
    def test_subclass_lat_range(self):
        class NarrowBand(LatitudeDistributionFunction):
            @property
            def lat_range(self):
                return (-np.pi / 6, np.pi / 6)

        lat = NarrowBand()
        assert lat.lat_range == (-np.pi / 6, np.pi / 6)

    def test_subclass_call(self):
        class GaussianLat(LatitudeDistributionFunction):
            def __init__(self, sigma=np.pi / 6):
                self.sigma = sigma
            def __call__(self, phi):
                return np.exp(-0.5 * (phi / self.sigma) ** 2)

        lat = GaussianLat(sigma=np.pi / 6)
        # Peak at equator
        assert lat(0.0) == pytest.approx(1.0)
        # Decays away from equator
        assert lat(np.pi / 4) < lat(0.0)


class TestSympyPdf:
    def test_default_returns_one(self):
        pytest.importorskip("sympy")
        lat = LatitudeDistributionFunction()
        expr = lat.sympy_pdf()
        import sympy as sp
        assert expr == sp.Integer(1)

    def test_get_sympy_returns_dict(self):
        pytest.importorskip("sympy")
        lat = LatitudeDistributionFunction()
        result = lat.get_sympy(display=False)
        assert "pdf" in result
