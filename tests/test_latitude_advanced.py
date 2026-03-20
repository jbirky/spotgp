"""Tests for LatitudeDistributionFunction display and sympy methods."""

import numpy as np
import pytest

from spotgp.latitude import LatitudeDistributionFunction


class TestGetSympyDisplay:
    def test_display_false_returns_dict(self):
        pytest.importorskip("sympy")
        lat = LatitudeDistributionFunction()
        result = lat.get_sympy(display=False)
        assert "pdf" in result

    def test_display_true_with_status(self):
        pytest.importorskip("sympy")
        lat = LatitudeDistributionFunction()
        result = lat.get_sympy(display=True, status="default")
        assert "pdf" in result

    def test_display_true_without_status(self):
        pytest.importorskip("sympy")
        lat = LatitudeDistributionFunction()
        result = lat.get_sympy(display=True)
        assert "pdf" in result


class TestCustomLatitudeSympyPdf:
    def test_subclass_returns_none(self):
        """A subclass that doesn't override sympy_pdf returns 1 (base)."""
        pytest.importorskip("sympy")

        class Custom(LatitudeDistributionFunction):
            def __call__(self, phi):
                return np.cos(phi)

        lat = Custom()
        import sympy as sp
        assert lat.sympy_pdf() == sp.Integer(1)
