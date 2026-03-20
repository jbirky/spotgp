"""Shared fixtures for spotgp tests."""

import numpy as np
import pytest


@pytest.fixture
def default_hparam():
    """Standard hyperparameter dict with sigma_k amplitude."""
    return dict(
        peq=10.0,
        kappa=0.2,
        inc=np.pi / 4,
        lspot=5.0,
        tau_spot=1.0,
        sigma_k=0.01,
    )


@pytest.fixture
def physical_hparam():
    """Hyperparameter dict using nspot/fspot/alpha_max amplitude."""
    return dict(
        peq=10.0,
        kappa=0.2,
        inc=np.pi / 4,
        lspot=5.0,
        tau_spot=1.0,
        nspot=5,
        fspot=0.1,
        alpha_max=0.1,
    )


@pytest.fixture
def asymmetric_hparam():
    """Hyperparameter dict with asymmetric envelope."""
    return dict(
        peq=10.0,
        kappa=0.2,
        inc=np.pi / 4,
        lspot=5.0,
        tau_em=0.5,
        tau_dec=1.5,
        sigma_k=0.01,
    )


@pytest.fixture
def synthetic_data(default_hparam):
    """Small synthetic dataset for GP tests."""
    rng = np.random.default_rng(42)
    N = 30
    x = np.linspace(0, 20, N)
    y = 1.0 + 0.005 * rng.standard_normal(N)
    yerr = np.full(N, 0.001)
    return x, y, yerr
