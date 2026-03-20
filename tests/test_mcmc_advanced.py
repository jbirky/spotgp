"""Tests for MCMCSampler and BlackJAXSampler advanced functionality."""

import numpy as np
import pytest
import jax.numpy as jnp

from spotgp.gp_solver import GPSolver
from spotgp.mcmc import MCMCSampler, BlackJAXSampler


class TestMCMCSamplerSummary:
    def test_summary_with_samples(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        sampler = MCMCSampler(gp)
        # Manually set fake samples
        rng = np.random.default_rng(42)
        sampler.samples = rng.standard_normal((50, 6)) * 0.01 + gp.theta0[None, :]
        result = sampler.summary()
        assert isinstance(result, dict)
        assert len(result) == 6

    def test_summary_keys_match_params(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        sampler = MCMCSampler(gp)
        rng = np.random.default_rng(42)
        sampler.samples = rng.standard_normal((50, 6)) * 0.01 + gp.theta0[None, :]
        result = sampler.summary()
        for key in gp.param_keys:
            assert key in result


class TestBlackJAXSamplerInit:
    def test_inherits_gp(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        sampler = BlackJAXSampler(gp)
        assert sampler.gp is gp
        assert sampler.n_params == gp.n_params

    def test_samples_initially_none(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        sampler = BlackJAXSampler(gp)
        assert sampler.samples is None
