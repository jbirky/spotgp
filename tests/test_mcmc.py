"""Tests for src.mcmc — MCMCSampler and BlackJAXSampler."""

import numpy as np
import pytest

from spotgp.gp_solver import GPSolver
from spotgp.mcmc import MCMCSampler, BlackJAXSampler


class TestMCMCSampler:
    def test_init(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        sampler = MCMCSampler(gp)
        assert sampler.n_params == 6
        assert sampler.param_keys == gp.param_keys

    def test_requires_gpsolver(self):
        with pytest.raises(TypeError):
            MCMCSampler("not a gp")

    def test_no_samples_raises(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        sampler = MCMCSampler(gp)
        with pytest.raises(RuntimeError):
            sampler.summary()

    def test_samples_initially_none(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        sampler = MCMCSampler(gp)
        assert sampler.samples is None


class TestBlackJAXSampler:
    def test_init(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        sampler = BlackJAXSampler(gp)
        assert sampler.n_params == 6

    def test_inherits_mcmc_sampler(self, default_hparam, synthetic_data):
        x, y, yerr = synthetic_data
        gp = GPSolver(x, y, yerr, default_hparam)
        sampler = BlackJAXSampler(gp)
        assert isinstance(sampler, MCMCSampler)
