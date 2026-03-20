"""Tests for src.params — hyperparameter validation and registry."""

import numpy as np
import pytest
from src.params import (
    resolve_hparam,
    BASE_REQUIRED_KEYS,
    KERNEL_HPARAM_KEYS,
    HPARAM_KEYS_WITH_NOISE,
)


class TestResolveHparam:
    def test_sigma_k_direct(self, default_hparam):
        hp = resolve_hparam(default_hparam)
        assert hp["sigma_k"] == 0.01
        assert hp["tau_spot"] == 1.0

    def test_physical_count_amplitude(self, physical_hparam):
        hp = resolve_hparam(physical_hparam)
        expected = np.sqrt(5) * (1 - 0.1) * 0.01 / np.pi
        np.testing.assert_allclose(hp["sigma_k"], expected)

    def test_physical_rate_amplitude(self):
        hp = resolve_hparam(dict(
            peq=10.0, kappa=0.2, inc=1.0, lspot=5.0, tau_spot=1.0,
            nspot_rate=0.5, fspot=0.0, alpha_max=0.1,
        ))
        expected = np.sqrt(0.5) * 1.0 * 0.01
        np.testing.assert_allclose(hp["sigma_k"], expected)

    def test_asymmetric_envelope(self, asymmetric_hparam):
        hp = resolve_hparam(asymmetric_hparam)
        assert hp["tau_em"] == 0.5
        assert hp["tau_dec"] == 1.5
        np.testing.assert_allclose(hp["tau_spot"], 1.0)

    def test_missing_base_keys_raises(self):
        with pytest.raises(ValueError, match="missing required"):
            resolve_hparam({"peq": 10.0})

    def test_not_dict_raises(self):
        with pytest.raises(TypeError, match="dict"):
            resolve_hparam([1, 2, 3])

    def test_no_envelope_match_raises(self):
        with pytest.raises(ValueError, match="No envelope"):
            resolve_hparam(dict(peq=10.0, kappa=0.2, inc=1.0, lspot=5.0,
                                sigma_k=0.01))

    def test_no_amplitude_match_raises(self):
        with pytest.raises(ValueError, match="No amplitude"):
            resolve_hparam(dict(peq=10.0, kappa=0.2, inc=1.0, lspot=5.0,
                                tau_spot=1.0))

    def test_preserves_extra_keys(self, default_hparam):
        default_hparam["custom_key"] = 42
        hp = resolve_hparam(default_hparam)
        assert hp["custom_key"] == 42


class TestConstants:
    def test_kernel_hparam_keys_order(self):
        assert KERNEL_HPARAM_KEYS == ("peq", "kappa", "inc", "lspot", "tau_spot", "sigma_k")

    def test_hparam_keys_with_noise(self):
        assert HPARAM_KEYS_WITH_NOISE == KERNEL_HPARAM_KEYS + ("sigma_n",)

    def test_base_required_keys(self):
        assert BASE_REQUIRED_KEYS == frozenset({"peq", "kappa", "inc", "lspot"})
