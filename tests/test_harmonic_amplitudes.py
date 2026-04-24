"""
Tests for harmonic_amplitudes.py — velocity-dependent harmonic amplitudes
A_n(Δv; Φ) for the spectral-temporal kernel.
"""
import numpy as np
import pytest

from spotgp.harmonic_amplitudes import (
    compute_h_m,
    compute_g_n,
    compute_harmonic_amplitudes,
    compute_harmonic_amplitudes_direct,
)
from spotgp.visibility import _cn_general_jax


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(params=[
    # (sigma_H, vsini, inc, phi, label)
    (3.0, 2.0, np.pi / 3, 0.2, "slow_rotator"),
    (3.0, 10.0, np.pi / 2, 0.0, "moderate_edge_on"),
    (2.0, 30.0, np.pi / 4, 0.3, "fast_rotator"),
    (5.0, 5.0, np.pi / 2, np.pi / 4, "broad_line"),
])
def stellar_params(request):
    sigma_H, vsini, inc, phi, label = request.param
    return dict(sigma_H=sigma_H, vsini=vsini, inc=inc, phi=phi, label=label)


# ── h_m parity tests ────────────────────────────────────────────────────────

class TestHmParity:
    """h_m(v) is real for even m and purely imaginary for odd m."""

    def test_even_m_real(self):
        v = np.linspace(-20, 20, 512)
        h = np.asarray(compute_h_m(v, a=8.0, sigma_H=3.0, n_theta=64))
        for m in [0, 2, 4]:
            np.testing.assert_allclose(
                h[:, m].imag, 0.0, atol=1e-14,
                err_msg=f"h_{m} should be real"
            )

    def test_odd_m_imaginary(self):
        v = np.linspace(-20, 20, 512)
        h = np.asarray(compute_h_m(v, a=8.0, sigma_H=3.0, n_theta=64))
        for m in [1, 3, 5]:
            np.testing.assert_allclose(
                h[:, m].real, 0.0, atol=1e-14,
                err_msg=f"h_{m} should be purely imaginary"
            )

    def test_reality_condition(self):
        """h_{-m} = h_m* (FFT reality condition)."""
        v = np.linspace(-20, 20, 512)
        n_theta = 64
        h = np.asarray(compute_h_m(v, a=8.0, sigma_H=3.0, n_theta=n_theta))
        for m in [1, 2, 3, 5]:
            neg_m_idx = n_theta - m
            np.testing.assert_allclose(
                h[:, neg_m_idx], h[:, m].conj(), atol=1e-13,
                err_msg=f"h_{{-{m}}} should equal h_{m}*"
            )

    def test_h0_positive(self):
        """h_0(v) = time-averaged profile, should be positive."""
        v = np.linspace(-20, 20, 512)
        h = np.asarray(compute_h_m(v, a=5.0, sigma_H=3.0, n_theta=64))
        assert np.all(h[:, 0].real >= -1e-15)


# ── Parseval identity ────────────────────────────────────────────────────────

class TestParseval:
    """∫ A_n(Δv) dΔv = c_n² · (∫ H dv)² = c_n² · 2π σ_H²."""

    def test_parseval_identity(self, stellar_params):
        sigma_H = stellar_params["sigma_H"]
        vsini = stellar_params["vsini"]
        inc = stellar_params["inc"]
        phi = stellar_params["phi"]
        n_harmonics = 3

        a = vsini * np.cos(phi)
        dv_max = 3 * (abs(a) + 4 * sigma_H)
        dv = np.linspace(-dv_max, dv_max, 2048)

        A_n = compute_harmonic_amplitudes(
            dv, phi, inc, sigma_H, vsini,
            n_harmonics=n_harmonics,
            n_v_internal=2048,
            v_pad_factor=6.0,
        )

        W_H_sq = 2 * np.pi * sigma_H ** 2

        for n in range(n_harmonics + 1):
            cn = float(_cn_general_jax(n, inc, phi))
            expected = cn ** 2 * W_H_sq
            integral = np.trapezoid(A_n[n].real, dv)
            if expected < 1e-15:
                assert abs(integral) < 1e-8, (
                    f"n={n}: expected ~0 but got {integral}"
                )
            else:
                np.testing.assert_allclose(
                    integral, expected, rtol=0.02,
                    err_msg=(
                        f"Parseval failed for n={n}, "
                        f"{stellar_params['label']}: "
                        f"got {integral:.6e}, expected {expected:.6e}"
                    ),
                )


# ── A_0 properties ───────────────────────────────────────────────────────────

class TestA0Properties:
    """A_0(Δv) is the time-averaged profile autocorrelation."""

    def test_a0_symmetric(self, stellar_params):
        """A_0(-Δv) = A_0(Δv) (real and symmetric)."""
        sigma_H = stellar_params["sigma_H"]
        vsini = stellar_params["vsini"]
        inc = stellar_params["inc"]
        phi = stellar_params["phi"]
        a = vsini * np.cos(phi)

        dv_max = 2 * (abs(a) + 4 * sigma_H)
        N = 512
        dv = np.linspace(-dv_max, dv_max, N)

        A_n = compute_harmonic_amplitudes(
            dv, phi, inc, sigma_H, vsini, n_harmonics=0,
        )

        np.testing.assert_allclose(
            A_n[0, :], A_n[0, ::-1], rtol=0.02,
            err_msg=f"A_0 not symmetric for {stellar_params['label']}",
        )

    def test_a0_peak_at_zero_lag(self, stellar_params):
        """A_0(0) ≥ A_0(Δv) for all Δv (autocorrelation peak)."""
        sigma_H = stellar_params["sigma_H"]
        vsini = stellar_params["vsini"]
        inc = stellar_params["inc"]
        phi = stellar_params["phi"]
        a = vsini * np.cos(phi)

        dv_max = 2 * (abs(a) + 4 * sigma_H)
        dv = np.linspace(-dv_max, dv_max, 512)

        A_n = compute_harmonic_amplitudes(
            dv, phi, inc, sigma_H, vsini, n_harmonics=0,
        )

        mid = len(dv) // 2
        A0_peak = A_n[0, mid]
        assert A0_peak >= np.max(A_n[0]) - 1e-10 * abs(A0_peak)

    def test_a0_nonnegative_at_zero(self, stellar_params):
        """A_0(0) ≥ 0 (it's ||g_0||²)."""
        sigma_H = stellar_params["sigma_H"]
        vsini = stellar_params["vsini"]
        inc = stellar_params["inc"]
        phi = stellar_params["phi"]

        A_n = compute_harmonic_amplitudes(
            np.array([0.0]), phi, inc, sigma_H, vsini, n_harmonics=0,
        )
        assert A_n[0, 0] >= -1e-15


# ── Agreement between Fourier and direct methods ─────────────────────────────

class TestDirectAgreement:
    """The FFT-based and direct θ-integral methods should agree."""

    @pytest.mark.parametrize("params", [
        dict(sigma_H=3.0, vsini=5.0, inc=np.pi / 2, phi=0.0),
        dict(sigma_H=3.0, vsini=10.0, inc=np.pi / 3, phi=0.3),
        dict(sigma_H=5.0, vsini=2.0, inc=np.pi / 4, phi=0.5),
    ])
    def test_A0_agrees(self, params):
        """A_0 from both methods should match."""
        a = params["vsini"] * np.cos(params["phi"])
        dv_max = 2 * (abs(a) + 4 * params["sigma_H"])
        dv = np.linspace(-dv_max, dv_max, 128)

        A_fourier = compute_harmonic_amplitudes(
            dv, params["phi"], params["inc"],
            params["sigma_H"], params["vsini"],
            n_harmonics=0,
            n_v_internal=2048,
            v_pad_factor=6.0,
        )

        A_direct = compute_harmonic_amplitudes_direct(
            dv, params["phi"], params["inc"],
            params["sigma_H"], params["vsini"],
            n_harmonics=0,
            n_theta=512, n_omega_tau=256,
        )

        norm = np.max(np.abs(A_direct[0]))
        if norm > 1e-15:
            np.testing.assert_allclose(
                A_fourier[0] / norm, A_direct[0] / norm, atol=0.05,
                err_msg="A_0 mismatch between Fourier and direct methods",
            )

    @pytest.mark.parametrize("params", [
        dict(sigma_H=3.0, vsini=5.0, inc=np.pi / 2, phi=0.0),
        dict(sigma_H=4.0, vsini=8.0, inc=np.pi / 3, phi=0.2),
    ])
    def test_higher_harmonics_agree(self, params):
        """A_1, A_2 should agree between methods."""
        a = params["vsini"] * np.cos(params["phi"])
        dv_max = 2 * (abs(a) + 4 * params["sigma_H"])
        dv = np.linspace(-dv_max, dv_max, 128)

        A_fourier = compute_harmonic_amplitudes(
            dv, params["phi"], params["inc"],
            params["sigma_H"], params["vsini"],
            n_harmonics=2,
            n_v_internal=2048,
            v_pad_factor=6.0,
        )

        A_direct = compute_harmonic_amplitudes_direct(
            dv, params["phi"], params["inc"],
            params["sigma_H"], params["vsini"],
            n_harmonics=2,
            n_theta=512, n_omega_tau=256,
        )

        for n in range(3):
            cn = float(_cn_general_jax(n, params["inc"], params["phi"]))
            if abs(cn) < 1e-10:
                continue
            norm = np.max(np.abs(A_direct[n]))
            if norm < 1e-15:
                continue
            np.testing.assert_allclose(
                A_fourier[n] / norm, A_direct[n] / norm,
                atol=0.1,
                err_msg=f"A_{n} mismatch for {params}",
            )


# ── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_no_rotation(self):
        """vsini → 0: A_n should be concentrated at Δv = 0."""
        sigma_H = 3.0
        vsini = 0.01
        inc = np.pi / 2
        phi = 0.0
        dv = np.linspace(-20, 20, 256)

        A_n = compute_harmonic_amplitudes(
            dv, phi, inc, sigma_H, vsini, n_harmonics=2,
        )

        peak_idx = np.argmax(A_n[0])
        assert abs(dv[peak_idx]) < 1.0, "A_0 peak should be near Δv=0"

    def test_polar_spot(self):
        """Spot at pole (phi = π/2): a = vsini cos(π/2) = 0, no Doppler shift."""
        sigma_H = 3.0
        vsini = 10.0
        inc = np.pi / 3
        phi = np.pi / 2 - 0.01
        dv = np.linspace(-30, 30, 256)

        A_n = compute_harmonic_amplitudes(
            dv, phi, inc, sigma_H, vsini, n_harmonics=2,
        )

        peak_idx = np.argmax(A_n[0])
        assert abs(dv[peak_idx]) < 2.0

    def test_broad_line_limit(self):
        """σ_H ≫ a: profile perturbation barely affected by Doppler shift."""
        sigma_H = 30.0
        vsini = 5.0
        inc = np.pi / 2
        phi = 0.0
        a = vsini * np.cos(phi)
        dv = np.linspace(-100, 100, 512)

        A_n = compute_harmonic_amplitudes(
            dv, phi, inc, sigma_H, vsini,
            n_harmonics=2,
            n_v_internal=2048,
            v_pad_factor=5.0,
        )

        W_H_sq = 2 * np.pi * sigma_H ** 2
        for n in range(3):
            cn = float(_cn_general_jax(n, inc, phi))
            integral = np.trapezoid(A_n[n], dv)
            expected = cn ** 2 * W_H_sq
            if expected > 1e-10:
                np.testing.assert_allclose(
                    integral, expected, rtol=0.05,
                    err_msg=f"Parseval fails in broad-line limit for n={n}",
                )


# ── Photometric kernel recovery ─────────────────────────────────────────────

class TestPhotometricRecovery:
    """Velocity-integrated harmonic amplitudes should recover photometric
    kernel structure: Σ_n ∫A_n dΔv cos(nωτ) ∝ Σ_n |c_n|² cos(nωτ)."""

    def test_ratio_matches_cn_squared(self, stellar_params):
        sigma_H = stellar_params["sigma_H"]
        vsini = stellar_params["vsini"]
        inc = stellar_params["inc"]
        phi = stellar_params["phi"]
        n_harmonics = 3

        a = vsini * np.cos(phi)
        dv_max = 3 * (abs(a) + 4 * sigma_H)
        dv = np.linspace(-dv_max, dv_max, 2048)

        A_n = compute_harmonic_amplitudes(
            dv, phi, inc, sigma_H, vsini,
            n_harmonics=n_harmonics,
            n_v_internal=2048,
            v_pad_factor=6.0,
        )

        integrals = np.array([np.trapezoid(A_n[n].real, dv) for n in range(n_harmonics + 1)])
        cn_sq = np.array([
            float(_cn_general_jax(n, inc, phi)) ** 2
            for n in range(n_harmonics + 1)
        ])
        W_H_sq = 2 * np.pi * sigma_H ** 2

        nonzero = cn_sq > 1e-12
        if np.any(nonzero):
            ratios = integrals[nonzero] / (cn_sq[nonzero] * W_H_sq)
            np.testing.assert_allclose(
                ratios, 1.0, rtol=0.03,
                err_msg=(
                    f"∫A_n dΔv / (c_n² W_H²) not unity for "
                    f"{stellar_params['label']}"
                ),
            )
