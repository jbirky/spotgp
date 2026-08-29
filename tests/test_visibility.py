"""Tests for spotgp.visibility — FullGeometry and LimbDarkened variants."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from spotgp.visibility import (
    VisibilityFunction,
    EdgeOnVisibilityFunction,
    FullGeometryVisibilityFunction,
    LimbDarkenedVisibilityFunction,
    FullGeometryLimbDarkenedVisibilityFunction,
    _cn_general_jax,
)


# ── harmonic-order selection ───────────────────────────────────────────────

class TestHarmonicOrders:
    """VisibilityFunction(harmonics=[...]) selects which orders c_n covers."""

    INC = np.pi / 3
    PHI = 0.3

    def test_default_is_zero_one_two(self):
        vis = VisibilityFunction(4.0, 0.0, self.INC)
        assert vis.harmonics == (0, 1, 2)
        assert vis.cn_squared(self.PHI).shape == (3,)

    def test_scalar_is_contiguous_shorthand(self):
        """A scalar keeps the historical `n_harmonics` meaning: 0..n."""
        assert VisibilityFunction(4.0, 0.0, self.INC, harmonics=3).harmonics \
            == (0, 1, 2, 3)

    def test_sparse_orders_match_the_contiguous_subset(self):
        vis = VisibilityFunction(4.0, 0.0, self.INC, harmonics=[0, 2, 4])
        full = np.array(VisibilityFunction(
            4.0, 0.0, self.INC).cn_squared(self.PHI, 4))
        np.testing.assert_allclose(
            np.array(vis.cn_squared(self.PHI)), full[[0, 2, 4]], rtol=1e-12)

    def test_explicit_argument_overrides_the_attribute(self):
        """Callers holding an int (the kernel classes) keep 0..n behaviour."""
        vis = VisibilityFunction(4.0, 0.0, self.INC, harmonics=[0, 2, 4])
        got = np.array(vis.cn_squared(self.PHI, 3))
        ref = np.array([float(_cn_general_jax(n, self.INC, self.PHI)) ** 2
                        for n in range(4)])
        np.testing.assert_allclose(got, ref, rtol=1e-12)

    @pytest.mark.parametrize("bad", [[0, 1, 1], [-1, 0], [], -1, 2.5,
                                     [[0, 1], [2, 3]]])
    def test_invalid_orders_rejected(self, bad):
        with pytest.raises((ValueError, TypeError)):
            VisibilityFunction(4.0, 0.0, self.INC, harmonics=bad)

    def test_edge_on_selects_orders(self):
        """Closed-form g_n vanish above n=2 whichever orders are requested."""
        vis = EdgeOnVisibilityFunction(4.0, harmonics=[2, 5])
        got = np.array(vis.cn_squared(0.0))
        np.testing.assert_allclose(
            got, [1.0 / (18.0 * np.pi ** 2), 0.0], rtol=1e-12)

    @pytest.mark.parametrize("cls,kwargs", [
        (FullGeometryVisibilityFunction, {}),
        (LimbDarkenedVisibilityFunction, {"u": (0.4, 0.2)}),
    ])
    def test_dft_subclasses_select_orders(self, cls, kwargs):
        """The DFT-based subclasses index the same coefficients, sparsely."""
        sparse = cls(4.0, 0.0, self.INC, harmonics=[0, 1, 3], **kwargs)
        full = np.array(sparse.cn_squared(self.PHI, 3))
        np.testing.assert_allclose(
            np.array(sparse.cn_squared(self.PHI)), full[[0, 1, 3]], rtol=1e-12)

    def _model(self, vis):
        from spotgp import SpotEvolutionModel
        from spotgp.envelope import TrapezoidSymmetricEnvelope
        return SpotEvolutionModel(
            envelope=TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=2.0),
            visibility=vis, sigma_k=0.01)

    def test_kernel_inherits_visibility_harmonics(self):
        """AnalyticKernel takes its orders from the visibility by default."""
        from spotgp import AnalyticKernel
        vis = VisibilityFunction(4.0, 0.1, self.INC, harmonics=[0, 2, 4])
        assert AnalyticKernel(self._model(vis)).harmonics == (0, 2, 4)
        # ...and an explicit n_harmonics still overrides it.
        assert AnalyticKernel(
            self._model(vis), n_harmonics=3).harmonics == (0, 1, 2, 3)

    def test_kernel_matches_hand_rolled_series(self):
        """Non-contiguous orders reach kernel(): c_0 undoubled, rest doubled."""
        from spotgp import AnalyticKernel
        orders, peq, kappa, sig = [0, 2, 4], 4.0, 0.1, 0.01
        vis = VisibilityFunction(peq, kappa, self.INC, harmonics=orders)
        model = self._model(vis)
        ak = AnalyticKernel(model, n_lat=32)
        lag = np.linspace(0.0, 8.0, 9)

        phi = np.linspace(-np.pi / 2, np.pi / 2, 32)
        R = np.asarray(model.envelope.R_Gamma(jnp.asarray(lag)))
        acc = np.zeros_like(lag)
        for p in phi:
            w0 = 2 * np.pi * (1 - kappa * np.sin(p) ** 2) / peq
            for n in orders:
                c_sq = float(_cn_general_jax(n, self.INC, p)) ** 2
                acc += (1.0 if n == 0 else 2.0) * c_sq * np.cos(n * w0 * lag)
        np.testing.assert_allclose(
            ak.kernel(lag), sig ** 2 * R * acc / len(phi), rtol=1e-12)

    def test_dropping_orders_changes_the_likelihood(self):
        """The order set must reach the GP log-likelihood, not just kernel()."""
        from spotgp import GPSolver, TimeSeriesData
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0, 40, 50))
        y = 0.01 * np.sin(2 * np.pi * t / 4.0) + 0.002 * rng.standard_normal(50)
        data = TimeSeriesData(t, y, np.full_like(t, 0.002))
        bounds = {"peq": (2, 8), "kappa": (0, 0.4), "inc": (0.3, 1.5),
                  "lspot": (1, 10), "tau_spot": (1, 6),
                  "log_sigma_k": (-4, -1)}

        def solver(harmonics):
            vis = VisibilityFunction(4.0, 0.1, self.INC, harmonics=harmonics)
            return GPSolver(data, self._model(vis), bounds=bounds,
                            matrix_solver="cholesky_full",
                            n_lat=32).build_jax()

        dense, sparse = solver([0, 1, 2, 3, 4]), solver([0, 2, 4])
        assert dense.harmonics == (0, 1, 2, 3, 4)
        assert sparse.harmonics == (0, 2, 4)
        assert sparse.n_harmonics == 4      # int summary is the largest order
        assert abs(float(dense.log_likelihood_fn(dense.theta0))
                   - float(sparse.log_likelihood_fn(sparse.theta0))) > 1e-6
        grad = np.asarray(sparse.grad_log_posterior(sparse.theta0))
        assert np.all(np.isfinite(grad))

    # ── per-harmonic decomposition ──────────────────────────────────────

    @pytest.mark.parametrize("orders", [[0, 1, 2, 3], [0, 2, 4], [1, 2]])
    @pytest.mark.parametrize("quadrature", ["trapezoid", "gauss-legendre"])
    def test_components_sum_to_total(self, orders, quadrature):
        """The decomposition is exact: no cross terms between orders."""
        from spotgp import AnalyticKernel
        vis = VisibilityFunction(4.0, 0.15, self.INC, harmonics=orders)
        ak = AnalyticKernel(self._model(vis), n_lat=32, quadrature=quadrature)
        lag = np.linspace(0.0, 12.0, 40)
        omega = np.linspace(0.05, 6.0, 50)

        K_n = ak.kernel_components(lag)
        assert K_n.shape == (len(orders), len(lag))
        np.testing.assert_allclose(K_n.sum(axis=0), ak.kernel(lag), rtol=1e-12)

        freq, P_n = ak.psd_components(omega)
        assert P_n.shape == (len(orders), len(omega))
        f_ref, P_ref = ak.compute_psd(omega)
        np.testing.assert_allclose(freq, f_ref, rtol=1e-12)
        np.testing.assert_allclose(P_n.sum(axis=0), P_ref, rtol=1e-12)

    @pytest.mark.parametrize("vis_factory", [
        lambda inc: EdgeOnVisibilityFunction(4.0, harmonics=[0, 1, 2, 3]),
        lambda inc: LimbDarkenedVisibilityFunction(
            4.0, 0.15, inc, u=(0.4, 0.2), harmonics=[0, 1, 2, 3]),
    ])
    def test_components_exact_on_alternate_cn_paths(self, vis_factory):
        """Holds for the edge-on fast path and the cn_sq_func hook alike."""
        from spotgp import AnalyticKernel
        ak = AnalyticKernel(self._model(vis_factory(self.INC)), n_lat=32)
        lag = np.linspace(0.0, 12.0, 40)
        np.testing.assert_allclose(
            ak.kernel_components(lag).sum(axis=0), ak.kernel(lag), rtol=1e-12)
        _, P_n = ak.psd_components(np.linspace(0.05, 6.0, 50))
        np.testing.assert_allclose(
            P_n.sum(axis=0),
            ak.compute_psd(np.linspace(0.05, 6.0, 50))[1], rtol=1e-12)

    def test_components_match_single_order_kernels(self):
        """Each row equals the kernel built from that order alone."""
        from spotgp import AnalyticKernel
        orders = [0, 2, 4]
        model = self._model(
            VisibilityFunction(4.0, 0.15, self.INC, harmonics=orders))
        ak = AnalyticKernel(model, n_lat=32)
        lag = np.linspace(0.0, 12.0, 40)
        for row, n in zip(ak.kernel_components(lag), orders):
            solo = AnalyticKernel(model, n_harmonics=[n], n_lat=32).kernel(lag)
            np.testing.assert_allclose(row, solo, rtol=1e-12)

    def test_components_preserve_lag_shape(self):
        """A 2-D lag grid keeps its shape behind the leading order axis."""
        from spotgp import AnalyticKernel
        vis = VisibilityFunction(4.0, 0.15, self.INC, harmonics=[0, 1, 2])
        ak = AnalyticKernel(self._model(vis), n_lat=32)
        t = np.linspace(0.0, 20.0, 9)
        lag2d = np.abs(t[:, None] - t[None, :])
        K_n = ak.kernel_components(lag2d)
        assert K_n.shape == (3, 9, 9)
        np.testing.assert_allclose(
            K_n.sum(axis=0), ak.kernel(lag2d), rtol=1e-12)

    def test_psd_components_do_not_clobber_cache(self):
        """psd_components leaves compute_psd's cached arrays alone."""
        from spotgp import AnalyticKernel
        vis = VisibilityFunction(4.0, 0.15, self.INC, harmonics=[0, 1, 2])
        ak = AnalyticKernel(self._model(vis), n_lat=32)
        _, power = ak.compute_psd(np.linspace(0.05, 6.0, 50))
        ak.psd_components(np.linspace(0.05, 3.0, 17))
        np.testing.assert_allclose(ak.psd_power, power, rtol=1e-12)
        assert ak.psd_power.shape == (50,)

    def test_harmonics_survive_save_load(self, tmp_path):
        import h5py
        from spotgp.io import _read_model, _write_model
        from spotgp import SpotEvolutionModel
        from spotgp.envelope import TrapezoidSymmetricEnvelope

        vis = VisibilityFunction(4.0, 0.1, self.INC, harmonics=[0, 2, 4])
        model = SpotEvolutionModel(
            envelope=TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=2.0),
            visibility=vis, sigma_k=0.01)
        path = str(tmp_path / "vis.h5")
        with h5py.File(path, "w") as f:
            _write_model(f, model)
        with h5py.File(path, "r") as f:
            assert _read_model(f).visibility.harmonics == (0, 2, 4)

        # Files written before harmonics existed fall back to the default.
        with h5py.File(path, "a") as f:
            del f["model/visibility"].attrs["harmonics"]
        with h5py.File(path, "r") as f:
            assert _read_model(f).visibility.harmonics == (0, 1, 2)


class TestFullGeometryProjectedArea:
    def test_fully_visible_matches_formula(self):
        """In fully-visible regime, A = pi sin^2(alpha) cos(beta)."""
        alpha = 0.1
        beta = 0.3  # well within fully-visible
        A = float(FullGeometryVisibilityFunction.projected_area(alpha, beta))
        expected = np.pi * np.sin(alpha)**2 * np.cos(beta)
        np.testing.assert_allclose(A, expected, rtol=1e-10)

    def test_hidden_is_zero(self):
        """Spot on far side has zero projected area."""
        A = float(FullGeometryVisibilityFunction.projected_area(0.1, np.pi * 0.9))
        assert A == pytest.approx(0.0, abs=1e-12)

    def test_partial_visibility_intermediate(self):
        """Near the limb, area should be between 0 and fully-visible value."""
        alpha = 0.3
        beta = np.pi / 2  # right at the limb
        A = float(FullGeometryVisibilityFunction.projected_area(alpha, beta))
        assert A > 0
        A_full = np.pi * np.sin(alpha)**2 * np.cos(beta)
        assert A > A_full  # partial area exceeds the (negative) cos approximation

    def test_zero_alpha_gives_zero(self):
        A = float(FullGeometryVisibilityFunction.projected_area(0.0, 0.5))
        assert A == pytest.approx(0.0, abs=1e-12)

    def test_vectorized(self):
        alpha = 0.1
        betas = jnp.linspace(0, jnp.pi, 100)
        A = FullGeometryVisibilityFunction.projected_area(alpha, betas)
        assert A.shape == (100,)
        # Should be non-negative everywhere
        assert np.all(np.array(A) >= -1e-12)

    def test_monotone_decrease_fully_visible(self):
        """Area decreases as beta increases in the fully-visible region."""
        alpha = 0.05
        betas = jnp.linspace(0, jnp.pi / 2 - alpha - 0.01, 50)
        A = np.array(FullGeometryVisibilityFunction.projected_area(alpha, betas))
        assert np.all(np.diff(A) <= 1e-10)


class TestFullGeometryCosBeta:
    def test_pole_on(self):
        """inc=0 (pole-on): cos(beta) = sin(phi) regardless of longitude."""
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=0.0)
        cb = float(vis.cos_beta(np.pi / 4, 0.0))
        np.testing.assert_allclose(cb, np.sin(np.pi / 4), rtol=1e-10)

    def test_edge_on_equator(self):
        """inc=pi/2, phi=0: cos(beta) = cos(longitude)."""
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 2)
        lon = 0.5
        cb = float(vis.cos_beta(0.0, lon))
        np.testing.assert_allclose(cb, np.cos(lon), rtol=1e-10)


class TestFullGeometryVisibilityProfile:
    def test_shape(self):
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 3)
        lon, A = vis.visibility_profile(np.pi / 6, 0.1)
        assert lon.shape == (512,)
        assert A.shape == (512,)

    def test_custom_n_lon(self):
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 3)
        lon, A = vis.visibility_profile(np.pi / 6, 0.1, n_lon=64)
        assert lon.shape == (64,)

    def test_non_negative(self):
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 3)
        _, A = vis.visibility_profile(np.pi / 6, 0.1)
        assert np.all(np.array(A) >= -1e-12)


class TestFullGeometryCnSquared:
    def test_shape(self):
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 3)
        cn = vis.cn_squared(0.3, n_harmonics=3)
        assert cn.shape == (4,)

    def test_non_negative(self):
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.0, inc=np.pi / 3)
        cn = np.array(vis.cn_squared(0.3, n_harmonics=3))
        assert np.all(cn >= 0)

    def test_small_spot_matches_base(self):
        """For small alpha_ref, full geometry should match small-spot approx."""
        inc = np.pi / 3
        phi = 0.3
        vis_small = VisibilityFunction(peq=10.0, kappa=0.0, inc=inc)
        vis_full = FullGeometryVisibilityFunction(
            peq=10.0, kappa=0.0, inc=inc, alpha_ref=0.01)
        cn_small = np.array(vis_small.cn_squared(phi, 3))
        cn_full = np.array(vis_full.cn_squared(phi, 3))
        np.testing.assert_allclose(cn_full, cn_small, rtol=0.1)

    def test_large_spot_diverges(self):
        """For large alpha_ref, results should differ from small-spot approx."""
        inc = np.pi / 3
        phi = 0.3
        vis_small = VisibilityFunction(peq=10.0, kappa=0.0, inc=inc)
        vis_full = FullGeometryVisibilityFunction(
            peq=10.0, kappa=0.0, inc=inc, alpha_ref=0.8)
        cn_small = np.array(vis_small.cn_squared(phi, 3))
        cn_full = np.array(vis_full.cn_squared(phi, 3))
        # At least one coefficient should differ by > 10%
        rel_diff = np.abs(cn_full - cn_small) / np.maximum(cn_small, 1e-15)
        assert np.any(rel_diff > 0.1)

    def test_inherits_omega0(self):
        vis = FullGeometryVisibilityFunction(peq=10.0, kappa=0.3, inc=np.pi / 3)
        omega = float(vis.omega0(0.0))
        np.testing.assert_allclose(omega, 2 * np.pi / 10.0, rtol=1e-10)


# ── _cn_general_jax pole handling ──────────────────────────────────────────

class TestCnGeneralPoles:
    """A spot at a rotational pole sits at constant beta from the observer."""

    def test_visible_pole(self):
        """phi=+pi/2: spot always visible at beta=inc, so c_0 = cos(inc)."""
        inc = np.pi / 3
        c0 = float(_cn_general_jax(0, inc, np.pi / 2))
        np.testing.assert_allclose(c0, np.cos(inc), atol=1e-10)

    def test_hidden_pole_is_zero(self):
        """phi=-pi/2: spot is never visible, so c_0 must be 0 (not -cos inc)."""
        for inc in [np.pi / 3, np.pi / 4, np.pi / 6]:
            c0 = float(_cn_general_jax(0, inc, -np.pi / 2))
            assert c0 == pytest.approx(0.0, abs=1e-10), f"inc={inc}"

    def test_pole_harmonics_vanish(self):
        """At a pole the spot never moves, so there are no rotation harmonics."""
        for n in [1, 2, 3]:
            for phi in [np.pi / 2, -np.pi / 2]:
                cn = float(_cn_general_jax(n, np.pi / 3, phi))
                assert cn == pytest.approx(0.0, abs=1e-10)

    def test_pole_on_view_unchanged(self):
        """inc=0: c_0 = sin(phi) for a visible (northern) spot."""
        c0 = float(_cn_general_jax(0, 0.0, np.pi / 4))
        np.testing.assert_allclose(c0, np.sin(np.pi / 4), atol=1e-10)

    def test_pole_on_hidden_hemisphere_is_zero(self):
        """inc=0, phi<0: southern spot is hidden from a north-pole view."""
        c0 = float(_cn_general_jax(0, 0.0, -np.pi / 4))
        assert c0 == pytest.approx(0.0, abs=1e-10)

    def test_matches_numerical_dft_at_poles(self):
        """Analytic c_n agrees with a direct DFT of max(cos beta, 0)."""
        inc = np.pi / 3
        for phi in [np.pi / 2, -np.pi / 2]:
            lon = jnp.linspace(0, 2 * jnp.pi, 4096, endpoint=False)
            mu = (jnp.cos(inc) * jnp.sin(phi)
                  + jnp.sin(inc) * jnp.cos(phi) * jnp.cos(lon))
            V = jnp.clip(mu, 0.0, None)
            num = np.abs(np.asarray(jnp.fft.rfft(V) / V.shape[0]))[:4]
            ana = np.abs(np.array(
                [float(_cn_general_jax(n, inc, phi)) for n in range(4)]))
            np.testing.assert_allclose(ana, num, atol=1e-8)


# ── LimbDarkenedVisibilityFunction ─────────────────────────────────────────

class TestLimbDarkenedVisibility:
    def test_zero_coefficients_match_base_quadratic(self):
        """u=(0,0) reduces exactly to the uniform-disk analytic c_n."""
        for inc in [np.pi / 2, np.pi / 3, np.pi / 6]:
            for phi in [0.0, 0.3, -0.7, 1.2]:
                vis = LimbDarkenedVisibilityFunction(
                    10.0, 0.0, inc, u=(0.0, 0.0), n_lon=4096)
                got = np.array(vis.cn_squared(phi, 3))
                ref = np.array([float(_cn_general_jax(n, inc, phi)) ** 2
                                for n in range(4)])
                np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_zero_coefficients_match_base_claret(self):
        vis = LimbDarkenedVisibilityFunction(
            10.0, 0.0, np.pi / 3, u=(0.0,) * 4, law="claret", n_lon=4096)
        got = np.array(vis.cn_squared(0.3, 3))
        ref = np.array([float(_cn_general_jax(n, np.pi / 3, 0.3)) ** 2
                        for n in range(4)])
        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_flux_norm_quadratic(self):
        """F = 1 - u1/3 - u2/6 for the quadratic law."""
        vis = LimbDarkenedVisibilityFunction(10.0, 0.0, 1.0, u=(0.4, 0.2))
        assert vis.flux_norm == pytest.approx(1.0 - 0.4 / 3 - 0.2 / 6)

    def test_flux_norm_matches_quadrature(self):
        """F should equal the numerical integral int_0^1 2 mu I(mu) dmu."""
        for law, u in [("quadratic", (0.4, 0.2)),
                       ("claret", (0.3999, 0.4269, -0.0227, -0.0839))]:
            vis = LimbDarkenedVisibilityFunction(
                10.0, 0.0, 1.0, u=u, law=law)
            mu = np.linspace(1e-9, 1.0, 200001)
            num = np.trapezoid(2 * mu * np.array(vis.intensity(mu)), mu)
            assert vis.flux_norm == pytest.approx(num, rel=1e-5), law

    def test_intensity_normalized_at_disk_center(self):
        """I(mu=1) = 1 by construction for both laws."""
        for law, u in [("quadratic", (0.4, 0.2)),
                       ("claret", (0.3999, 0.4269, -0.0227, -0.0839))]:
            vis = LimbDarkenedVisibilityFunction(10.0, 0.0, 1.0, u=u, law=law)
            assert float(vis.intensity(1.0)) == pytest.approx(1.0)

    def test_limb_darkening_adds_higher_harmonics(self):
        """LD sharpens V(theta), moving power into harmonics that were zero."""
        phi, inc = 0.2, np.pi / 2
        c3_off = float(LimbDarkenedVisibilityFunction(
            10.0, 0.0, inc, u=(0.0, 0.0), n_lon=4096).cn_squared(phi, 3)[3])
        c3_on = float(LimbDarkenedVisibilityFunction(
            10.0, 0.0, inc, u=(0.6, 0.1), n_lon=4096).cn_squared(phi, 3)[3])
        assert c3_off == pytest.approx(0.0, abs=1e-12)
        assert c3_on > 1e-4

    def test_hidden_pole_is_zero(self):
        """A never-visible spot contributes nothing, as for the base class."""
        vis = LimbDarkenedVisibilityFunction(10.0, 0.0, np.pi / 3, u=(0.4, 0.2))
        cn = np.array(vis.cn_squared(-np.pi / 2, 3))
        np.testing.assert_allclose(cn, 0.0, atol=1e-12)

    def test_non_negative(self):
        vis = LimbDarkenedVisibilityFunction(10.0, 0.0, np.pi / 3, u=(0.4, 0.2))
        assert np.all(np.array(vis.cn_squared(0.3, 3)) >= 0)

    def test_profile_non_negative(self):
        vis = LimbDarkenedVisibilityFunction(10.0, 0.0, np.pi / 3, u=(0.4, 0.2))
        _, V = vis.visibility_profile(0.3)
        assert np.all(np.array(V) >= -1e-12)

    def test_n_lon_convergence(self):
        """Default n_lon=512 is converged to ~1e-6 against a fine grid."""
        kw = dict(u=(0.3, 0.2))
        ref = np.array(LimbDarkenedVisibilityFunction(
            10.0, 0.0, np.pi / 3, n_lon=16384, **kw).cn_squared(0.2, 3))
        got = np.array(LimbDarkenedVisibilityFunction(
            10.0, 0.0, np.pi / 3, n_lon=512, **kw).cn_squared(0.2, 3))
        np.testing.assert_allclose(got, ref, atol=1e-6)

    @pytest.mark.parametrize("law,u", [
        ("quadratic", (0.3, 0.2)),
        ("claret", (0.3999, 0.4269, -0.0227, -0.0839)),
    ])
    def test_grad_wrt_inc_finite(self, law, u):
        """Autodiff through the DFT matches finite differences.

        The Claret law's mu**(k/2) terms have infinite slope at mu=0, so this
        guards the positive clip floor in visibility_profile.
        """
        vis = LimbDarkenedVisibilityFunction(
            10.0, 0.0, np.pi / 3, u=u, law=law, n_lon=256)

        def f(inc):
            return jnp.sum(vis.cn_sq_jax(jnp.array([10.0, 0.0, inc]), 0.2, 3))

        g = float(jax.grad(f)(np.pi / 3))
        fd = float((f(np.pi / 3 + 1e-6) - f(np.pi / 3 - 1e-6)) / 2e-6)
        assert np.isfinite(g)
        assert g == pytest.approx(fd, abs=1e-6)

    def test_rejects_bad_law(self):
        with pytest.raises(ValueError, match="Unknown limb-darkening law"):
            LimbDarkenedVisibilityFunction(10.0, 0.0, 1.0, law="linear")

    def test_rejects_wrong_coefficient_count(self):
        with pytest.raises(ValueError, match="2 coefficients"):
            LimbDarkenedVisibilityFunction(10.0, 0.0, 1.0, u=(0.1, 0.2, 0.3))
        with pytest.raises(ValueError, match="4 coefficients"):
            LimbDarkenedVisibilityFunction(
                10.0, 0.0, 1.0, u=(0.1, 0.2), law="claret")

    def test_get_sympy_raises(self):
        vis = LimbDarkenedVisibilityFunction(10.0, 0.0, 1.0)
        with pytest.raises(NotImplementedError, match="numerically by DFT"):
            vis.get_sympy()


# ── cn_sq_func hook: custom coefficients must reach the kernel ─────────────

class TestCustomCnSqReachesKernel:
    """A custom visibility must affect kernel()/logL, not just the PSD.

    Before the cn_sq_func hook existed, _kernel_eval hardcoded the analytic
    _cn_general_jax coefficients, so overriding cn_squared() silently had no
    effect on the latitude-averaged kernel or the GP log-likelihood.
    """

    INC = np.pi / 3
    PEQ = 4.0

    def _kernel(self, vis, n_harmonics=4, n_lat=32):
        from spotgp import SpotEvolutionModel, AnalyticKernel
        from spotgp.envelope import TrapezoidSymmetricEnvelope
        model = SpotEvolutionModel(
            envelope=TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=2.0),
            visibility=vis, sigma_k=0.01)
        return AnalyticKernel(model, n_harmonics=n_harmonics, n_lat=n_lat)

    def test_get_cn_sq_func_none_for_default_visibility(self):
        """The built-in visibility keeps using the inline analytic path."""
        from spotgp import SpotEvolutionModel
        from spotgp.envelope import TrapezoidSymmetricEnvelope
        model = SpotEvolutionModel(
            envelope=TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=2.0),
            visibility=VisibilityFunction(self.PEQ, 0.0, self.INC),
            sigma_k=0.01)
        assert model.get_cn_sq_func(3) is None

    def test_get_cn_sq_func_shape(self):
        from spotgp import SpotEvolutionModel
        from spotgp.envelope import TrapezoidSymmetricEnvelope
        model = SpotEvolutionModel(
            envelope=TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=2.0),
            visibility=LimbDarkenedVisibilityFunction(
                self.PEQ, 0.0, self.INC, u=(0.4, 0.2), n_lon=128),
            sigma_k=0.01)
        fn = model.get_cn_sq_func(3)
        assert fn is not None
        phi_grid = jnp.linspace(-np.pi / 2, np.pi / 2, 8)
        out = fn(jnp.asarray(model.theta0), phi_grid)
        assert out.shape == (8, 4)

    def test_limb_darkening_changes_kernel(self):
        plain = self._kernel(VisibilityFunction(self.PEQ, 0.0, self.INC))
        ld = self._kernel(LimbDarkenedVisibilityFunction(
            self.PEQ, 0.0, self.INC, u=(0.6, 0.1)))
        lag = np.linspace(0.0, 8.0, 7)
        assert not np.allclose(plain.kernel(lag), ld.kernel(lag))

    def test_zero_limb_darkening_reproduces_analytic_kernel(self):
        plain = self._kernel(VisibilityFunction(self.PEQ, 0.0, self.INC))
        ld_off = self._kernel(LimbDarkenedVisibilityFunction(
            self.PEQ, 0.0, self.INC, u=(0.0, 0.0), n_lon=4096))
        lag = np.linspace(0.0, 8.0, 7)
        np.testing.assert_allclose(
            ld_off.kernel(lag), plain.kernel(lag), rtol=1e-5, atol=1e-12)

    @pytest.mark.parametrize("solver", ["cholesky_full", "cholesky_banded"])
    def test_limb_darkening_reaches_log_posterior(self, solver):
        from spotgp import (SpotEvolutionModel, GPSolver, TimeSeriesData)
        from spotgp.envelope import TrapezoidSymmetricEnvelope

        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0, 40, 50))
        y = 0.01 * np.sin(2 * np.pi * t / 4.0) + 0.002 * rng.standard_normal(50)
        data = TimeSeriesData(t, y, np.full_like(t, 0.002))
        bounds = {"peq": (2, 8), "kappa": (0, 0.4), "inc": (0.3, 1.5),
                  "lspot": (1, 10), "tau_spot": (1, 6),
                  "log_sigma_k": (-4, -1)}

        def build(vis):
            model = SpotEvolutionModel(
                envelope=TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=2.0),
                visibility=vis, sigma_k=0.01)
            return GPSolver(data, model, bounds=bounds,
                            matrix_solver=solver).build_jax()

        gp_plain = build(VisibilityFunction(self.PEQ, 0.1, self.INC))
        gp_ld = build(LimbDarkenedVisibilityFunction(
            self.PEQ, 0.1, self.INC, u=(0.6, 0.1)))

        logL_plain = float(gp_plain.log_posterior(gp_plain.theta0))
        logL_ld = float(gp_ld.log_posterior(gp_ld.theta0))
        assert abs(logL_plain - logL_ld) > 1e-6

        grad = np.asarray(gp_ld.grad_log_posterior(gp_ld.theta0))
        assert np.all(np.isfinite(grad))
        inc_idx = list(gp_ld.param_keys).index("inc")
        assert abs(grad[inc_idx]) > 1e-9


# ── full geometry + limb darkening ──────────────────────────────────────────
class TestFullGeometryLimbDarkened:
    PEQ, KAPPA, INC, PHI = 5.4, 0.1, np.deg2rad(65), np.deg2rad(25)

    def test_uniform_disk_deficit_matches_exact_area(self):
        """u=(0,0): the cap integral equals Eq. 5 in all three regimes."""
        alpha = 0.25
        vis = FullGeometryLimbDarkenedVisibilityFunction(
            self.PEQ, self.KAPPA, self.INC, alpha_ref=alpha,
            u=(0.0, 0.0), n_rho=48, n_psi=96)
        beta = np.linspace(0.0, np.pi, 721)
        A_ref = np.asarray(
            FullGeometryVisibilityFunction.projected_area(alpha, beta))
        D = np.asarray(vis.spot_deficit(np.cos(beta)))
        np.testing.assert_allclose(D, A_ref, atol=1e-4 * A_ref.max())

    def test_hidden_regime_is_zero(self):
        alpha = 0.25
        vis = FullGeometryLimbDarkenedVisibilityFunction(
            self.PEQ, self.KAPPA, self.INC, alpha_ref=alpha, u=(0.4, 0.2))
        beta_hidden = np.pi / 2 + alpha + 0.05
        assert float(vis.spot_deficit(np.cos(beta_hidden))) == 0.0

    def test_uniform_disk_cn_matches_fullgeometry(self):
        alpha = 0.25
        vis_fg = FullGeometryVisibilityFunction(
            self.PEQ, self.KAPPA, self.INC, alpha_ref=alpha)
        vis = FullGeometryLimbDarkenedVisibilityFunction(
            self.PEQ, self.KAPPA, self.INC, alpha_ref=alpha,
            u=(0.0, 0.0), n_rho=48, n_psi=96)
        np.testing.assert_allclose(
            np.asarray(vis.cn_squared(self.PHI)),
            np.asarray(vis_fg.cn_squared(self.PHI)), rtol=2e-3, atol=1e-9)

    def test_small_spot_limit_matches_limbdarkened(self):
        u = (0.4, 0.2)
        vis_ld = LimbDarkenedVisibilityFunction(
            self.PEQ, self.KAPPA, self.INC, u=u)
        vis = FullGeometryLimbDarkenedVisibilityFunction(
            self.PEQ, self.KAPPA, self.INC, alpha_ref=1e-3, u=u)
        np.testing.assert_allclose(
            np.asarray(vis.cn_squared(self.PHI)),
            np.asarray(vis_ld.cn_squared(self.PHI)), rtol=1e-4, atol=1e-12)

    def test_both_limits_match_base_analytic(self):
        vis = FullGeometryLimbDarkenedVisibilityFunction(
            self.PEQ, self.KAPPA, self.INC, alpha_ref=1e-3, u=(0.0, 0.0))
        ref = np.array([float(_cn_general_jax(n, self.INC, self.PHI)) ** 2
                        for n in range(3)])
        np.testing.assert_allclose(
            np.asarray(vis.cn_squared(self.PHI)), ref, rtol=1e-3, atol=1e-10)

    def test_finite_spot_shifts_harmonics(self):
        """A large spot must change |c_n|^2 relative to the alpha->0 limit."""
        u = (0.4, 0.2)
        small = FullGeometryLimbDarkenedVisibilityFunction(
            self.PEQ, self.KAPPA, self.INC, alpha_ref=1e-3, u=u)
        big = FullGeometryLimbDarkenedVisibilityFunction(
            self.PEQ, self.KAPPA, self.INC, alpha_ref=0.4, u=u,
            n_rho=64, n_psi=128)
        assert not np.allclose(np.asarray(big.cn_squared(self.PHI)),
                               np.asarray(small.cn_squared(self.PHI)),
                               rtol=1e-3)

    def test_grad_inc_finite_and_correct_at_disk_center_crossing(self):
        """inc+phi = pi/2 puts the spot through disk centre (cos beta = 1),
        the geometry that made sqrt(1 - cos^2 beta) non-differentiable."""
        vis = FullGeometryLimbDarkenedVisibilityFunction(
            self.PEQ, self.KAPPA, self.INC, alpha_ref=0.15, u=(0.4, 0.2))
        f = lambda i: jnp.sum(vis.cn_sq_at(i, self.PHI))
        g_ad = float(jax.grad(f)(jnp.asarray(self.INC)))
        eps = 1e-6
        g_fd = (float(f(jnp.asarray(self.INC + eps)))
                - float(f(jnp.asarray(self.INC - eps)))) / (2 * eps)
        assert np.isfinite(g_ad)
        assert g_ad == pytest.approx(g_fd, rel=1e-4)

    def test_claret_grad_finite(self):
        vis = FullGeometryLimbDarkenedVisibilityFunction(
            self.PEQ, self.KAPPA, self.INC, alpha_ref=0.15,
            u=(0.5, -0.1, 0.3, -0.05), law="claret")
        g = jax.grad(lambda i: jnp.sum(vis.cn_sq_at(i, self.PHI)))(
            jnp.asarray(self.INC))
        assert np.isfinite(float(g))

    def test_alpha_ref_validation(self):
        for bad in [0.0, -0.1, np.pi / 2, 2.0]:
            with pytest.raises(ValueError, match="alpha_ref"):
                FullGeometryLimbDarkenedVisibilityFunction(
                    self.PEQ, self.KAPPA, self.INC, alpha_ref=bad)

    def test_cn_sq_jax_hook_reaches_spot_model(self):
        from spotgp import SpotEvolutionModel, TrapezoidSymmetricEnvelope
        vis = FullGeometryLimbDarkenedVisibilityFunction(
            self.PEQ, self.KAPPA, self.INC, alpha_ref=0.2, u=(0.4, 0.2))
        model = SpotEvolutionModel(
            envelope=TrapezoidSymmetricEnvelope(lspot=20.0, tau_spot=2.0),
            visibility=vis, sigma_k=1e-3)
        fn = model.get_cn_sq_func((0, 1, 2))
        assert fn is not None
        out = fn(jnp.array(model.theta0), jnp.linspace(-1.0, 1.0, 5))
        assert out.shape == (5, 3)
        assert np.all(np.isfinite(np.asarray(out)))
