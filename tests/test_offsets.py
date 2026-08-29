"""Tests for spotgp.offsets — marginalized per-segment flux offsets."""

import numpy as np
import pytest
import jax.numpy as jnp
import jax.scipy.linalg as jla

from spotgp import GPSolver, TimeSeriesData, segment_labels, offset_design
from spotgp.offsets import apply_offset_marginalization, offset_posterior


# ── helpers ────────────────────────────────────────────────────────────────

HPARAM = dict(peq=5.0, kappa=0.1, inc=np.pi / 3,
              lspot=4.0, tau_spot=2.0, sigma_k=5e-3)


def three_segment_data(seed=0, n_per=60, seg_len=12.0, spacing=40.0,
                       offsets=(0.0, 0.0, 0.0)):
    """Three well-separated segments, each optionally shifted."""
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for s, off in enumerate(offsets):
        x = np.linspace(s * spacing, s * spacing + seg_len, n_per)
        y = 1.0 + 5e-3 * np.sin(2 * np.pi * x / 5.0) + off
        xs.append(x)
        ys.append(y)
    x = np.concatenate(xs)
    y = np.concatenate(ys) + 1e-4 * rng.standard_normal(len(x))
    return x, y, np.full_like(y, 1e-4)


def solver(x, y, yerr, offsets=None, matrix_solver="cholesky_full"):
    return GPSolver(TimeSeriesData(x, y, yerr), model_or_hparam=dict(HPARAM),
                    matrix_solver=matrix_solver, offsets=offsets)


def logL(gp):
    return float(gp.log_likelihood_fn(jnp.asarray(gp.theta0)))


# ── design matrix construction ─────────────────────────────────────────────

class TestDesign:

    def test_segment_labels_split_on_gaps(self):
        x, _, _ = three_segment_data()
        labels = segment_labels(x, gap=1.0)
        assert set(np.unique(labels)) == {0, 1, 2}

    def test_labels_are_gap_threshold_dependent(self):
        """A threshold larger than the inter-segment gap merges everything."""
        x, _, _ = three_segment_data()
        assert len(np.unique(segment_labels(x, gap=1e3))) == 1

    def test_labels_follow_time_order_for_unsorted_input(self):
        x, _, _ = three_segment_data()
        perm = np.random.default_rng(1).permutation(len(x))
        labels = segment_labels(x[perm], gap=1.0)
        # Undo the permutation: must match the sorted-input labelling
        assert np.array_equal(labels[np.argsort(perm)],
                              segment_labels(x, gap=1.0))

    def test_design_is_partition_of_unity(self):
        x, _, _ = three_segment_data()
        M = offset_design(x, segment_labels(x, gap=1.0), order=0)
        assert M.shape == (len(x), 3)
        np.testing.assert_allclose(M.sum(axis=1), 1.0)

    def test_ramp_columns_are_scaled_to_unit_range(self):
        x, _, _ = three_segment_data()
        M = offset_design(x, segment_labels(x, gap=1.0), order=1)
        assert M.shape == (len(x), 6)
        for col in (1, 3, 5):
            np.testing.assert_allclose(np.abs(M[:, col]).max(), 1.0)

    def test_rejects_mismatched_shapes(self):
        with pytest.raises(ValueError, match="same shape"):
            offset_design(np.arange(10.0), np.zeros(5, dtype=int))


# ── the defining property ──────────────────────────────────────────────────

class TestShiftInvariance:
    """
    Under a flat prior the marginal likelihood must be EXACTLY invariant
    to adding an arbitrary constant to any segment. This is what makes it
    a statement of ignorance about the zero points rather than an
    assertion that each segment has zero mean.
    """

    SHIFTS = (0.0, 0.01, -0.003)

    def test_offsets_change_likelihood_when_not_modelled(self):
        """Control: without offsets= the shift must matter."""
        base = solver(*three_segment_data())
        shifted = solver(*three_segment_data(offsets=self.SHIFTS))
        assert not np.isclose(logL(base), logL(shifted), rtol=1e-6)

    @pytest.mark.parametrize("matrix_solver",
                             ["cholesky_full", "cholesky_banded"])
    def test_invariant_when_marginalized(self, matrix_solver):
        base = solver(*three_segment_data(), offsets=1.0,
                      matrix_solver=matrix_solver)
        shifted = solver(*three_segment_data(offsets=self.SHIFTS), offsets=1.0,
                         matrix_solver=matrix_solver)
        np.testing.assert_allclose(logL(base), logL(shifted), rtol=1e-8)

    def test_invariant_to_large_shifts(self):
        """Invariance is exact, not a small-shift approximation."""
        base = solver(*three_segment_data(), offsets=1.0)
        wild = solver(*three_segment_data(offsets=(0.0, 50.0, -20.0)),
                      offsets=1.0)
        np.testing.assert_allclose(logL(base), logL(wild), rtol=1e-8)

    def test_order_one_absorbs_per_segment_ramps(self):
        """With order=1 a linear drift per segment is also marginalized."""
        x, y, yerr = three_segment_data()
        labels = segment_labels(x, gap=1.0)

        y_ramped = y.copy()
        for s, slope in zip(np.unique(labels), [2e-3, -1e-3, 5e-4]):
            m = labels == s
            y_ramped[m] += slope * (x[m] - x[m].mean())

        base = solver(x, y, yerr, offsets=dict(gap=1.0, order=1))
        ramped = solver(x, y_ramped, yerr, offsets=dict(gap=1.0, order=1))
        np.testing.assert_allclose(logL(base), logL(ramped), rtol=1e-8)

    def test_order_zero_does_not_absorb_ramps(self):
        """Guard against the design matrix silently spanning too much."""
        x, y, yerr = three_segment_data()
        labels = segment_labels(x, gap=1.0)
        y_ramped = y.copy()
        for s, slope in zip(np.unique(labels), [2e-3, -1e-3, 5e-4]):
            m = labels == s
            y_ramped[m] += slope * (x[m] - x[m].mean())

        base = solver(x, y, yerr, offsets=1.0)
        ramped = solver(x, y_ramped, yerr, offsets=1.0)
        assert not np.isclose(logL(base), logL(ramped), rtol=1e-6)


# ── correctness against an independent reference ───────────────────────────

class TestAgainstDenseReference:

    @staticmethod
    def _reference_logL(gp):
        """Marginalized log-likelihood computed from scratch, densely."""
        x = np.asarray(gp.x)
        y = np.asarray(gp.y)
        yerr = np.asarray(gp.yerr)
        M = np.asarray(gp.design)
        N, S = M.shape

        # k_of_lag broadcasts over harmonics internally: pass flat lags
        lag = np.abs(x[:, None] - x[None, :]).ravel()
        K = np.asarray(gp.kernel_sum.k_of_lag(
            jnp.asarray(gp.theta0), jnp.asarray(lag))).reshape(N, N)
        C = K + np.diag(yerr ** 2) + 1e-8 * np.eye(N)

        r = y - float(gp.mean_val)
        Cinv = np.linalg.inv(C)
        A = M.T @ Cinv @ M
        b = M.T @ Cinv @ r

        quad = r @ Cinv @ r - b @ np.linalg.solve(A, b)
        _, logdetC = np.linalg.slogdet(C)
        _, logdetA = np.linalg.slogdet(A)
        return -0.5 * (quad + logdetC + logdetA + (N - S) * np.log(2 * np.pi))

    def test_matches_dense_formula(self):
        gp = solver(*three_segment_data(), offsets=1.0)
        np.testing.assert_allclose(logL(gp), self._reference_logL(gp),
                                   rtol=1e-6)

    def test_banded_matches_full(self):
        args = three_segment_data()
        full = solver(*args, offsets=1.0, matrix_solver="cholesky_full")
        banded = solver(*args, offsets=1.0, matrix_solver="cholesky_banded")
        np.testing.assert_allclose(logL(full), logL(banded), rtol=1e-5)

    def test_no_offsets_path_is_untouched(self):
        """design=None must reproduce the pre-existing likelihood exactly."""
        x, y, yerr = three_segment_data()
        gp = solver(x, y, yerr, offsets=None)

        N = len(x)
        lag = np.abs(x[:, None] - x[None, :]).ravel()
        K = np.asarray(gp.kernel_sum.k_of_lag(
            jnp.asarray(gp.theta0), jnp.asarray(lag))).reshape(N, N)
        C = K + np.diag(np.asarray(gp.yerr) ** 2) + 1e-8 * np.eye(N)
        r = np.asarray(gp.y) - float(gp.mean_val)
        _, logdetC = np.linalg.slogdet(C)
        ref = -0.5 * (r @ np.linalg.solve(C, r) + logdetC
                      + N * np.log(2 * np.pi))
        np.testing.assert_allclose(logL(gp), ref, rtol=1e-6)


# ── offset recovery ────────────────────────────────────────────────────────

class TestOffsetRecovery:

    SHIFTS = (0.0, 0.012, -0.005)

    @pytest.mark.parametrize("matrix_solver",
                             ["cholesky_full", "cholesky_banded"])
    def test_recovers_injected_offsets(self, matrix_solver):
        gp = solver(*three_segment_data(offsets=self.SHIFTS), offsets=1.0,
                    matrix_solver=matrix_solver)
        a_hat = gp.offset_estimates()
        assert a_hat.shape == (3,)
        # Only differences are identifiable: the overall level is degenerate
        # with the GP mean, so compare shifts relative to the first segment.
        rel_hat = a_hat - a_hat[0]
        rel_true = np.array(self.SHIFTS) - self.SHIFTS[0]
        np.testing.assert_allclose(rel_hat, rel_true, atol=2e-3)

    def test_returns_covariance(self):
        gp = solver(*three_segment_data(offsets=self.SHIFTS), offsets=1.0)
        a_hat, a_cov = gp.offset_estimates(return_cov=True)
        assert a_cov.shape == (3, 3)
        np.testing.assert_allclose(a_cov, a_cov.T, rtol=1e-8)
        assert np.all(np.diag(a_cov) > 0)

    def test_raises_without_offsets(self):
        gp = solver(*three_segment_data())
        with pytest.raises(ValueError, match="without offsets"):
            gp.offset_estimates()


# ── argument handling ──────────────────────────────────────────────────────

class TestOffsetsArgument:

    def test_float_is_gap_shorthand(self):
        gp = solver(*three_segment_data(), offsets=1.0)
        assert gp.n_offsets == 3

    def test_explicit_labels(self):
        x, y, yerr = three_segment_data()
        gp = solver(x, y, yerr, offsets=segment_labels(x, gap=1.0))
        assert gp.n_offsets == 3

    def test_explicit_design_matrix(self):
        x, y, yerr = three_segment_data()
        M = offset_design(x, segment_labels(x, gap=1.0), order=1)
        gp = solver(x, y, yerr, offsets=M)
        assert gp.n_offsets == 6

    def test_none_disables(self):
        gp = solver(*three_segment_data(), offsets=None)
        assert gp.design is None and gp.n_offsets == 0

    def test_rejects_wrong_length_labels(self):
        x, y, yerr = three_segment_data()
        with pytest.raises(ValueError, match="length"):
            solver(x, y, yerr, offsets=np.zeros(7, dtype=int))

    def test_rejects_saturating_design(self):
        """A design with as many columns as points explains everything."""
        x, y, yerr = three_segment_data(n_per=4)
        with pytest.raises(ValueError, match="absorb the entire dataset"):
            solver(x, y, yerr, offsets=np.eye(len(x)))

    def test_rejects_unknown_dict_keys(self):
        x, y, yerr = three_segment_data()
        with pytest.raises(ValueError, match="Unknown keys"):
            solver(x, y, yerr, offsets=dict(gap=1.0, window=3))
