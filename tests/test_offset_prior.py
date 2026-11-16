"""Gaussian priors on marginalized linear nuisance terms (``offset_prior``).

The Phase 2 gate: the Woodbury implementation matches a dense solve, the
all-flat case is unchanged, and the shift invariances hold.
"""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from spotgp import GPSolver, TimeSeriesData, offset_design, segment_labels
from spotgp.offsets import resolve_offset_prior, offset_prior_constants

HPARAM = dict(peq=5.0, kappa=0.1, inc=np.pi / 3,
              lspot=4.0, tau_spot=2.0, sigma_k=5e-3)
SOLVERS = ["cholesky_full", "cholesky_banded"]


def three_segment_data(seed=0, n_per=60, seg_len=12.0, spacing=40.0):
    rng = np.random.default_rng(seed)
    x = np.concatenate([np.linspace(s * spacing, s * spacing + seg_len, n_per)
                        for s in range(3)])
    y = 1.0 + 5e-3 * np.sin(2 * np.pi * x / 5.0) + 1e-4 * rng.standard_normal(x.size)
    return x, y, np.full_like(y, 1e-4)


def nuisance_design(x):
    """Three flat segment offsets plus two smooth 'CBV-like' columns."""
    flat = offset_design(x, segment_labels(x, 1.0), order=0)
    u = (x - x.mean()) / np.ptp(x)
    cbv = np.column_stack([u, np.cos(2 * np.pi * x / 70.0)])
    return flat, cbv


def solver(x, y, yerr, matrix_solver="cholesky_full", **kw):
    return GPSolver(TimeSeriesData(x, y, yerr), model_or_hparam=dict(HPARAM),
                    matrix_solver=matrix_solver, **kw)


def logL(gp):
    return float(gp.log_likelihood_fn(jnp.asarray(gp.theta0)))


def dense_cov(gp):
    x = np.asarray(gp.x)
    lag = np.abs(x[:, None] - x[None, :]).ravel()
    K = np.asarray(gp.kernel_sum.k_of_lag(jnp.asarray(gp.theta0), jnp.asarray(lag)))
    return K.reshape(x.size, x.size) + np.diag(np.asarray(gp.yerr) ** 2)


def gauss_logpdf(r, C):
    _, logdet = np.linalg.slogdet(C)
    return -0.5 * (r @ np.linalg.solve(C, r) + logdet + r.size * np.log(2 * np.pi))


def flat_marginal(r, C, M):
    """Flat-prior marginal likelihood of offsets M (the pre-existing formula)."""
    Ci = np.linalg.inv(C)
    A, b = M.T @ Ci @ M, M.T @ Ci @ r
    _, ldC = np.linalg.slogdet(C)
    _, ldA = np.linalg.slogdet(A)
    return -0.5 * (r @ Ci @ r - b @ np.linalg.solve(A, b) + ldC + ldA
                   + (r.size - M.shape[1]) * np.log(2 * np.pi))


# ── prior specification ────────────────────────────────────────────────────

class TestResolvePrior:

    def test_forms(self):
        mu, s = resolve_offset_prior(None, 3)
        assert np.all(np.isinf(s)) and np.all(mu == 0)
        mu, s = resolve_offset_prior(0.1, 3)
        np.testing.assert_array_equal(s, 0.1)
        mu, s = resolve_offset_prior(np.array([np.inf, 0.2, 0.3]), 3)
        assert np.isinf(s[0]) and s[2] == 0.3
        mu, s = resolve_offset_prior(([5.0, 1.0, 2.0], [np.inf, 0.1, 0.1]), 3)
        np.testing.assert_array_equal(mu, [0.0, 1.0, 2.0])   # mu ignored on flat
        mu, s = resolve_offset_prior(dict(mu=1.0, sigma=0.5), 2)
        np.testing.assert_array_equal(mu, [1.0, 1.0])

    @pytest.mark.parametrize("bad", [-1.0, 0.0, np.nan, dict(sigma=1, bogus=2),
                                     dict(mu=1.0), (1.0, 2.0, 3.0)])
    def test_rejects(self, bad):
        with pytest.raises(ValueError):
            resolve_offset_prior(bad, 2)

    def test_constants(self):
        prec, logdet, n_flat = offset_prior_constants([np.inf, 0.5, 2.0])
        np.testing.assert_allclose(prec, [0.0, 4.0, 0.25])
        assert logdet == pytest.approx(2 * np.log(0.5) + 2 * np.log(2.0))
        assert n_flat == 1

    def test_prior_without_offsets_raises(self):
        with pytest.raises(ValueError, match="requires offsets"):
            solver(*three_segment_data(), offset_prior=0.1)


# ── against an independent dense computation ───────────────────────────────

class TestAgainstDense:

    @pytest.mark.parametrize("matrix_solver", SOLVERS)
    def test_all_gaussian_equals_augmented_covariance(self, matrix_solver):
        """Gaussian columns integrate into the covariance: C + M Lambda M^T."""
        x, y, yerr = three_segment_data()
        _, M = nuisance_design(x)
        mu, sig = np.array([1e-3, -2e-3]), np.array([2e-3, 5e-3])
        gp = solver(x, y, yerr, matrix_solver, offsets=M, offset_prior=(mu, sig))
        C = dense_cov(gp) + M @ np.diag(sig ** 2) @ M.T
        r = np.asarray(gp.y) - float(gp.mean_val) - M @ mu
        rtol = 1e-6 if matrix_solver == "cholesky_full" else 1e-5
        np.testing.assert_allclose(logL(gp), gauss_logpdf(r, C), rtol=rtol)

    @pytest.mark.parametrize("matrix_solver", SOLVERS)
    def test_mixed_flat_and_gaussian(self, matrix_solver):
        """Integrate the Gaussian columns into C, then the flat ones as before."""
        x, y, yerr = three_segment_data(seed=1)
        F, G = nuisance_design(x)
        M = np.hstack([F, G])
        mu = np.r_[0.0, 0.0, 0.0, 5e-4, 0.0]
        sig = np.r_[np.inf, np.inf, np.inf, 3e-3, 1e-3]
        gp = solver(x, y, yerr, matrix_solver, offsets=M, offset_prior=(mu, sig))
        C = dense_cov(gp) + G @ np.diag(sig[3:] ** 2) @ G.T
        r = np.asarray(gp.y) - float(gp.mean_val) - G @ mu[3:]
        rtol = 1e-6 if matrix_solver == "cholesky_full" else 1e-5
        np.testing.assert_allclose(logL(gp), flat_marginal(r, C, F), rtol=rtol)

    def test_banded_matches_full(self):
        x, y, yerr = three_segment_data(seed=2)
        F, G = nuisance_design(x)
        kw = dict(offsets=np.hstack([F, G]),
                  offset_prior=np.r_[np.full(3, np.inf), 2e-3, 2e-3])
        np.testing.assert_allclose(logL(solver(x, y, yerr, "cholesky_full", **kw)),
                                   logL(solver(x, y, yerr, "cholesky_banded", **kw)),
                                   rtol=1e-5)


# ── the flat-prior limit is unchanged ──────────────────────────────────────

class TestFlatLimit:

    @pytest.mark.parametrize("matrix_solver", SOLVERS)
    def test_explicit_flat_prior_is_identical(self, matrix_solver):
        args = three_segment_data()
        base = logL(solver(*args, matrix_solver, offsets=1.0))
        assert logL(solver(*args, matrix_solver, offsets=1.0, offset_prior=None)) == base
        assert logL(solver(*args, matrix_solver, offsets=1.0, offset_prior=np.inf)) == base

    def test_wide_gaussian_converges_to_flat(self):
        """logL_gauss + 0.5 log|Lambda| + 0.5 S log 2pi -> logL_flat as sigma -> inf."""
        x, y, yerr = three_segment_data()
        F, G = nuisance_design(x)
        M = np.hstack([F, G])
        flat = logL(solver(x, y, yerr, offsets=M))
        for s in (1e2, 1e4):
            gp = solver(x, y, yerr, offsets=M, offset_prior=s)
            corr = 0.5 * M.shape[1] * (2 * np.log(s) + np.log(2 * np.pi))
            np.testing.assert_allclose(logL(gp) + corr, flat, rtol=1e-7)

    def test_tight_gaussian_fixes_the_column(self):
        """sigma -> 0 pins a column at its prior mean: same as dropping it from y."""
        x, y, yerr = three_segment_data()
        F, G = nuisance_design(x)
        mu = np.r_[np.zeros(3), 4e-3, 0.0]
        pinned = solver(x, y, yerr, offsets=np.hstack([F, G]),
                        offset_prior=(mu, np.r_[np.full(3, np.inf), 1e-9, 1e-9]))
        y_fixed = y - G @ mu[3:]
        ref = solver(x, y_fixed, yerr, offsets=F)
        # mean_val defaults to mean(y), which differs between the two; flat
        # offsets make the likelihood invariant to it.
        np.testing.assert_allclose(logL(pinned), logL(ref), rtol=1e-7)


# ── shift invariances ──────────────────────────────────────────────────────

class TestInvariance:

    @pytest.mark.parametrize("matrix_solver", SOLVERS)
    def test_flat_columns_stay_exactly_invariant(self, matrix_solver):
        x, y, yerr = three_segment_data()
        F, G = nuisance_design(x)
        kw = dict(offsets=np.hstack([F, G]),
                  offset_prior=np.r_[np.full(3, np.inf), 2e-3, 2e-3])
        a = logL(solver(x, y, yerr, matrix_solver, **kw))
        b = logL(solver(x, y + F @ np.array([0.02, -0.01, 0.05]), yerr, matrix_solver, **kw))
        np.testing.assert_allclose(a, b, rtol=1e-8)

    def test_gaussian_shift_equals_mean_shift(self):
        """y + M_G c with prior mean mu  ==  y with prior mean mu - c."""
        x, y, yerr = three_segment_data()
        F, G = nuisance_design(x)
        M = np.hstack([F, G])
        sig = np.r_[np.full(3, np.inf), 2e-3, 3e-3]
        c = np.array([3e-3, -1e-3])
        mu = np.r_[np.zeros(3), 1e-3, 1e-3]
        a = logL(solver(x, y + G @ c, yerr, offsets=M, offset_prior=(mu, sig)))
        b = logL(solver(x, y, yerr, offsets=M, offset_prior=(mu - np.r_[np.zeros(3), c], sig)))
        np.testing.assert_allclose(a, b, rtol=1e-8)

    def test_gaussian_columns_are_not_free(self):
        """Unlike flat columns, a Gaussian-prior shift changes the likelihood.

        Flat columns are invariant to ~1e-8 (relative, ~1e-5 absolute) above;
        here the change is O(0.1-1). It stays modest because ~60% of the
        70 d cosine lies in the span of the flat segment offsets and the spot
        GP absorbs much of the rest as a slow trend.
        """
        x, y, yerr = three_segment_data()
        F, G = nuisance_design(x)
        kw = dict(offsets=np.hstack([F, G]), offset_prior=np.r_[np.full(3, np.inf), 1e-3, 1e-3])
        a = logL(solver(x, y, yerr, **kw))
        b = logL(solver(x, y + G @ np.array([0.0, 5e-3]), yerr, **kw))
        assert a - b > 0.1


# ── posterior of the nuisance weights ──────────────────────────────────────

class TestPosterior:

    @pytest.mark.parametrize("matrix_solver", SOLVERS)
    def test_matches_dense_formula(self, matrix_solver):
        x, y, yerr = three_segment_data()
        F, G = nuisance_design(x)
        M = np.hstack([F, G])
        mu = np.r_[np.zeros(3), 1e-3, -1e-3]
        sig = np.r_[np.full(3, np.inf), 2e-3, 2e-3]
        y_inj = y + G @ np.array([4e-3, 2e-3])
        gp = solver(x, y_inj, yerr, matrix_solver, offsets=M, offset_prior=(mu, sig))
        a_hat, a_cov = gp.offset_estimates(return_cov=True)

        Ci = np.linalg.inv(dense_cov(gp))
        P = np.diag(np.where(np.isinf(sig), 0.0, 1.0 / np.where(np.isinf(sig), 1.0, sig) ** 2))
        A = M.T @ Ci @ M + P
        r = np.asarray(gp.y) - float(gp.mean_val) - M @ mu
        np.testing.assert_allclose(a_hat, mu + np.linalg.solve(A, M.T @ Ci @ r), rtol=1e-4, atol=1e-9)
        np.testing.assert_allclose(a_cov, np.linalg.inv(A), rtol=1e-4)

    def test_tight_prior_pins_estimate(self):
        x, y, yerr = three_segment_data()
        F, G = nuisance_design(x)
        mu = np.r_[np.zeros(3), 7e-4, -3e-4]
        gp = solver(x, y + G @ np.array([5e-3, 5e-3]), yerr, offsets=np.hstack([F, G]),
                    offset_prior=(mu, np.r_[np.full(3, np.inf), 1e-9, 1e-9]))
        np.testing.assert_allclose(gp.offset_estimates()[3:], mu[3:], atol=1e-8)


# ── Laplace curvature now includes the marginalized nuisance terms ─────────

class TestLaplace:

    @pytest.mark.parametrize("prior", [None, np.r_[np.full(3, np.inf), 2e-3, 2e-3]])
    def test_hessian_uses_marginal_likelihood(self, prior):
        x, y, yerr = three_segment_data()
        F, G = nuisance_design(x)
        gp = solver(x, y, yerr, offsets=np.hstack([F, G]), offset_prior=prior)
        theta = jnp.asarray(gp.theta0)
        gp.mass_matrix_laplace(theta)
        H_ref = jax.hessian(lambda t: -gp.log_likelihood_fn(t))(theta)
        np.testing.assert_allclose(np.asarray(gp._laplace_hessian), np.asarray(H_ref),
                                   rtol=1e-6, atol=1e-6)
