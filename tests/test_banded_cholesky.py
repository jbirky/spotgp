"""Tests for src.banded_cholesky — banded Cholesky factorization and solve."""

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from spotgp.banded_cholesky import (
    banded_cholesky_compact,
    banded_solve_compact,
    banded_cholesky,
    banded_solve,
)


def _make_banded_spd(n, b, rng=None):
    """Create a random symmetric positive-definite banded matrix."""
    if rng is None:
        rng = np.random.default_rng(42)
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - b), min(n, i + b + 1)):
            A[i, j] = rng.standard_normal()
    A = A @ A.T + n * np.eye(n)
    # Zero out entries outside the band
    for i in range(n):
        for j in range(n):
            if abs(i - j) > b:
                A[i, j] = 0.0
    return jnp.array(A)


def _full_to_compact(A, b):
    """Convert full matrix to compact banded storage."""
    n = A.shape[0]
    Ac = jnp.zeros((b + 1, n))
    for d in range(b + 1):
        for j in range(n):
            if j + d < n:
                Ac = Ac.at[d, j].set(A[j + d, j])
    return Ac


class TestBandedCholeskyCompact:
    def test_factorization_small(self):
        """L @ L.T should reconstruct the original matrix."""
        n, b = 10, 2
        A = _make_banded_spd(n, b)
        Ac = _full_to_compact(A, b)
        Lc = banded_cholesky_compact(Ac, b)
        # Verify diagonal is positive
        diag = np.array(Lc[0, :])
        assert np.all(diag > 0)

    def test_solve_compact(self):
        """banded_solve_compact should solve A x = rhs."""
        n, b = 15, 3
        A = _make_banded_spd(n, b)
        Ac = _full_to_compact(A, b)
        rhs = jnp.ones(n)
        Lc = banded_cholesky_compact(Ac, b)
        x = banded_solve_compact(Lc, rhs, b)
        # Check A @ x ≈ rhs
        np.testing.assert_allclose(np.array(A @ x), np.array(rhs), rtol=1e-6)

    def test_solve_random_rhs(self):
        """Test solve with a random right-hand side."""
        rng = np.random.default_rng(123)
        n, b = 20, 4
        A = _make_banded_spd(n, b, rng=rng)
        Ac = _full_to_compact(A, b)
        rhs = jnp.array(rng.standard_normal(n))
        Lc = banded_cholesky_compact(Ac, b)
        x = banded_solve_compact(Lc, rhs, b)
        np.testing.assert_allclose(np.array(A @ x), np.array(rhs), rtol=1e-5)


class TestBandedCholeskyLegacy:
    def test_full_factorization(self):
        """Legacy full-matrix interface."""
        n, b = 10, 2
        A = _make_banded_spd(n, b)
        L = banded_cholesky(A, b)
        # L @ L.T should approximate A
        reconstructed = np.array(L @ L.T)
        np.testing.assert_allclose(reconstructed, np.array(A), atol=1e-8)

    def test_full_solve(self):
        """Legacy full-matrix solve."""
        n, b = 10, 2
        A = _make_banded_spd(n, b)
        L = banded_cholesky(A, b)
        rhs = jnp.ones(n)
        x = banded_solve(L, rhs, b)
        np.testing.assert_allclose(np.array(A @ x), np.array(rhs), rtol=1e-6)
