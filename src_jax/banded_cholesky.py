"""
Banded Cholesky factorization and triangular solve for JAX.

Provides O(n * b²) factorization for symmetric positive definite matrices
whose non-zero entries are confined to a band of width b around the diagonal.
The bandwidth b must be a Python integer (compile-time constant) so that
the inner loop over offsets is unrolled at JAX trace time.

Useful for GP covariance matrices whose kernel has compact support — e.g.
the trapezoidal AnalyticKernel, which is exactly zero beyond lag > lspot + 2*tau.

Functions
---------
banded_cholesky(A, b)       Cholesky factor of a banded SPD matrix.
banded_solve(L, rhs)        Solve A x = rhs given its Cholesky factor.
"""

import jax
import jax.numpy as jnp

__all__ = ["banded_cholesky", "banded_solve"]


def banded_cholesky(A, b):
    """
    Cholesky factorization of a symmetric positive definite banded matrix.

    Uses ``jax.lax.scan`` over columns, unrolling the inner loop over the
    b sub-diagonal offsets at trace time so b must be a Python ``int``.

    Parameters
    ----------
    A : jnp.ndarray, shape (n, n)
        Symmetric positive definite matrix. Only entries within distance b
        of the diagonal are read; all others are ignored.
    b : int
        Bandwidth (compile-time constant). Must satisfy 0 <= b < n.

    Returns
    -------
    L : jnp.ndarray, shape (n, n)
        Lower triangular Cholesky factor such that ``L @ L.T == A``
        (within the band). Entries outside the band are zero.
    """
    n = A.shape[0]

    def step(L, i):
        # Row i of L has been partially filled by previous steps;
        # L[i, 0:i] contains the already-computed off-diagonal entries.
        row_i = L[i]

        # Diagonal: L[i,i] = sqrt(A[i,i] - sum_k L[i,k]^2)
        lii = jnp.sqrt(A[i, i] - jnp.dot(row_i, row_i))

        # Sub-diagonal entries L[i+k, i] for k = 1 .. b
        # L[i+k, i] = (A[i+k, i] - dot(L[i+k, :], L[i, :])) / L[i,i]
        # At this point L[i+k, :i] holds previously computed values;
        # L[i+k, i] is still 0, so the dot product is correct.
        #
        # Guard: when j = i+k >= n, use jnp.where to skip the write entirely
        # rather than clamping the index (clamping would overwrite valid entries).
        new_col = jnp.zeros(n).at[i].set(lii)
        for k in range(1, b + 1):          # unrolled at trace time
            j = i + k
            j_safe = jnp.minimum(j, n - 1)  # safe index for array reads
            val = jnp.where(
                j < n,
                (A[j_safe, i] - jnp.dot(L[j_safe], row_i)) / lii,
                0.0,
            )
            # Only write when j is a valid row; otherwise leave new_col unchanged
            new_col = jnp.where(j < n, new_col.at[j_safe].set(val), new_col)

        return L.at[:, i].set(new_col), None

    L, _ = jax.lax.scan(step, jnp.zeros_like(A), jnp.arange(n))
    return L


def _banded_solve_vec(L, rhs):
    """
    Solve A x = rhs for a 1-D rhs given the lower Cholesky factor L of A.

    Two sequential scans: forward substitution (L y = rhs) then
    backward substitution (L.T x = y).

    Parameters
    ----------
    L   : (n, n) lower triangular Cholesky factor.
    rhs : (n,) right-hand side vector.

    Returns
    -------
    x : (n,) solution vector.
    """
    n = L.shape[0]

    # Forward substitution: L y = rhs
    def fwd(y, i):
        yi = (rhs[i] - jnp.dot(L[i], y)) / L[i, i]
        return y.at[i].set(yi), None

    y, _ = jax.lax.scan(fwd, jnp.zeros(n), jnp.arange(n))

    # Backward substitution: L.T x = y  (scan in reverse via n-1-i)
    # When processing actual = n-1-i, all x[actual+1:] are already set
    # and x[actual] is still 0, so dot(L[:, actual], x) correctly sums
    # only the super-diagonal contributions.
    def bwd(x, i):
        actual = n - 1 - i
        xi = (y[actual] - jnp.dot(L[:, actual], x)) / L[actual, actual]
        return x.at[actual].set(xi), None

    x, _ = jax.lax.scan(bwd, jnp.zeros(n), jnp.arange(n))
    return x


def banded_solve(L, rhs):
    """
    Solve A x = rhs given the lower Cholesky factor L of A.

    Parameters
    ----------
    L   : jnp.ndarray, shape (n, n)
        Lower triangular Cholesky factor from ``banded_cholesky``.
    rhs : jnp.ndarray, shape (n,) or (n, k)
        Right-hand side. For a matrix rhs, each column is solved
        independently via ``jax.vmap``.

    Returns
    -------
    x : jnp.ndarray, same shape as rhs.
    """
    if rhs.ndim == 1:
        return _banded_solve_vec(L, rhs)
    # Matrix rhs: vmap over columns
    return jax.vmap(
        lambda col: _banded_solve_vec(L, col), in_axes=1, out_axes=1
    )(rhs)
