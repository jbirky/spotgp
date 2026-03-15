"""
Banded Cholesky factorization and triangular solve for JAX.

Provides O(n * b²) factorization for symmetric positive definite matrices
whose non-zero entries are confined to a band of width b around the diagonal.
The bandwidth b must be a Python integer (compile-time constant) so that
array shapes are known at trace time, but all inner loops use
``jax.lax.fori_loop`` to keep the XLA graph compact.

Uses compact banded storage of shape (b+1, n) throughout, giving true
O(n·b²) compute and O(n·b) memory.

Compact storage convention
--------------------------
A lower-triangular banded matrix L with bandwidth b has at most (b+1)
non-zero entries per column.  We store it in a (b+1, n) array ``Lc`` where::

    Lc[d, j] = L[j + d, j]     for d = 0 .. b

So ``Lc[0, :]`` holds the diagonal, ``Lc[1, :]`` the first sub-diagonal, etc.

Functions
---------
banded_cholesky_compact(Ac, b)  Cholesky in compact storage.
banded_solve_compact(Lc, rhs, b)  Solve A x = rhs from compact Cholesky factor.
banded_cholesky(A, b)           Legacy wrapper: full (n, n) in/out.
banded_solve(L, rhs, b)         Legacy wrapper: full (n, n) in.
"""

import jax
import jax.numpy as jnp

__all__ = [
    "banded_cholesky_compact", "banded_solve_compact",
    "banded_cholesky", "banded_solve",
]


# ── Compact banded storage helpers ──────────────────────────────

def _full_to_compact(A, b):
    """Extract lower band of (n, n) matrix A into compact (b+1, n) storage."""
    n = A.shape[0]
    j_idx = jnp.arange(n)
    rows = []
    for d in range(b + 1):
        i_idx = jnp.minimum(j_idx + d, n - 1)
        vals = A[i_idx, j_idx]
        vals = jnp.where(j_idx + d < n, vals, 0.0)
        rows.append(vals)
    return jnp.stack(rows, axis=0)  # (b+1, n)


def _compact_to_full(Lc, n):
    """Expand compact (b+1, n) storage back to full (n, n) lower triangular."""
    bp1 = Lc.shape[0]
    L = jnp.zeros((n, n))
    j_idx = jnp.arange(n)
    for d in range(bp1):
        i_idx = j_idx + d
        mask = i_idx < n
        i_safe = jnp.minimum(i_idx, n - 1)
        L = L.at[i_safe, j_idx].add(jnp.where(mask, Lc[d], 0.0))
    return L


# ── Core factorization in compact storage ───────────────────────

def banded_cholesky_compact(Ac, b):
    """
    Cholesky factorization of a symmetric positive definite banded matrix.

    Operates entirely in compact (b+1, n) storage — no (n, n) matrix
    is ever allocated.

    Parameters
    ----------
    Ac : jnp.ndarray, shape (b+1, n)
        Input matrix in compact lower-banded storage:
        ``Ac[d, j] = A[j+d, j]`` for d = 0 .. b.
    b : int
        Bandwidth (compile-time constant). Must satisfy 0 <= b < n.

    Returns
    -------
    Lc : jnp.ndarray, shape (b+1, n)
        Lower Cholesky factor in compact storage:
        ``Lc[d, j] = L[j+d, j]``.
    """
    n = Ac.shape[1]

    def step(Lc, j):
        # ── Diagonal ──
        # Lc[0, j] = sqrt(Ac[0, j] - sum_{s=1..b} Lc[s, j-s]^2)
        s_idx = jnp.arange(1, b + 1)           # [1, 2, ..., b]
        k_idx = j - s_idx                       # [j-1, j-2, ..., j-b]
        k_safe = jnp.maximum(k_idx, 0)
        valid = k_idx >= 0
        vals = Lc[s_idx, k_safe]
        vals = jnp.where(valid, vals, 0.0)
        diag_sum = jnp.dot(vals, vals)

        ljj = jnp.sqrt(Ac[0, j] - diag_sum)
        Lc = Lc.at[0, j].set(ljj)

        # ── Sub-diagonal entries for d = 1..b ──
        d_idx = jnp.arange(1, b + 1)  # [1, 2, ..., b]

        ds_idx = d_idx[:, None] + s_idx[None, :]  # (b, b)
        ds_safe = jnp.minimum(ds_idx, b)
        row_vals = Lc[ds_safe, k_safe[None, :]]    # (b, b)
        mask = (ds_idx <= b) & valid[None, :]       # (b, b)
        row_vals = jnp.where(mask, row_vals, 0.0)

        dots = row_vals @ vals  # (b,)

        new_vals = (Ac[d_idx, j] - dots) / ljj     # (b,)
        i_idx = j + d_idx
        new_vals = jnp.where(i_idx < n, new_vals, 0.0)

        Lc = Lc.at[d_idx, j].set(new_vals)

        return Lc, None

    Lc = jnp.zeros((b + 1, n))
    Lc, _ = jax.lax.scan(step, Lc, jnp.arange(n))

    return Lc


# ── Banded triangular solve in compact storage ──────────────────

def _banded_solve_vec_compact(Lc, rhs, b):
    """
    Solve A x = rhs for a 1-D rhs given compact Cholesky factor Lc.

    Parameters
    ----------
    Lc  : (b+1, n) compact lower Cholesky factor.
    rhs : (n,) right-hand side vector.
    b   : int  bandwidth.

    Returns
    -------
    x : (n,) solution vector.
    """
    n = Lc.shape[1]
    s_idx = jnp.arange(1, b + 1)  # [1, 2, ..., b]

    # Forward substitution: L y = rhs
    # y[i] = (rhs[i] - sum_{s=1..b} L[i, i-s] * y[i-s]) / L[i,i]
    # L[i, i-s] = Lc[s, i-s]
    def fwd_banded(y, i):
        k_idx = i - s_idx
        k_safe = jnp.maximum(k_idx, 0)
        valid = k_idx >= 0
        l_vals = jnp.where(valid, Lc[s_idx, k_safe], 0.0)
        y_vals = jnp.where(valid, y[k_safe], 0.0)
        yi = (rhs[i] - jnp.dot(l_vals, y_vals)) / Lc[0, i]
        return y.at[i].set(yi), None

    y, _ = jax.lax.scan(fwd_banded, jnp.zeros(n), jnp.arange(n))

    # Backward substitution: L^T x = y
    # x[i] = (y[i] - sum_{s=1..b} L[i+s, i] * x[i+s]) / L[i,i]
    # L[i+s, i] = Lc[s, i]
    def bwd_banded(x, idx):
        i = n - 1 - idx
        j_idx = i + s_idx
        j_safe = jnp.minimum(j_idx, n - 1)
        valid = j_idx < n
        l_vals = jnp.where(valid, Lc[s_idx, i], 0.0)
        x_vals = jnp.where(valid, x[j_safe], 0.0)
        xi = (y[i] - jnp.dot(l_vals, x_vals)) / Lc[0, i]
        return x.at[i].set(xi), None

    x, _ = jax.lax.scan(bwd_banded, jnp.zeros(n), jnp.arange(n))
    return x


def banded_solve_compact(Lc, rhs, b):
    """
    Solve A x = rhs given the compact lower Cholesky factor Lc of A.

    Parameters
    ----------
    Lc  : jnp.ndarray, shape (b+1, n)
        Lower Cholesky factor in compact banded storage.
    rhs : jnp.ndarray, shape (n,) or (n, k)
        Right-hand side. For a matrix rhs, each column is solved
        independently via ``jax.vmap``.
    b   : int
        Bandwidth.

    Returns
    -------
    x : jnp.ndarray, same shape as rhs.
    """
    if rhs.ndim == 1:
        return _banded_solve_vec_compact(Lc, rhs, b)
    # Matrix rhs: vmap over columns
    return jax.vmap(
        lambda col: _banded_solve_vec_compact(Lc, col, b),
        in_axes=1, out_axes=1,
    )(rhs)


# ── Legacy wrappers (full N×N in/out) ───────────────────────────

def banded_cholesky(A, b):
    """
    Cholesky factorization of a symmetric positive definite banded matrix.

    Legacy wrapper that accepts/returns full (n, n) matrices.
    Prefer ``banded_cholesky_compact`` for O(n·b) memory.

    Parameters
    ----------
    A : jnp.ndarray, shape (n, n)
        Symmetric positive definite matrix.
    b : int
        Bandwidth (compile-time constant).

    Returns
    -------
    L : jnp.ndarray, shape (n, n)
        Lower triangular Cholesky factor.
    """
    n = A.shape[0]
    Ac = _full_to_compact(A, b)
    Lc = banded_cholesky_compact(Ac, b)
    return _compact_to_full(Lc, n)


def banded_solve(L, rhs, b=None):
    """
    Solve A x = rhs given the lower Cholesky factor L of A.

    Legacy wrapper that accepts a full (n, n) Cholesky factor.
    Prefer ``banded_solve_compact`` for O(n·b) memory.

    Parameters
    ----------
    L   : jnp.ndarray, shape (n, n)
        Lower triangular Cholesky factor from ``banded_cholesky``.
    rhs : jnp.ndarray, shape (n,) or (n, k)
        Right-hand side.
    b   : int or None
        Bandwidth. If None, falls back to full O(n²) dot products.

    Returns
    -------
    x : jnp.ndarray, same shape as rhs.
    """
    n = L.shape[0]

    if b is None:
        # Fallback: full dot products
        def fwd(y, i):
            yi = (rhs[i] - jnp.dot(L[i], y)) / L[i, i]
            return y.at[i].set(yi), None

        y, _ = jax.lax.scan(fwd, jnp.zeros(n), jnp.arange(n))

        def bwd(x, i):
            actual = n - 1 - i
            xi = (y[actual] - jnp.dot(L[:, actual], x)) / L[actual, actual]
            return x.at[actual].set(xi), None

        x, _ = jax.lax.scan(bwd, jnp.zeros(n), jnp.arange(n))
        return x

    # Convert to compact and use the compact solver
    Lc = _full_to_compact(L, b)
    return banded_solve_compact(Lc, rhs, b)
