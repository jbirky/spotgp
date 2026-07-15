"""
Banded Cholesky factorization and triangular solve for JAX.

Provides O(n * b²) factorization for symmetric positive definite matrices
whose non-zero entries are confined to a band of width b around the diagonal.
The bandwidth b must be a Python integer (compile-time constant) so that
array shapes are known at trace time.

Uses compact banded storage of shape (b+1, n) at the interface, giving true
O(n·b²) compute and O(n·b) memory.

Algorithm
---------
A banded matrix with bandwidth b, partitioned into blocks of size m >= b,
is block-tridiagonal: only the diagonal blocks D_k and the first
sub-diagonal blocks E_k are non-zero.  The Cholesky factor is then block
lower-bidiagonal (diagonal blocks L_k, sub-diagonal blocks F_k) and is
computed by the recurrence::

    F_{k-1} = E_{k-1} L_{k-1}^{-T}          (triangular solve)
    L_k     = chol(D_k - F_{k-1} F_{k-1}^T) (dense m x m Cholesky)

Each step is a chunky dense operation, so the sequential ``lax.scan`` has
only ceil(n/m) steps instead of the n scalar-column steps of the textbook
banded algorithm.  That matters for XLA: per-step dispatch overhead and
the reverse-mode sweep both scale with scan length, so the blocked form is
dramatically faster on CPU and GPU while performing the same O(n·b²)
arithmetic.  The forward/backward substitutions use the analogous block
recurrences.  All operations are standard JAX primitives with well-defined
JVP/VJP rules, so the factorization and solve are differentiable.

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

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsla

__all__ = [
    "banded_cholesky_compact", "banded_solve_compact",
    "banded_cholesky", "banded_solve",
]

# Minimum block size.  Larger blocks shorten the sequential scan (fewer,
# chunkier steps); the extra arithmetic on the zero-padded region of a
# block is negligible for m up to ~this size.
_MIN_BLOCK = 64


def _block_size(b, n):
    """Block size m for the block-tridiagonal reformulation.

    Must satisfy b <= m <= n so that the band of width b couples only
    adjacent blocks.  ``b <= n`` is assumed (callers clamp b to n - 1).
    """
    return min(n, max(b, _MIN_BLOCK))


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


def _pad_compact(Xc, n_pad, diag_val):
    """Pad compact storage with extra columns holding ``diag_val`` on the
    diagonal and zeros elsewhere (i.e. append an identity-scaled block)."""
    n = Xc.shape[1]
    if n_pad == n:
        return Xc
    pad = jnp.zeros((Xc.shape[0], n_pad - n), dtype=Xc.dtype)
    pad = pad.at[0, :].set(diag_val)
    return jnp.concatenate([Xc, pad], axis=1)


def _compact_to_blocks(Xc_p, b, m, M, symmetrize):
    """
    Gather block-tridiagonal blocks from padded compact storage.

    Parameters
    ----------
    Xc_p : jnp.ndarray, shape (b+1, M*m)
        Compact lower-banded storage, padded to a whole number of blocks.
    b, m, M : int
        Bandwidth, block size, number of blocks.
    symmetrize : bool
        If True, mirror the lower triangle into the upper triangle of the
        diagonal blocks (for SPD input matrices).  If False, keep the
        diagonal blocks lower triangular (for Cholesky factors).

    Returns
    -------
    Dg : jnp.ndarray, shape (M, m, m)
        Diagonal blocks.
    Eg : jnp.ndarray, shape (M, m, m)
        Sub-diagonal coupling blocks, aligned so that ``Eg[k]`` is the
        block at block-row k, block-column k-1 (``Eg[0]`` is zero).
    """
    r = jnp.arange(m)[:, None]      # (m, 1)
    c = jnp.arange(m)[None, :]      # (1, m)
    base = (jnp.arange(M) * m)[:, None, None]   # (M, 1, 1)
    cols = base + c[None]                        # (M, m, m)

    # Diagonal blocks: entry (r, c) sits on band diagonal d = r - c
    dD = r - c                                   # (m, m)
    maskD = (dD >= 0) & (dD <= b)
    dDs = jnp.clip(dD, 0, b)
    Dg = jnp.where(maskD[None], Xc_p[dDs, cols], 0.0)
    if symmetrize:
        strict_lower = maskD & (dD > 0)
        Dg = Dg + jnp.swapaxes(
            jnp.where(strict_lower[None], Dg, 0.0), 1, 2)

    # Sub-diagonal blocks (block-row k, block-col k-1): d = m + r - c
    dE = m + dD
    maskE = dE <= b                              # dE >= 1 always
    dEs = jnp.clip(dE, 0, b)
    cols_prev = cols - m
    valid_prev = cols_prev >= 0                  # False for block k = 0
    cols_prev_s = jnp.clip(cols_prev, 0, None)
    Eg = jnp.where(maskE[None] & valid_prev,
                   Xc_p[dEs, cols_prev_s], 0.0)

    return Dg, Eg


def _blocks_to_compact(L_blocks, F_blocks, b, m, n):
    """
    Scatter block-bidiagonal Cholesky blocks back to compact (b+1, n)
    storage, dropping any padding columns.

    ``F_blocks[k]`` is the block at block-row k, block-column k-1.
    """
    M = L_blocks.shape[0]
    d = jnp.arange(b + 1)[:, None]   # (b+1, 1)
    j = jnp.arange(n)[None, :]       # (1, n)
    k = j // m
    c = j % m
    r = c + d                        # (b+1, n)

    in_block = r < m
    r_in = jnp.clip(r, 0, m - 1)
    vals_in = L_blocks[k, r_in, c]

    r_out = jnp.clip(r - m, 0, m - 1)
    k_next = jnp.clip(k + 1, 0, M - 1)
    vals_out = F_blocks[k_next, r_out, c]
    has_next = (k + 1) < M

    Lc = jnp.where(in_block, vals_in,
                   jnp.where(has_next, vals_out, 0.0))
    # Entries whose row j + d falls outside the original matrix are zero
    return jnp.where(j + d < n, Lc, 0.0)


# ── Core factorization ──────────────────────────────────────────

@partial(jax.jit, static_argnums=1)
def banded_cholesky_compact(Ac, b):
    """
    Cholesky factorization of a symmetric positive definite banded matrix.

    JIT-compiled with ``b`` static, so eager callers (covariance builds,
    prediction) reuse the compiled block kernels across calls; inside an
    outer ``jax.jit`` the nested jit simply inlines.

    Operates in compact (b+1, n) storage at the interface — no (n, n)
    matrix is ever allocated.  Internally the matrix is viewed as a
    block-tridiagonal matrix with block size m >= b and factorized by a
    ``lax.scan`` of ceil(n/m) dense block steps (see module docstring),
    which is far faster than a scalar per-column scan while performing
    the same O(n·b²) arithmetic.

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
    m = _block_size(b, n)
    M = -(-n // m)  # ceil(n / m)

    # Pad with an identity block so n divides into whole blocks; the
    # factor of blockdiag(A, I) is blockdiag(L, I), so padding does not
    # perturb the factor of the original matrix.
    Ac_p = _pad_compact(Ac, M * m, 1.0)
    D, E_prev = _compact_to_blocks(Ac_p, b, m, M, symmetrize=True)

    def step(L_prev, xs):
        D_k, E_km1 = xs
        # F_{k-1}^T = L_{k-1}^{-1} E_{k-1}^T ; E_prev[0] = 0 gives F = 0
        Ft = jsla.solve_triangular(L_prev, jnp.swapaxes(E_km1, 0, 1),
                                   lower=True)
        S = D_k - Ft.T @ Ft
        L_k = jnp.linalg.cholesky(S)
        return L_k, (L_k, Ft.T)

    L_init = jnp.eye(m, dtype=Ac.dtype)
    _, (L_blocks, F_blocks) = jax.lax.scan(step, L_init, (D, E_prev))

    return _blocks_to_compact(L_blocks, F_blocks, b, m, n)


# ── Banded triangular solve ─────────────────────────────────────

def _banded_solve_vec_compact(Lc, rhs, b):
    """
    Solve A x = rhs for a 1-D rhs given compact Cholesky factor Lc.

    Uses the block-bidiagonal form of L: forward substitution
    ``y_k = L_k^{-1} (r_k - F_k y_{k-1})`` followed by backward
    substitution ``x_k = L_k^{-T} (y_k - F_{k+1}^T x_{k+1})``, each a
    ``lax.scan`` of ceil(n/m) dense block steps.

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
    m = _block_size(b, n)
    M = -(-n // m)
    n_pad = M * m

    # Pad the factor with an identity block and the rhs with zeros:
    # the padded solution is zero there and the real part is unchanged.
    Lc_p = _pad_compact(Lc, n_pad, 1.0)
    L_blocks, F_blocks = _compact_to_blocks(Lc_p, b, m, M,
                                            symmetrize=False)
    r_blocks = jnp.concatenate(
        [rhs, jnp.zeros(n_pad - n, dtype=rhs.dtype)]).reshape(M, m)

    # Forward substitution: L y = rhs
    def fwd(y_prev, xs):
        L_k, F_k, r_k = xs
        y_k = jsla.solve_triangular(L_k, r_k - F_k @ y_prev, lower=True)
        return y_k, y_k

    zeros_m = jnp.zeros(m, dtype=rhs.dtype)
    _, y = jax.lax.scan(fwd, zeros_m, (L_blocks, F_blocks, r_blocks))

    # Backward substitution: L^T x = y  (F_next[k] = F_{k+1}, zero at k = M-1)
    F_next = jnp.concatenate(
        [F_blocks[1:], jnp.zeros((1, m, m), dtype=Lc.dtype)])

    def bwd(x_next, xs):
        L_k, F_k1, y_k = xs
        x_k = jsla.solve_triangular(
            L_k.T, y_k - jnp.swapaxes(F_k1, 0, 1) @ x_next, lower=False)
        return x_k, x_k

    _, x = jax.lax.scan(bwd, zeros_m, (L_blocks, F_next, y), reverse=True)
    return x.reshape(-1)[:n]


@partial(jax.jit, static_argnums=2)
def banded_solve_compact(Lc, rhs, b):
    """
    Solve A x = rhs given the compact lower Cholesky factor Lc of A.

    JIT-compiled with ``b`` static (see ``banded_cholesky_compact``).

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
