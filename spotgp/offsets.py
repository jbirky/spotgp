"""
offsets.py — analytically marginalized per-segment flux offsets.

Kepler quarters and TESS sectors each carry an unknown flux zero point:
the aperture, CCD position, crowding and focus all change at a segment
boundary.  The usual fix is to subtract each segment's own median before
stitching.  That is a *plug-in* estimate — it asserts the offset with
zero uncertainty — and it annihilates all power below ``1/T_segment``,
which places a hard ceiling on the recoverable spot lifetime and reports
it as a confident answer rather than a wide posterior.

The offsets enter the model linearly, so they can be integrated out in
closed form instead.  With :math:`y = Ma + f + \\epsilon`,
:math:`f \\sim \\mathcal{GP}(0, K)`, a flat prior on :math:`a`, and
:math:`C = K + \\mathrm{diag}(\\sigma^2)`:

.. math::

    A = M^{\\top}C^{-1}M, \\qquad b = M^{\\top}C^{-1}y

    \\log L = -\\tfrac{1}{2}\\left(y^{\\top}C^{-1}y - b^{\\top}A^{-1}b
             + \\log|C| + \\log|A| + (N-S)\\log 2\\pi\\right)

Only :math:`C^{-1}y` and :math:`C^{-1}M` are needed, so the cost is
``S`` extra triangular solves on top of the one the likelihood already
does.  Those are ``O(N*b)`` each while the Cholesky factorization is
``O(N*b^2)``, so for a typical bandwidth of several hundred the overhead
is a few percent — and, critically, the banded structure is preserved.
Adding the offsets to the kernel instead, as ``K + Lambda*M M^T``, would
make each segment block dense and destroy the bandwidth ``GPSolver``
depends on.

The defining property of the flat-prior result is that the likelihood is
*exactly* invariant to adding an arbitrary constant to any segment of
``y``.  ``tests/test_offsets.py`` asserts this.

Gaussian priors
---------------
Any column can instead carry a Gaussian prior :math:`a_j \\sim
\\mathcal{N}(\\mu_j, \\sigma_j^2)`, e.g. cotrending basis vectors whose
weights should be shrunk rather than fitted freely.  With
:math:`\\Lambda = \\mathrm{diag}(\\sigma_G^2)` over the Gaussian columns
:math:`G`, flat priors on the remaining :math:`S_F` columns, and
:math:`r = y - M\\mu` (:math:`\\mu_j = 0` on flat columns):

.. math::

    A = M^{\\top}C^{-1}M + \\mathrm{diag}(1/\\sigma^2), \\qquad
    b = M^{\\top}C^{-1}r

    \\log L = -\\tfrac{1}{2}\\left(r^{\\top}C^{-1}r - b^{\\top}A^{-1}b
             + \\log|C| + \\log|A| + \\log|\\Lambda| + (N-S_F)\\log 2\\pi\\right)

(:math:`1/\\sigma^2 = 0` on flat columns).  This is the Woodbury form of
integrating the Gaussian columns into the covariance,
:math:`C + M_G\\Lambda M_G^{\\top}`, and then the flat ones as above; it
still needs only :math:`C^{-1}r` and :math:`C^{-1}M`, so the banded
structure is preserved.  With every column flat it reduces exactly to the
flat-prior formula.  Shifting :math:`y` by :math:`M_G c` is equivalent to
shifting :math:`\\mu_G` by :math:`-c`.

Usage
-----
::

    from spotgp import GPSolver
    from spotgp.offsets import segment_labels

    labels = segment_labels(data.x, gap=0.5)      # split at gaps > 0.5 d
    gp = GPSolver(data, model, offsets=labels)
    gp.fit_map()
    a_hat = gp.offset_estimates()                 # posterior mean offsets

    # flat quarter offsets plus CBV columns with a N(0, 0.05^2) prior
    M = np.hstack([offset_design(x, labels), cbvs])
    sigma = np.r_[np.full(n_quarters, np.inf), np.full(cbvs.shape[1], 0.05)]
    gp = GPSolver(data, model, offsets=M, offset_prior=sigma)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["segment_labels", "offset_design", "apply_offset_marginalization",
           "offset_posterior", "resolve_offset_prior", "offset_prior_constants"]


def segment_labels(x, gap):
    """
    Label contiguous segments of ``x``, splitting wherever the spacing
    exceeds ``gap``.

    For Kepler use a gap comparable to a downlink (~0.5 d) to get one
    segment per contiguous block, or larger (~5 d) to get one per
    quarter.  For TESS, ~1 d splits at the mid-sector downlink, which is
    usually the right boundary: scattered light and pointing differ
    between a sector's two orbits, so a per-sector offset leaves a step
    in the middle.

    Parameters
    ----------
    x : array_like, shape (N,)
        Observation times. Need not be sorted.
    gap : float
        Split wherever consecutive (sorted) times differ by more than
        this, in the same units as ``x``.

    Returns
    -------
    labels : ndarray of int, shape (N,)
        Segment index per point, numbered 0..S-1 in time order.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    breaks = np.diff(x[order]) > float(gap)
    labels_sorted = np.concatenate([[0], np.cumsum(breaks)]).astype(int)

    labels = np.empty(len(x), dtype=int)
    labels[order] = labels_sorted
    return labels


def offset_design(x, labels, order=0):
    """
    Build the design matrix ``M`` of per-segment nuisance functions.

    ``order=0`` gives one indicator column per segment (a free constant).
    ``order=1`` adds a linear ramp per segment, ``order=2`` a quadratic,
    and so on.  Ramp columns are centred and scaled to [-1, 1] within
    their segment so that ``M`` stays well conditioned.

    Every column you add buys robustness against a systematic and spends
    low-frequency information: ``M`` is exactly the dial trading the two.
    Marginalizing rather than fitting is what makes that cost show up in
    the error bars instead of the point estimate.

    Parameters
    ----------
    x : array_like, shape (N,)
        Observation times.
    labels : array_like of int, shape (N,)
        Segment index per point (see :func:`segment_labels`).
    order : int
        Highest polynomial order per segment (0 = offset only).

    Returns
    -------
    M : ndarray, shape (N, S * (order + 1))
    """
    x = np.asarray(x, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if x.shape != labels.shape:
        raise ValueError(
            f"x and labels must have the same shape, got {x.shape} "
            f"and {labels.shape}")
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")

    cols = []
    for s in np.unique(labels):
        mask = labels == s
        cols.append(mask.astype(float))

        if order > 0:
            xs = x[mask]
            centred = xs - xs.mean()
            scale = np.abs(centred).max()
            if scale == 0:            # single-point segment: ramps are degenerate
                scale = 1.0
            for p in range(1, order + 1):
                col = np.zeros(len(x))
                col[mask] = (centred / scale) ** p
                cols.append(col)

    return np.column_stack(cols)


def resolve_offset_prior(prior, n_columns):
    """
    Normalize a prior specification for the ``n_columns`` design columns.

    Parameters
    ----------
    prior : None, float, array_like, tuple or dict
        - ``None``: flat prior on every column.
        - a float: Gaussian width ``sigma`` shared by every column, mean 0.
        - an array of shape (S,): per-column ``sigma``; ``np.inf`` marks a
          flat column.
        - ``(mu, sigma)`` or ``dict(mu=..., sigma=...)``: means and widths,
          each a scalar or shape (S,).  ``mu`` is ignored on flat columns.

    Returns
    -------
    mu, sigma : ndarray, shape (S,)
        ``sigma`` is ``inf`` on flat columns, where ``mu`` is set to 0.
    """
    S = int(n_columns)
    if prior is None:
        return np.zeros(S), np.full(S, np.inf)
    if isinstance(prior, dict):
        spec = dict(prior)
        mu, sigma = spec.pop("mu", 0.0), spec.pop("sigma", None)
        if spec:
            raise ValueError(f"Unknown keys in offset_prior: {sorted(spec)}. "
                             f"Expected 'mu' and/or 'sigma'.")
        if sigma is None:
            raise ValueError("offset_prior dict requires a 'sigma' entry.")
    elif isinstance(prior, tuple):
        if len(prior) != 2:
            raise ValueError("offset_prior tuple must be (mu, sigma).")
        mu, sigma = prior
    else:
        mu, sigma = 0.0, prior

    mu = np.broadcast_to(np.asarray(mu, dtype=float), (S,)).copy()
    sigma = np.broadcast_to(np.asarray(sigma, dtype=float), (S,)).copy()
    if np.any(np.isnan(sigma)) or np.any(sigma <= 0):
        raise ValueError("offset_prior sigma must be positive (np.inf = flat).")
    if np.any(~np.isfinite(mu)):
        raise ValueError("offset_prior mu must be finite.")
    mu[np.isinf(sigma)] = 0.0
    return mu, sigma


def offset_prior_constants(sigma):
    """
    Constants the marginal likelihood needs from the prior widths.

    Returns
    -------
    precision : ndarray, shape (S,)
        ``1 / sigma**2`` (0 on flat columns).
    logdet : float
        :math:`\\log|\\Lambda|` over the Gaussian columns.
    n_flat : int
        Number of flat columns (sets the ``(N - S_F) log 2 pi`` term).
    """
    sigma = np.asarray(sigma, dtype=float)
    finite = np.isfinite(sigma)
    precision = np.where(finite, 1.0 / np.where(finite, sigma, 1.0) ** 2, 0.0)
    logdet = float(np.sum(2.0 * np.log(sigma[finite])))
    return precision, logdet, int((~finite).sum())


def apply_offset_marginalization(data_fit, log_det, alpha, design, Z,
                                 prior_precision=None, prior_logdet=0.0):
    """
    Fold the marginalized offsets into an existing quadratic form and
    log-determinant.

    Both likelihood paths in ``gp_solver`` reduce to ``data_fit``
    (:math:`r^{\\top}C^{-1}r`) and ``log_det`` (:math:`\\log|C|`); this
    applies the corrections that integrate the offsets out.

    Parameters
    ----------
    data_fit : scalar
        :math:`r^{\\top}C^{-1}r`.
    log_det : scalar
        :math:`\\log|C|`.
    alpha : jnp.ndarray, shape (N,)
        :math:`C^{-1}r`.
    design : jnp.ndarray, shape (N, S)
        The design matrix :math:`M`.
    Z : jnp.ndarray, shape (N, S)
        :math:`C^{-1}M`.
    prior_precision : jnp.ndarray, shape (S,), optional
        :math:`1/\\sigma^2` per column (0 = flat).  None: all flat.
    prior_logdet : float
        :math:`\\log|\\Lambda|` over the Gaussian columns (0 if none).

    Returns
    -------
    data_fit, log_det : scalars
        Corrected values, ready for the usual
        ``-0.5 * (data_fit + log_det + (N - S_flat) * log(2*pi))``.
    """
    A = design.T @ Z
    if prior_precision is not None:
        A = A + jnp.diag(prior_precision)
    b = design.T @ alpha

    # Cholesky rather than an explicit inverse: A is small (S x S) but can
    # be poorly conditioned when a segment holds few points.
    La = jnp.linalg.cholesky(A)
    w = jax.scipy.linalg.solve_triangular(La, b, lower=True)

    return (data_fit - w @ w,
            log_det + 2.0 * jnp.sum(jnp.log(jnp.diag(La))) + prior_logdet)


def offset_posterior(alpha, design, Z, prior_precision=None):
    """
    Posterior mean and covariance of the offsets, given the solves.

    The conditional posterior is Gaussian with mean :math:`A^{-1}b` and
    covariance :math:`A^{-1}`, where ``A`` includes the prior precision.
    With a Gaussian prior, ``alpha`` must be :math:`C^{-1}(y - M\\mu)` and
    the returned mean is the offset *relative to* :math:`\\mu`.

    Returns
    -------
    a_hat : jnp.ndarray, shape (S,)
    a_cov : jnp.ndarray, shape (S, S)
    """
    A = design.T @ Z
    if prior_precision is not None:
        A = A + jnp.diag(prior_precision)
    b = design.T @ alpha
    a_cov = jnp.linalg.inv(A)
    return a_cov @ b, a_cov
