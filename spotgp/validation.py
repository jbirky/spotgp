"""Data validation and user-friendly error formatting for spotgp."""

import logging
import warnings

import numpy as np

logger = logging.getLogger("spotgp")

__all__ = [
    "validate_data",
    "validate_data_vs_model",
    "CholeskyWarning",
    "warn_cholesky_failure",
    "format_nan_gradient_warning",
]


class CholeskyWarning(UserWarning):
    """The GP covariance matrix could not be Cholesky-factorized.

    JAX does not raise on a matrix that is not positive definite: the
    factor fills with NaN, and so do the log-likelihood and its
    gradient.  The solver emits this warning when it detects that.
    """
    pass


# ── Data validation ──────────────────────────────────────────────────────

def validate_data(x, y, yerr):
    """Check data arrays for common problems and warn.

    Called during GPSolver construction.  All checks emit warnings
    rather than errors so that advanced users can proceed with
    unusual data if they know what they are doing.

    Parameters
    ----------
    x, y, yerr : array_like
        Observation times, flux values, and uncertainties.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    yerr = np.asarray(yerr)
    N = len(x)

    if N < 10:
        warnings.warn(
            f"Only {N} data points. The GP needs enough observations to "
            f"constrain the kernel — fits with fewer than ~20 points are "
            f"typically unreliable.",
            stacklevel=3,
        )

    if N > 0 and not np.all(np.diff(x) >= 0):
        warnings.warn(
            "Observation times are not sorted. GPSolver assumes "
            "monotonically increasing x — the banded solver will "
            "produce incorrect results on unsorted data.",
            stacklevel=3,
        )

    if N > 1 and np.all(yerr == yerr[0]):
        if yerr[0] == 0.0:
            warnings.warn(
                "All measurement uncertainties (yerr) are zero. The GP "
                "covariance matrix will have no noise term on the "
                "diagonal, which almost always causes a Cholesky failure. "
                "Set yerr to realistic measurement uncertainties, or add "
                "a JitterTerm to the kernel to fit white noise.",
                stacklevel=3,
            )
        else:
            logger.info(
                "All yerr values are identical (%.2e). This is fine if "
                "the data really has uniform noise, but real photometry "
                "usually has varying uncertainties — check that yerr "
                "was loaded correctly.", yerr[0])

    rms = np.std(y)
    if N > 1 and rms < 1e-15:
        warnings.warn(
            "The flux values have near-zero variance (std = {:.2e}). "
            "The GP has nothing to model — check that the data was "
            "loaded correctly.".format(rms),
            stacklevel=3,
        )

    if N > 1:
        median_yerr = np.median(yerr)
        if median_yerr > 0 and rms > 0 and median_yerr > 5 * rms:
            warnings.warn(
                f"Median yerr ({median_yerr:.2e}) is much larger than the "
                f"flux scatter (std = {rms:.2e}). The GP will attribute "
                f"all variability to noise and the kernel signal will be "
                f"unconstrained. Check the uncertainty units or consider "
                f"rescaling.",
                stacklevel=3,
            )


def validate_data_vs_model(x, bounds, param_keys):
    """Cross-check data properties against model configuration and warn.

    Parameters
    ----------
    x : array_like
        Observation times.
    bounds : array_like, shape (n_params, 2)
        Parameter bounds.
    param_keys : tuple of str
        Parameter names corresponding to rows of bounds.
    """
    x = np.asarray(x)
    bounds = np.asarray(bounds)
    N = len(x)
    if N < 2:
        return

    baseline = float(x[-1] - x[0])
    median_dt = float(np.median(np.diff(x)))
    # Strip term prefixes ("spot0.peq" → "peq") so the checks below see
    # composite-kernel keys; .index() then reports on the first term
    # carrying the parameter, which is enough for these heuristics.
    param_keys = [k.rpartition(".")[2] for k in param_keys]

    # Period bounds vs baseline
    if "peq" in param_keys:
        idx = param_keys.index("peq")
        peq_lo, peq_hi = float(bounds[idx, 0]), float(bounds[idx, 1])

        if peq_hi > 2 * baseline:
            warnings.warn(
                f"Period upper bound ({peq_hi:.1f} days) is more than "
                f"twice the data baseline ({baseline:.1f} days). The GP "
                f"cannot constrain periods much longer than the baseline "
                f"— consider tightening the peq upper bound to "
                f"~{baseline:.0f} days.",
                stacklevel=3,
            )

        if peq_lo < 2 * median_dt:
            warnings.warn(
                f"Period lower bound ({peq_lo:.2f} days) is shorter than "
                f"twice the cadence ({2 * median_dt:.2f} days). Periods "
                f"below the Nyquist limit are not recoverable — consider "
                f"raising the peq lower bound.",
                stacklevel=3,
            )

    # Log-space period bounds
    if "log_peq" in param_keys:
        idx = param_keys.index("log_peq")
        log_peq_hi = float(bounds[idx, 1])
        peq_hi = 10.0 ** log_peq_hi
        if peq_hi > 2 * baseline:
            warnings.warn(
                f"Period upper bound (10^{log_peq_hi:.1f} = {peq_hi:.1f} "
                f"days) is more than twice the data baseline "
                f"({baseline:.1f} days). The GP cannot constrain periods "
                f"much longer than the baseline — consider lowering the "
                f"log_peq upper bound.",
                stacklevel=3,
            )

    # Spot lifetime bounds vs baseline
    for tau_key in ("tau_spot", "tau_em", "tau_dec"):
        if tau_key in param_keys:
            idx = param_keys.index(tau_key)
            tau_hi = float(bounds[idx, 1])
            if tau_hi > baseline:
                logger.info(
                    "%s upper bound (%.1f days) exceeds the data baseline "
                    "(%.1f days). Spot lifetimes longer than the baseline "
                    "are weakly constrained.",
                    tau_key, tau_hi, baseline)

    # Number of free parameters vs data points
    n_free = len(param_keys)
    if N < 5 * n_free:
        warnings.warn(
            f"Only {N} data points for {n_free} free parameters. A "
            f"rough guideline is N > 5 * n_params for reliable "
            f"inference. Consider fixing some parameters or adding "
            f"more data.",
            stacklevel=3,
        )


# ── Cholesky failure warning ────────────────────────────────────────────

_JITTER_SUGGESTION = (
    "The solver adds only yerr**2 to the diagonal. To fit white noise and "
    "regularize the matrix, add a JitterTerm to the kernel, e.g.\n\n"
    "    KernelSum(SpotTerm(model, prefix='spot'), JitterTerm(prefix='jit'))")


def warn_cholesky_failure(context, theta=None, param_keys=None, bounds=None,
                          stacklevel=3, suggestion=None):
    """Warn that the covariance matrix failed to Cholesky-factorize.

    The solver adds nothing to the covariance diagonal except ``yerr**2``
    and any :class:`~spotgp.terms.JitterTerm` in the kernel, so a smooth
    kernel sampled densely with small uncertainties can be numerically
    singular.  By default the message suggests a ``JitterTerm`` as the
    regularizer.

    Parameters
    ----------
    context : str
        Where the failure occurred (for the message).
    theta : array_like or None
        Parameter values at the time of failure.
    param_keys : tuple of str or None
        Parameter names.
    bounds : array_like or None
        Parameter bounds, aligned with ``param_keys``.
    stacklevel : int
        Forwarded to :func:`warnings.warn`.
    suggestion : str or None
        What to try, for solvers that cannot take a ``JitterTerm``.
        Defaults to the ``JitterTerm`` suggestion.
    """
    lines = [
        f"Cholesky factorization failed during {context}: the covariance "
        "matrix is not positive definite at these parameters, so the "
        "log-likelihood is NaN.",
        "",
        _JITTER_SUGGESTION if suggestion is None else suggestion,
    ]

    if theta is not None and param_keys is not None:
        theta = np.asarray(theta)
        lines.append("")
        lines.append("Parameters at failure:")
        for i, k in enumerate(param_keys):
            if i < len(theta):
                val = float(theta[i])
                at_bound = ""
                if bounds is not None:
                    bnd = np.asarray(bounds)
                    if i < len(bnd):
                        lo, hi = float(bnd[i, 0]), float(bnd[i, 1])
                        if abs(val - lo) < 1e-6 * (hi - lo):
                            at_bound = "  [AT LOWER BOUND]"
                        elif abs(val - hi) < 1e-6 * (hi - lo):
                            at_bound = "  [AT UPPER BOUND]"
                lines.append(f"  {k:>12s} = {val:.6g}{at_bound}")

    warnings.warn("\n".join(lines), CholeskyWarning, stacklevel=stacklevel)


# ── NaN gradient formatting ─────────────────────────────────────────────

def format_nan_gradient_warning(theta, grad, param_keys, bounds):
    """Format a warning message when gradients contain NaN.

    Parameters
    ----------
    theta : array_like
        Current parameter values.
    grad : array_like
        Gradient vector (may contain NaN/Inf).
    param_keys : tuple of str
        Parameter names.
    bounds : array_like, shape (n_params, 2)
        Parameter bounds.

    Returns
    -------
    msg : str
        Human-readable warning message.
    """
    theta = np.asarray(theta)
    grad = np.asarray(grad)
    bounds = np.asarray(bounds)

    bad_idx = np.where(~np.isfinite(grad))[0]
    if len(bad_idx) == 0:
        return ""

    lines = [
        "NaN or Inf in gradient — the optimizer cannot proceed from "
        "this point.",
    ]

    for i in bad_idx:
        if i < len(param_keys):
            k = param_keys[i]
            val = float(theta[i]) if i < len(theta) else float("nan")
            lo, hi = float(bounds[i, 0]), float(bounds[i, 1])
            near = ""
            rng = hi - lo
            if rng > 0:
                frac = (val - lo) / rng
                if frac < 0.01:
                    near = " (at lower bound)"
                elif frac > 0.99:
                    near = " (at upper bound)"
            lines.append(
                f"  {k} = {val:.6g}, bounds = [{lo:.4g}, {hi:.4g}]{near}")

    lines.append(
        "This usually means the log-likelihood is -inf at these "
        "parameters (non-positive-definite covariance). The optimizer "
        "will fall back to a zero gradient for this step, but if this "
        "persists, try:"
    )
    lines.append("  - Narrowing the parameter bounds")
    lines.append("  - Using a different starting point (nopt > 1)")
    lines.append("  - Adding a JitterTerm to the kernel (the solver adds "
                 "only yerr**2 to the covariance diagonal)")

    return "\n".join(lines)
