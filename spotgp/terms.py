"""
Composable kernel terms: the additive Term / KernelSum layer.

A :class:`Term` is anything that can produce a stationary kernel
contribution ``K(tau)`` from its own slice of the flat parameter vector
and describe how it occupies that vector (``param_keys``, ``theta0``,
``default_bounds``).  A sum of stationary kernels is itself stationary,
so :class:`KernelSum` composes terms by summing their per-lag
contributions behind the one seam the solver already relies on — the
Toeplitz/banded structure, the noise diagonal, and the JIT closures are
all preserved unchanged.

:class:`SpotTerm` wraps the existing :class:`~spotgp.AnalyticKernel` /
``_kernel_eval`` machinery, so the current single-spot GP is exactly
``KernelSum(SpotTerm(model))``.
"""
import logging

import jax.numpy as jnp
import numpy as np

from .spot_model import (
    SpotEvolutionModel, EdgeOnVisibilityFunction, _gauss_legendre_grid,
)

__all__ = ["Term", "KernelSum", "SpotTerm", "DEFAULT_TERM_BOUNDS"]

logger = logging.getLogger("spotgp")


# Default bounds for spot-model parameters, keyed by bare (unprefixed)
# name.  This is the single source of truth; GPSolver.DEFAULT_BOUNDS is
# built from it for backward compatibility.
DEFAULT_TERM_BOUNDS = {
    "peq":       (0.5, 50.0),
    "kappa":     (0.001, 0.999),
    "inc":       (0.01, np.pi - 0.01),
    "lspot":     (0.1, 20.0),
    "tau_spot":  (0.05, 10.0),
    "tau_em":    (0.05, 10.0),
    "tau_dec":   (0.05, 10.0),
    "sigma_sn":  (0.05, 10.0),
    "n_sn":      (-10.0, 10.0),
    "lat_min":   (0.0, np.pi / 2),
    "lat_max":   (0.0, np.pi / 2),
    "sigma_k":   (1e-6, 1.0),
    "sigma_n":   (1e-6, 0.1),
}


def _strip_log(key):
    """'log_lspot' -> 'lspot'; other keys unchanged."""
    return key[4:] if key.startswith("log_") else key


class Term:
    """
    Base class for additive kernel components.

    A term owns a contiguous slice of the flat parameter vector and
    produces a stationary kernel contribution from it.  Subclasses must
    implement ``k_of_lag`` and the parameter-layout properties; ``psd``
    is optional.

    Attributes
    ----------
    stationary : bool
        Gates the Toeplitz/banded fast path in the solver.  A
        non-stationary term must set this to False, which forces the
        full-matrix route.
    prefix : str or None
        Namespace for this term's parameter keys.  When set, keys are
        reported as ``"<prefix>.<name>"`` (e.g. ``"spot0.peq"``).  A
        single unprefixed term reports bare keys, preserving the
        existing single-spot layout.
    """

    stationary = True
    prefix = None

    # Class tag used for auto-prefixing inside KernelSum (e.g. "spot0").
    _prefix_tag = "term"

    # ── Parameter layout ────────────────────────────────────────────────

    @property
    def base_keys(self):
        """Bare (unprefixed) parameter names, in theta-slice order."""
        raise NotImplementedError

    @property
    def param_keys(self):
        """Parameter names as seen by the solver (prefixed if set)."""
        if self.prefix is None:
            return tuple(self.base_keys)
        return tuple(f"{self.prefix}.{k}" for k in self.base_keys)

    @property
    def n_params(self):
        return len(self.base_keys)

    @property
    def theta0(self):
        """Initial physical parameter values, aligned with base_keys."""
        raise NotImplementedError

    @property
    def default_bounds(self):
        """Default (lo, hi) bounds keyed by bare parameter name."""
        return {k: DEFAULT_TERM_BOUNDS[k] for k in self.base_keys
                if k in DEFAULT_TERM_BOUNDS}

    # ── Kernel evaluation ───────────────────────────────────────────────

    def k_of_lag(self, theta_slice, lag_flat):
        """
        JAX-traceable stationary kernel contribution.

        Parameters
        ----------
        theta_slice : jnp.ndarray, shape (n_params,)
            This term's physical parameters, in ``base_keys`` order.
        lag_flat : jnp.ndarray, shape (M,)
            Time lags [days].

        Returns
        -------
        K_flat : jnp.ndarray, shape (M,)
        """
        raise NotImplementedError

    def bandwidth_support(self, param_keys, bounds_arr):
        """
        Kernel support [days] from prior upper bounds.

        Receives this term's *own* (bare, possibly log-prefixed) keys and
        the matching bounds rows; returns the longest correlation length
        so the banded approximation stays valid anywhere in the prior.
        """
        raise NotImplementedError

    def psd(self, omega, theta_slice=None):
        """Optional analytic PSD contribution (see subclasses)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement an analytic PSD")

    # ── Helpers ─────────────────────────────────────────────────────────

    def _own_bounds_rows(self, param_keys, bounds_arr):
        """
        Select this term's rows from the solver's (keys, bounds) layout.

        Solver keys may be log-remapped (``log_sigma_k``) and/or
        prefixed (``spot0.log_sigma_k``); the returned keys are stripped
        of the prefix but keep the ``log_`` marker, matching what
        ``bandwidth_support`` implementations expect.
        """
        bounds_arr = np.asarray(bounds_arr)
        own = set(self.base_keys)
        keys_out, rows = [], []
        for i, k in enumerate(param_keys):
            bare = k
            if self.prefix is not None:
                if not k.startswith(self.prefix + "."):
                    continue
                bare = k[len(self.prefix) + 1:]
            if _strip_log(bare) in own:
                keys_out.append(bare)
                rows.append(bounds_arr[i])
        return keys_out, (np.asarray(rows) if rows
                          else np.zeros((0, 2), dtype=float))


class SpotTerm(Term):
    """
    A single spot-population term wrapping :class:`AnalyticKernel`.

    Delegates kernel math to the module-level ``_kernel_eval`` (the
    single source of truth), with the envelope / latitude / visibility
    closures and quadrature configuration captured at construction —
    exactly the closures GPSolver previously built inline, so a solver
    using ``KernelSum(SpotTerm(model))`` reproduces the pre-seam
    likelihood bit-for-bit.

    Parameters
    ----------
    model_or_hparam : SpotEvolutionModel or dict
        The spot evolution model (or legacy hparam dict).
    prefix : str or None
        Optional namespace for parameter keys (e.g. ``"spot0"``).
    analytic_kernel : AnalyticKernel or None
        Reuse an already-constructed kernel object (avoids rebuilding
        envelope grids).  When given, ``model_or_hparam`` may be None.
    **kernel_kwargs
        Forwarded to :class:`AnalyticKernel` (n_harmonics, n_lat,
        lat_range, quadrature) when constructing one.
    """

    _prefix_tag = "spot"

    def __init__(self, model_or_hparam=None, prefix=None,
                 analytic_kernel=None, **kernel_kwargs):
        from .analytic_kernel import AnalyticKernel

        if analytic_kernel is not None:
            self.analytic_kernel = analytic_kernel
        elif isinstance(model_or_hparam, AnalyticKernel):
            self.analytic_kernel = model_or_hparam
        elif model_or_hparam is None:
            raise ValueError(
                "SpotTerm requires a SpotEvolutionModel, hparam dict, "
                "or analytic_kernel")
        else:
            self.analytic_kernel = AnalyticKernel(
                model_or_hparam, **kernel_kwargs)

        self.spot_model = self.analytic_kernel.spot_model
        self.prefix = prefix
        self._configure()

    def _configure(self):
        """
        Precompute the JAX-traceable closures and quadrature arrays.

        Mirrors (verbatim) the configuration GPSolver.__init__ built
        inline before the Term seam: when latitude parameters are free,
        the quadrature grid covers the full hemisphere so the dynamic
        weights can select any sub-range; Gauss-Legendre weights are
        pre-normalized (sum = 1.0) so the /norm division inside
        _kernel_eval reduces to /1.0, which XLA eliminates.
        """
        ak = self.analytic_kernel
        model = self.spot_model

        self.n_harmonics = ak.n_harmonics
        self.n_lat = ak.n_lat
        if model.latitude_distribution.param_dict:
            self.lat_range = (-np.pi / 2, np.pi / 2)
        else:
            self.lat_range = ak.lat_range

        if ak.quadrature == "gauss-legendre":
            if model.latitude_distribution.param_dict:
                gl_nodes, gl_weights = _gauss_legendre_grid(
                    self.n_lat, -np.pi / 2, np.pi / 2)
                _norm = float(jnp.sum(gl_weights))
                self._quad_nodes = gl_nodes
                self._quad_weights = gl_weights / _norm
            else:
                raw_w = ak._quad_weights
                _norm = float(jnp.sum(raw_w))
                self._quad_nodes = ak._quad_nodes
                self._quad_weights = raw_w / _norm   # sum = 1.0
        else:  # trapezoid: None → XLA constant-folds the linspace branch
            self._quad_nodes = None
            self._quad_weights = None

        self._r_gamma_func = model.get_r_gamma_func()
        self._lat_weight_func = model.get_lat_weight_func()
        self._cn_sq_func = model.get_cn_sq_func(self.n_harmonics)

        if isinstance(model.visibility, EdgeOnVisibilityFunction):
            self._edgeon_cn_sq = jnp.array(
                model.visibility.cn_squared(0.0, self.n_harmonics))
        else:
            self._edgeon_cn_sq = None

    # ── Parameter layout ────────────────────────────────────────────────

    @property
    def base_keys(self):
        return tuple(self.spot_model.param_keys)

    @property
    def theta0(self):
        return np.asarray(self.spot_model.theta0, dtype=np.float64)

    # ── Kernel evaluation ───────────────────────────────────────────────

    def k_of_lag(self, theta_slice, lag_flat):
        from .analytic_kernel import _kernel_eval
        return _kernel_eval(
            theta_slice, lag_flat,
            self.n_harmonics, self.n_lat, self.lat_range,
            quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
            r_gamma_func=self._r_gamma_func,
            edgeon_cn_sq=self._edgeon_cn_sq,
            lat_weight_func=self._lat_weight_func,
            cn_sq_func=self._cn_sq_func)

    def bandwidth_support(self, param_keys, bounds_arr):
        return self.spot_model.bandwidth_support(param_keys, bounds_arr)

    def psd(self, omega, theta_slice=None):
        """
        Analytic PSD of this spot term.

        Note: delegates to ``AnalyticKernel.compute_psd``, which uses the
        model's *current* (initial) parameters; ``theta_slice`` is
        ignored for now.
        """
        return self.analytic_kernel.compute_psd(omega)


class KernelSum(Term):
    """
    Additive composition of stationary terms.

    Slices the flat parameter vector per term and sums the per-term
    ``k_of_lag`` contributions — the summation *is* the composition.
    With a single term this is a transparent wrapper (identical keys,
    theta0, bounds, and kernel values).

    When several terms are combined, any term without an explicit
    ``prefix`` is assigned one automatically (``spot0``, ``spot1``,
    ``sho0``, ...) so that parameter keys stay unique.

    Parameters
    ----------
    *terms : Term
        The components to sum.  At least one is required.
    """

    def __init__(self, *terms):
        if not terms:
            raise ValueError("KernelSum requires at least one term")
        for t in terms:
            if not isinstance(t, Term):
                raise TypeError(
                    f"KernelSum components must be Term instances, "
                    f"got {type(t).__name__}")
        self.terms = tuple(terms)
        self._assign_prefixes()

        # Static per-term slice offsets into the flat theta vector.
        self._slices = []
        i = 0
        for t in self.terms:
            n = t.n_params
            self._slices.append((i, n))
            i += n

        dupes = [k for k in self.param_keys
                 if list(self.param_keys).count(k) > 1]
        if dupes:
            raise ValueError(
                f"Duplicate parameter keys across terms: {sorted(set(dupes))}. "
                "Give each term a unique prefix=.")

    def _assign_prefixes(self):
        """Auto-prefix unprefixed terms when more than one is present."""
        if len(self.terms) == 1:
            return  # single term keeps bare keys (backward compat)
        counters = {}
        for t in self.terms:
            tag = t._prefix_tag
            n = counters.get(tag, 0)
            counters[tag] = n + 1
            if t.prefix is None:
                t.prefix = f"{tag}{n}"

    # ── Parameter layout ────────────────────────────────────────────────

    @property
    def base_keys(self):
        # For a KernelSum the solver-facing keys ARE the base keys;
        # per-term prefixes are already baked in.
        return self.param_keys

    @property
    def param_keys(self):
        out = ()
        for t in self.terms:
            out = out + tuple(t.param_keys)
        return out

    @property
    def theta0(self):
        return np.concatenate([np.asarray(t.theta0, dtype=np.float64)
                               for t in self.terms])

    @property
    def default_bounds(self):
        """Default bounds keyed by the solver-facing (prefixed) keys."""
        out = {}
        for t in self.terms:
            tb = t.default_bounds
            for full, bare in zip(t.param_keys, t.base_keys):
                if bare in tb:
                    out[full] = tb[bare]
        return out

    @property
    def stationary(self):
        return all(t.stationary for t in self.terms)

    # ── Kernel evaluation ───────────────────────────────────────────────

    def k_of_lag(self, theta_slice, lag_flat):
        out = 0.0
        for t, (i, n) in zip(self.terms, self._slices):
            out = out + t.k_of_lag(theta_slice[i:i + n], lag_flat)
        return out

    def bandwidth_support(self, param_keys, bounds_arr):
        """Widest term wins: max of per-term supports [days]."""
        supports = []
        for t in self.terms:
            keys, rows = t._own_bounds_rows(param_keys, bounds_arr)
            supports.append(float(t.bandwidth_support(keys, rows)))
        return max(supports)

    def psd(self, omega, theta_slice=None):
        """Sum of the component PSDs on the shared omega grid."""
        freq, total = None, 0.0
        for t, (i, n) in zip(self.terms, self._slices):
            ts = None if theta_slice is None else theta_slice[i:i + n]
            freq, power = t.psd(omega, ts)
            total = total + power
        return freq, total
