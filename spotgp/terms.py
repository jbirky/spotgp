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

__all__ = ["Term", "KernelSum", "SpotTerm", "SharedVisibilitySpotSum",
           "SHOTerm", "Matern32Term", "JitterTerm", "DEFAULT_TERM_BOUNDS"]

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


def _bound_from_rows(keys, rows, name, which, fallback):
    """Look up a bound for ``name`` in a (keys, rows) pair.

    Accepts the physical key or its ``log_``-prefixed variant (bounds
    for log keys are in log10 units and are exponentiated).  ``which``
    is 0 for the lower bound, 1 for the upper.
    """
    keys = list(keys)
    rows = np.asarray(rows)
    if name in keys:
        return float(rows[keys.index(name), which])
    log_name = f"log_{name}"
    if log_name in keys:
        return 10.0 ** float(rows[keys.index(log_name), which])
    return float(fallback)


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
            # The log_ marker sits on the last component, which may be
            # nested ("pop0.log_lspot" -> membership under "pop0.lspot").
            _pre, _sep, _name = bare.rpartition(".")
            if _pre + _sep + _strip_log(_name) in own:
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


class SharedVisibilitySpotSum(Term):
    """
    N spot populations sharing one star's geometry — the composite
    fast path:

        K(tau) = V(tau) * sum_i sigma_k_i^2 * R_Gamma_i(tau)

    The latitude quadrature behind ``V(tau)`` — the expensive part of
    the spot kernel — is evaluated once per kernel call instead of once
    per population; each component contributes only its envelope
    autocorrelation and amplitude.  Numerically this matches
    ``KernelSum(SpotTerm(m0), SpotTerm(m1), ...)`` with equal geometry
    values to machine precision, but the geometry (peq, kappa, inc,
    latitude band) appears *once* in the flat vector — which is usually
    what "several spot populations on the same star" should mean in a
    fit.  Use separate ``SpotTerm``s instead when populations may have
    independent geometry (e.g. different latitude bands).

    Parameter layout: ``[<vis keys>, <lat keys>, pop0.<env keys>,
    pop0.sigma_k, pop1.<env keys>, pop1.sigma_k, ...]``.

    Parameters
    ----------
    models : sequence of SpotEvolutionModel
        One per population (>= 2).  All must share identical visibility
        (class and parameters) and latitude-distribution configuration;
        each carries its own envelope and sigma_k.
    labels : sequence of str, optional
        Component names namespacing the per-population keys
        (default ``pop0``, ``pop1``, ...).
    prefix : str or None
        Optional namespace for the whole term.
    **kernel_kwargs
        Shared kernel configuration forwarded to :class:`AnalyticKernel`
        (n_harmonics, n_lat, lat_range, quadrature).
    """

    _prefix_tag = "spots"

    def __init__(self, models, labels=None, prefix=None, **kernel_kwargs):
        models = list(models)
        if len(models) < 2:
            raise ValueError(
                "SharedVisibilitySpotSum requires at least 2 component "
                "models; use SpotTerm for a single population")
        ref = models[0]
        if ref.visibility is None or ref.envelope is None:
            raise ValueError(
                "component models must have a visibility function and "
                "an envelope")
        for i, m in enumerate(models[1:], 1):
            if (m.visibility is None
                    or type(m.visibility) is not type(ref.visibility)
                    or m.visibility.param_dict != ref.visibility.param_dict):
                raise ValueError(
                    f"component {i} visibility differs from component 0; "
                    "all populations must share one star's geometry "
                    "(use separate SpotTerms for independent geometry)")
            if (type(m.latitude_distribution)
                    is not type(ref.latitude_distribution)
                    or m.latitude_distribution.param_dict
                    != ref.latitude_distribution.param_dict):
                raise ValueError(
                    f"component {i} latitude distribution differs from "
                    "component 0")
            if m.envelope is None:
                raise ValueError(f"component {i} has no envelope")
        self.components = models
        self.labels = (list(labels) if labels is not None
                       else [f"pop{i}" for i in range(len(models))])
        if len(self.labels) != len(models):
            raise ValueError(
                f"expected {len(models)} labels, got {len(self.labels)}")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("component labels must be unique")
        self.prefix = prefix

        # Shared quadrature config and closures, borrowed from a
        # SpotTerm over component 0 (its per-model theta closures are
        # not used — only the static configuration).
        self._ref_term = SpotTerm(ref, **kernel_kwargs)
        self.n_harmonics = self._ref_term.n_harmonics
        self.n_lat = self._ref_term.n_lat
        self.lat_range = self._ref_term.lat_range
        self._quad_nodes = self._ref_term._quad_nodes
        self._quad_weights = self._ref_term._quad_weights
        self._edgeon_cn_sq = self._ref_term._edgeon_cn_sq
        self._cn_sq_func = ref.get_cn_sq_func(self.n_harmonics)

        # Static offsets into this term's theta slice.
        self._n_vis = len(ref.visibility.param_keys)
        self._n_lat_params = len(ref.latitude_distribution.param_keys)
        offset = self._n_vis + self._n_lat_params
        self._comp_slices = []   # (env_start, n_env, sigma_k_idx)
        for m in models:
            n_env = len(m.envelope.param_dict)
            self._comp_slices.append((offset, n_env, offset + n_env))
            offset += n_env + 1

        self._lat_weight_func = self._build_lat_weight_func(ref)

    def _build_lat_weight_func(self, ref):
        """Latitude weights at the composite offset (lat follows vis)."""
        from .latitude import UniformDoubleHemisphereBand

        lat_dist = ref.latitude_distribution
        if not lat_dist.param_dict:
            return None
        lat_offset = self._n_vis
        if isinstance(lat_dist, UniformDoubleHemisphereBand):
            def lat_weight_fn(theta_arr, phi_grid):
                lat_min = theta_arr[lat_offset]
                lat_max = theta_arr[lat_offset + 1]
                abs_phi = jnp.abs(phi_grid)
                return jnp.where((abs_phi > lat_min) & (abs_phi < lat_max),
                                 1.0, 0.0)
            return lat_weight_fn
        import warnings
        warnings.warn(
            f"{type(lat_dist).__name__} has free parameters but no "
            "JAX-traceable weight function; latitude parameters will "
            "have zero gradient.", stacklevel=3)
        return None

    # ── Parameter layout ────────────────────────────────────────────────

    @property
    def base_keys(self):
        ref = self.components[0]
        keys = (tuple(ref.visibility.param_keys)
                + tuple(ref.latitude_distribution.param_keys))
        for m, lab in zip(self.components, self.labels):
            keys += tuple(f"{lab}.{k}" for k in m.envelope.param_dict)
            keys += (f"{lab}.sigma_k",)
        return keys

    @property
    def theta0(self):
        ref = self.components[0]
        vis_d = ref.visibility.param_dict
        lat_d = ref.latitude_distribution.param_dict
        vals = [float(vis_d[k]) for k in ref.visibility.param_keys]
        vals += [float(lat_d[k])
                 for k in ref.latitude_distribution.param_keys]
        for m in self.components:
            vals += [float(v) for v in m.envelope.param_dict.values()]
            vals.append(float(m.sigma_k))
        return np.array(vals, dtype=np.float64)

    @property
    def default_bounds(self):
        out = {}
        for k in self.base_keys:
            bare = k.rpartition(".")[2]
            if bare in DEFAULT_TERM_BOUNDS:
                out[k] = DEFAULT_TERM_BOUNDS[bare]
        return out

    # ── Kernel evaluation ───────────────────────────────────────────────

    def k_of_lag(self, theta_slice, lag_flat):
        from .analytic_kernel import _kernel_eval

        envs = [m.envelope for m in self.components]
        slices = self._comp_slices

        def composite_r_gamma(theta_arr, lag):
            # sigma_k^2-weighted sum of the component envelopes; the
            # trailing 1.0 appended below neutralizes _kernel_eval's own
            # sigma_k^2 factor, so V(tau) multiplies this sum directly.
            total = 0.0
            for env, (start, n_env, sk) in zip(envs, slices):
                R = env.r_gamma_jax(theta_arr[start:start + n_env], lag)
                total = total + theta_arr[sk] ** 2 * R
            return total

        theta_eval = jnp.append(jnp.asarray(theta_slice), 1.0)
        return _kernel_eval(
            theta_eval, lag_flat,
            self.n_harmonics, self.n_lat, self.lat_range,
            quad_nodes=self._quad_nodes, quad_weights=self._quad_weights,
            r_gamma_func=composite_r_gamma,
            edgeon_cn_sq=self._edgeon_cn_sq,
            lat_weight_func=self._lat_weight_func,
            cn_sq_func=self._cn_sq_func)

    def bandwidth_support(self, param_keys, bounds_arr):
        """Max envelope support across components [days]."""
        keys = list(param_keys)
        rows = np.asarray(bounds_arr)
        supports = []
        for m, lab in zip(self.components, self.labels):
            sub_keys, sub_rows = [], []
            for i, k in enumerate(keys):
                pre, _sep, name = k.rpartition(".")
                if pre == lab:
                    sub_keys.append(name)
                    sub_rows.append(rows[i])
            supports.append(float(m.bandwidth_support(
                sub_keys,
                np.asarray(sub_rows) if sub_rows
                else np.zeros((0, 2), dtype=float))))
        return max(supports)

    def psd(self, omega, theta_slice=None):
        """Sum of per-component PSDs (current model parameters)."""
        from .analytic_kernel import AnalyticKernel

        freq, total = None, 0.0
        for m in self.components:
            freq, power = AnalyticKernel(m).compute_psd(omega)
            total = total + power
        return freq, total


class SHOTerm(Term):
    """
    Stochastically-driven damped harmonic oscillator (celerite;
    Foreman-Mackey et al. 2017), useful as a granulation / quasi-periodic
    noise floor alongside spot terms.

    PSD (with the celerite normalization):

        S(omega) = sqrt(2/pi) * S0 * w0^4 /
                   ((omega^2 - w0^2)^2 + w0^2 omega^2 / Q^2)

    The closed-form autocovariance covers all three damping regimes
    (under-, critically-, and over-damped), selected with ``jnp.where``
    and guarded square roots so gradients stay finite near Q = 1/2.
    ``k(0) = S0 * w0 * Q``.

    Parameters
    ----------
    S0 : float
        Power normalization.
    Q : float
        Quality factor (Q = 1/sqrt(2) gives the standard granulation
        background shape).
    w0 : float
        Undamped angular frequency [rad/day].
    prefix : str or None
        Optional namespace for parameter keys (e.g. ``"gran"``).
    """

    _prefix_tag = "sho"

    #: half-width of the |4Q^2 - 1| window that falls back to the
    #: critically-damped closed form (its exact limit).
    _CRIT_TOL = 1e-6

    DEFAULT_BOUNDS = {
        "S0": (1e-12, 1.0),
        "Q":  (0.01, 100.0),
        "w0": (0.01, 100.0),
    }

    def __init__(self, S0=1e-4, Q=1.0 / np.sqrt(2.0), w0=2.0 * np.pi,
                 prefix=None):
        self.S0 = float(S0)
        self.Q = float(Q)
        self.w0 = float(w0)
        self.prefix = prefix

    @property
    def base_keys(self):
        return ("S0", "Q", "w0")

    @property
    def theta0(self):
        return np.array([self.S0, self.Q, self.w0], dtype=np.float64)

    @property
    def default_bounds(self):
        return dict(self.DEFAULT_BOUNDS)

    def k_of_lag(self, theta_slice, lag_flat):
        S0, Q, w0 = theta_slice[0], theta_slice[1], theta_slice[2]
        t = jnp.abs(lag_flat)
        diff = 4.0 * Q ** 2 - 1.0   # > 0 under-damped, < 0 over-damped

        # Guarded eta keeps sqrt/1/eta finite in the branch that is not
        # selected; jnp.where then picks the valid expression.
        eta_u = jnp.sqrt(jnp.maximum(diff, self._CRIT_TOL)) / (2.0 * Q)
        arg_u = eta_u * w0 * t
        k_under = (S0 * w0 * Q * jnp.exp(-w0 * t / (2.0 * Q))
                   * (jnp.cos(arg_u) + jnp.sin(arg_u) / (2.0 * eta_u * Q)))

        # Over-damped: written as a sum of two decaying exponentials so
        # cosh/sinh never overflow (eta_o < 1/(2Q) makes both exponents
        # non-positive).
        eta_o = jnp.sqrt(jnp.maximum(-diff, self._CRIT_TOL)) / (2.0 * Q)
        c = 1.0 / (2.0 * eta_o * Q)
        k_over = (S0 * w0 * Q / 2.0
                  * ((1.0 + c) * jnp.exp((eta_o - 1.0 / (2.0 * Q)) * w0 * t)
                     + (1.0 - c) * jnp.exp(-(eta_o + 1.0 / (2.0 * Q))
                                           * w0 * t)))

        # Critically damped: the exact eta -> 0 limit of both branches,
        # k = S0 w0 Q e^{-w0 t/2Q} (1 + w0 t) at Q = 1/2, keeping
        # k(0) = S0 w0 Q continuous across the regimes.
        k_crit = 0.5 * S0 * w0 * jnp.exp(-w0 * t) * (1.0 + w0 * t)

        return jnp.where(
            jnp.abs(diff) < self._CRIT_TOL, k_crit,
            jnp.where(diff > 0.0, k_under, k_over))

    def bandwidth_support(self, param_keys, bounds_arr):
        """~5 e-folds of the slowest covariance decay within the prior.

        The decay rate is ``w0 * (1/(2Q) - sqrt(max(1/(4Q^2) - 1, 0)))``
        — ``w0/(2Q)`` when under-damped, slower when over-damped — so
        the longest timescale sits at the lowest w0 with Q at either
        bound.
        """
        q_lo = _bound_from_rows(param_keys, bounds_arr, "Q", 0,
                                self.DEFAULT_BOUNDS["Q"][0])
        q_hi = _bound_from_rows(param_keys, bounds_arr, "Q", 1,
                                self.DEFAULT_BOUNDS["Q"][1])
        w_lo = _bound_from_rows(param_keys, bounds_arr, "w0", 0,
                                self.DEFAULT_BOUNDS["w0"][0])

        def decay_time(q):
            rate = (1.0 / (2.0 * q)
                    - np.sqrt(max(1.0 / (4.0 * q ** 2) - 1.0, 0.0))) * w_lo
            return 1.0 / rate

        return 5.0 * max(decay_time(q_lo), decay_time(q_hi))

    def psd(self, omega, theta_slice=None):
        """Analytic PSD; returns ``(freq [cycles/day], power)``."""
        omega = jnp.asarray(omega, dtype=float)
        if theta_slice is None:
            S0, Q, w0 = self.S0, self.Q, self.w0
        else:
            S0, Q, w0 = theta_slice[0], theta_slice[1], theta_slice[2]
        power = (np.sqrt(2.0 / np.pi) * S0 * w0 ** 4
                 / ((omega ** 2 - w0 ** 2) ** 2
                    + (w0 * omega / Q) ** 2))
        return np.asarray(omega / (2 * np.pi)), np.asarray(power)


class Matern32Term(Term):
    """
    Matern-3/2 kernel: ``k(tau) = sigma^2 (1 + sqrt(3) tau / rho)
    exp(-sqrt(3) tau / rho)``.

    Parameters
    ----------
    sigma : float
        Amplitude (standard deviation).
    rho : float
        Length scale [days].
    prefix : str or None
        Optional namespace for parameter keys.
    """

    _prefix_tag = "m32"

    DEFAULT_BOUNDS = {
        "sigma": (1e-6, 1.0),
        "rho":   (0.05, 50.0),
    }

    def __init__(self, sigma=1e-2, rho=1.0, prefix=None):
        self.sigma = float(sigma)
        self.rho = float(rho)
        self.prefix = prefix

    @property
    def base_keys(self):
        return ("sigma", "rho")

    @property
    def theta0(self):
        return np.array([self.sigma, self.rho], dtype=np.float64)

    @property
    def default_bounds(self):
        return dict(self.DEFAULT_BOUNDS)

    def k_of_lag(self, theta_slice, lag_flat):
        sigma, rho = theta_slice[0], theta_slice[1]
        arg = np.sqrt(3.0) * jnp.abs(lag_flat) / rho
        return sigma ** 2 * (1.0 + arg) * jnp.exp(-arg)

    def bandwidth_support(self, param_keys, bounds_arr):
        """~5 e-folds of the exponential decay at the prior's upper rho."""
        rho_hi = _bound_from_rows(param_keys, bounds_arr, "rho", 1,
                                  self.DEFAULT_BOUNDS["rho"][1])
        return 5.0 * rho_hi / np.sqrt(3.0)

    def psd(self, omega, theta_slice=None):
        """Analytic PSD (``lambda = sqrt(3)/rho``); ``(freq, power)``."""
        omega = jnp.asarray(omega, dtype=float)
        if theta_slice is None:
            sigma, rho = self.sigma, self.rho
        else:
            sigma, rho = theta_slice[0], theta_slice[1]
        lam = np.sqrt(3.0) / rho
        power = 4.0 * sigma ** 2 * lam ** 3 / (lam ** 2 + omega ** 2) ** 2
        return np.asarray(omega / (2 * np.pi)), np.asarray(power)


class JitterTerm(Term):
    """
    White-noise sugar: ``k(tau) = sigma_j^2 * [tau == 0]``.

    Pure white noise is normally handled by the solver's ``sigma_n``
    diagonal (set ``fit_sigma_n=True``); this term exists for symmetry
    when composing kernels, e.g. an extra jitter tied to one component.
    Prefer ``sigma_n`` when you just need a noise floor.
    """

    _prefix_tag = "jit"

    DEFAULT_BOUNDS = {
        "sigma_j": (1e-6, 0.1),
    }

    def __init__(self, sigma_j=1e-3, prefix=None):
        self.sigma_j = float(sigma_j)
        self.prefix = prefix

    @property
    def base_keys(self):
        return ("sigma_j",)

    @property
    def theta0(self):
        return np.array([self.sigma_j], dtype=np.float64)

    @property
    def default_bounds(self):
        return dict(self.DEFAULT_BOUNDS)

    def k_of_lag(self, theta_slice, lag_flat):
        # Lags are computed as |x_i - x_j|, so the diagonal is exactly
        # 0.0 and float equality is safe.
        return jnp.where(jnp.asarray(lag_flat) == 0.0,
                         theta_slice[0] ** 2, 0.0)

    def bandwidth_support(self, param_keys, bounds_arr):
        return 0.0

    def psd(self, omega, theta_slice=None):
        """Flat PSD of discrete white noise; ``(freq, power)``."""
        omega = jnp.asarray(omega, dtype=float)
        sj = self.sigma_j if theta_slice is None else theta_slice[0]
        power = np.full(omega.shape, float(sj) ** 2)
        return np.asarray(omega / (2 * np.pi)), power


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
