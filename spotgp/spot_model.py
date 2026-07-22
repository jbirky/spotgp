"""
spot_model.py — SpotEvolutionModel.

SpotEvolutionModel combines an EnvelopeFunction and a VisibilityFunction
with an amplitude parameter (sigma_k) to fully describe the statistical
spot evolution model used by AnalyticKernel, NumericalKernel, GPSolver,
and LightcurveModel.

LatitudeDistributionFunction, VisibilityFunction, EdgeOnVisibilityFunction,
and low-level helpers are re-exported here for backward compatibility.
"""
from __future__ import annotations

import numpy as np

_UNSET = object()  # sentinel to distinguish "not passed" from explicit None
import jax
import jax.numpy as jnp

from .distributions import as_distribution, is_distributed, DeltaDistribution

try:
    from .envelope import (
        EnvelopeFunction,
        TrapezoidSymmetricEnvelope,
        TrapezoidAsymmetricEnvelope,
        SkewedGaussianEnvelope,
        ExponentialEnvelope,
    )
    from .params import resolve_hparam
    from .latitude import LatitudeDistributionFunction, UniformDoubleHemisphereBand
    from .visibility import (
        VisibilityFunction,
        EdgeOnVisibilityFunction,
        FullGeometryVisibilityFunction,
        LimbDarkenedVisibilityFunction,
        _cn_general_jax,
        _cn_squared_coefficients_jax,
        _gauss_legendre_grid,
    )
except ImportError:
    from envelope import (
        EnvelopeFunction,
        TrapezoidSymmetricEnvelope,
        TrapezoidAsymmetricEnvelope,
        SkewedGaussianEnvelope,
        ExponentialEnvelope,
    )
    from params import resolve_hparam
    from latitude import LatitudeDistributionFunction, UniformDoubleHemisphereBand
    from visibility import (
        VisibilityFunction,
        EdgeOnVisibilityFunction,
        FullGeometryVisibilityFunction,
        LimbDarkenedVisibilityFunction,
        _cn_general_jax,
        _cn_squared_coefficients_jax,
        _gauss_legendre_grid,
    )

__all__ = [
    # Re-exported for backward compatibility
    "LatitudeDistributionFunction",
    "VisibilityFunction",
    "EdgeOnVisibilityFunction",
    "FullGeometryVisibilityFunction",
    "LimbDarkenedVisibilityFunction",
    "_cn_general_jax",
    "_cn_squared_coefficients_jax",
    "_gauss_legendre_grid",
    # Defined here
    "SpotEvolutionModel",
]


# ── SpotEvolutionModel ──────────────────────────────────────────────────────

class SpotEvolutionModel:
    """
    Complete statistical spot evolution model.

    Combines an EnvelopeFunction (spot size evolution) with a
    VisibilityFunction (stellar rotation and inclination) and a kernel
    amplitude parameter sigma_k.

    Parameters
    ----------
    envelope : EnvelopeFunction or None
        Spot size envelope (e.g. TrapezoidSymmetricEnvelope).  When None
        the kernel contains only the visibility function (R_Gamma = 1).
    visibility : VisibilityFunction or None
        Stellar visibility function (peq, kappa, inc).  When None the
        kernel contains only the envelope function.
    sigma_k : float, optional
        Kernel amplitude prefactor. Provide either sigma_k directly or
        (nspot_rate, fspot, alpha_max) for the physical parameterization.
    nspot_rate : float, optional
        Spot emergence rate [spots/day].  Used when sigma_k is not given.
    fspot : float, optional
        Spot contrast fraction (default 0).
    alpha_max : float, optional
        Peak spot angular radius [rad].  Used when sigma_k is not given.
    latitude_distribution : LatitudeDistributionFunction, optional
        Latitude probability density for spot placement and kernel
        integration.  Defaults to a uniform distribution over
        [-pi/2, pi/2].

    Notes
    -----
    Exactly one of ``sigma_k`` or the triplet ``(nspot_rate, fspot, alpha_max)``
    must be supplied.
    """

    def __init__(
        self,
        envelope: EnvelopeFunction = _UNSET,
        visibility: VisibilityFunction = _UNSET,
        sigma_k: float = None,
        nspot_rate: float = None,
        fspot: float = 0.0,
        alpha_max: float = None,
        latitude_distribution: LatitudeDistributionFunction = _UNSET,
    ):
        # Resolve each component and record its provenance for get_sympy()
        if envelope is _UNSET:
            self.envelope = None
            self._envelope_status = "not specified"
        elif envelope is None:
            self.envelope = None
            self._envelope_status = "not specified"
        else:
            if not isinstance(envelope, EnvelopeFunction):
                raise TypeError(
                    f"envelope must be an EnvelopeFunction, got {type(envelope)}")
            self.envelope = envelope
            self._envelope_status = "user defined"

        if visibility is _UNSET:
            self.visibility = None
            self._visibility_status = "not specified"
        elif visibility is None:
            self.visibility = None
            self._visibility_status = "not specified"
        else:
            if not isinstance(visibility, VisibilityFunction):
                raise TypeError(
                    f"visibility must be a VisibilityFunction, got {type(visibility)}")
            self.visibility = visibility
            self._visibility_status = "user defined"

        if latitude_distribution is _UNSET:
            self.latitude_distribution = LatitudeDistributionFunction()
            self._latitude_status = "default"
        elif latitude_distribution is None:
            self.latitude_distribution = LatitudeDistributionFunction()
            self._latitude_status = "not specified"
        else:
            if not isinstance(latitude_distribution, LatitudeDistributionFunction):
                raise TypeError(
                    f"latitude_distribution must be a LatitudeDistributionFunction, "
                    f"got {type(latitude_distribution)}")
            self.latitude_distribution = latitude_distribution
            self._latitude_status = "user defined"
        self.fspot = float(fspot)
        self.alpha_max = float(alpha_max) if alpha_max is not None else None

        if sigma_k is not None:
            self._sigma_k_dist = as_distribution(sigma_k)
        elif nspot_rate is not None and alpha_max is not None:
            computed = (
                float(np.sqrt(float(nspot_rate)))
                * (1.0 - float(fspot))
                * float(alpha_max) ** 2
            )
            self._sigma_k_dist = as_distribution(computed)
        else:
            raise ValueError(
                "SpotEvolutionModel requires either sigma_k or "
                "(nspot_rate, fspot, alpha_max).")

    # ── Convenience accessors ───────────────────────────────────────────────

    @property
    def sigma_k(self) -> float:
        """Point estimate (mean) of sigma_k. Backward-compatible float."""
        return self._sigma_k_dist.mean

    @sigma_k.setter
    def sigma_k(self, value):
        self._sigma_k_dist = as_distribution(value)

    @property
    def sigma_k_distribution(self):
        """The full ParameterDistribution for sigma_k."""
        return self._sigma_k_dist

    @property
    def sigma_k_sq_expected(self) -> float:
        """E[sigma_k^2] under the distribution. Exact for DeltaDistribution."""
        return self._sigma_k_dist.expectation(lambda x: x ** 2)

    @property
    def peq(self) -> float:
        return self.visibility.peq if self.visibility is not None else None

    @property
    def kappa(self) -> float:
        return self.visibility.kappa if self.visibility is not None else None

    @property
    def inc(self) -> float:
        return self.visibility.inc if self.visibility is not None else None

    @property
    def lspot(self) -> float:
        return self.envelope.lspot if self.envelope is not None else None

    @property
    def tau_spot(self) -> float:
        return self.envelope.tau_spot if self.envelope is not None else None

    # ── Parameter keys ──────────────────────────────────────────────────────

    @property
    def param_keys(self) -> tuple:
        """
        Ordered parameter names for the theta vector used in GPSolver.

        When both are present, starts with (peq, kappa, inc) from the
        visibility function, followed by the envelope-specific keys, then
        sigma_k.  When envelope is None only the visibility keys are
        included; when visibility is None only the envelope keys are
        included.
        """
        vis_keys = self.visibility.param_keys if self.visibility is not None else ()
        env_keys = (tuple(self.envelope.param_dict.keys())
                    if self.envelope is not None else ())
        lat_keys = self.latitude_distribution.param_keys
        return vis_keys + env_keys + lat_keys + ("sigma_k",)

    @property
    def theta0(self) -> np.ndarray:
        """Initial parameter vector from current model values."""
        vals = {}
        if self.visibility is not None:
            vals.update(self.visibility.param_dict)
        if self.envelope is not None:
            vals.update(self.envelope.param_dict)
        vals.update(self.latitude_distribution.param_dict)
        vals["sigma_k"] = self.sigma_k
        return np.array([float(vals[k]) for k in self.param_keys])

    def theta_from_hparam(self, hparam: dict) -> np.ndarray:
        """
        Build a theta vector from a (possibly partial) hparam dict.
        Missing keys fall back to the model's current values.
        """
        current = dict(zip(self.param_keys, self.theta0))
        current.update({k: float(v) for k, v in hparam.items()
                        if k in current})
        return np.array([current[k] for k in self.param_keys])

    # ── JAX-compilable R_Gamma function ─────────────────────────────────────

    def get_r_gamma_func(self):
        """
        Return a JAX-traceable function r_gamma(theta_arr, lag) -> R_Gamma.

        The theta_arr layout follows self.param_keys:
          [peq, kappa, inc, <envelope params...>, sigma_k]

        When envelope is None, returns a function that always yields 1.0
        (pure visibility kernel).  Delegates to
        ``EnvelopeFunction.r_gamma_jax`` for the actual computation, so
        user-defined envelope subclasses that override ``r_gamma_jax``
        are automatically supported.
        """
        if self.envelope is None:
            def r_gamma(theta_arr, lag):  # noqa: ARG001
                return jnp.ones_like(jnp.asarray(lag))
            return r_gamma

        n_vis = len(self.visibility.param_keys) if self.visibility is not None else 0
        n_env = len(self.envelope.param_dict)
        envelope = self.envelope

        def r_gamma(theta_arr, lag):
            theta_env = theta_arr[n_vis:n_vis + n_env]
            return envelope.r_gamma_jax(theta_env, lag)

        return r_gamma

    # ── JAX-compilable latitude weight function ────────────────────────────

    def get_lat_weight_func(self):
        """
        Return a JAX-traceable function ``f(theta_arr, phi_grid) -> weights``
        that computes per-node latitude weights from the theta vector.

        When the latitude distribution has no free parameters, returns None
        (the caller should use the static weights precomputed at init).

        The theta_arr layout follows ``self.param_keys``.
        """
        lat_dist = self.latitude_distribution
        if not lat_dist.param_dict:
            return None

        # Index of first latitude param in theta_arr
        n_vis = len(self.visibility.param_keys) if self.visibility is not None else 0
        n_env = len(self.envelope.param_dict) if self.envelope is not None else 0
        lat_offset = n_vis + n_env

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
            f"{type(lat_dist).__name__} has free parameters "
            f"{tuple(lat_dist.param_dict.keys())} but no JAX-traceable "
            f"weight function. Latitude parameters will appear in the "
            f"fit vector but have zero gradient. Implement a "
            f"lat_weight_fn for this distribution or use fixed parameters.",
            stacklevel=2)
        return None

    # ── JAX-compilable visibility coefficient function ─────────────────────

    def get_cn_sq_func(self, n_harmonics: int):
        """
        Return a JAX-traceable ``cn_sq(theta_arr, phi_grid) -> (n_phi, n_h+1)``
        function, or ``None`` to use the built-in analytic ``c_n``.

        The default ``VisibilityFunction`` (and ``EdgeOnVisibilityFunction``)
        rely on the closed-form ``_cn_general_jax`` coefficients that the
        kernel already evaluates inline, so this returns ``None`` for them.
        A visibility subclass that computes its own coefficients (e.g.
        :class:`LimbDarkenedVisibilityFunction`) opts in by defining a
        ``cn_sq_jax(theta_vis, phi, n_harmonics)`` method; this wraps that
        method into the per-latitude-grid form expected by ``_kernel_eval``,
        so the custom coefficients reach the latitude-averaged kernel and
        not only the PSD / single-latitude paths.

        The theta_arr layout follows ``self.param_keys``: the visibility
        parameters are the leading ``len(visibility.param_keys)`` entries.
        """
        vis = self.visibility
        if vis is None or not hasattr(vis, "cn_sq_jax"):
            return None

        n_vis = len(vis.param_keys)

        def cn_sq_func(theta_arr, phi_grid):
            theta_vis = theta_arr[:n_vis]
            return jax.vmap(
                lambda phi: vis.cn_sq_jax(theta_vis, phi, n_harmonics)
            )(phi_grid)

        return cn_sq_func

    # ── Bandwidth support ───────────────────────────────────────────────────

    def bandwidth_support(self, param_keys, bounds_arr) -> float:
        """
        Estimate the kernel support using upper bounds of parameters.

        Used by GPSolver._compute_bandwidth to determine the banded
        Cholesky bandwidth as a compile-time constant.  Delegates to
        ``EnvelopeFunction.support_from_bounds``.

        Parameters
        ----------
        param_keys : sequence of str
            Parameter names in the same order as bounds_arr.
        bounds_arr : array_like, shape (n_params, 2)
            Lower and upper bounds for each parameter.
        """
        if self.envelope is None:
            return 0.0

        keys = list(param_keys)
        bounds_arr = np.asarray(bounds_arr)

        def upper_fn(key, fallback):
            if key in keys:
                return float(bounds_arr[keys.index(key), 1])
            log_key = f"log_{key}"
            if log_key in keys:
                return 10.0 ** float(bounds_arr[keys.index(log_key), 1])
            return float(fallback)

        return self.envelope.support_from_bounds(upper_fn)

    # ── Serialization ───────────────────────────────────────────────────────

    def to_hparam(self) -> dict:
        """
        Convert to a flat hparam dict for backward compatibility.

        The returned dict is accepted by resolve_hparam and by the
        old-style constructors of AnalyticKernel, GPSolver, etc.
        """
        d = {}
        if self.visibility is not None:
            d.update(self.visibility.param_dict)
        if self.envelope is not None:
            d.update(self.envelope.param_dict)
        d["sigma_k"] = self.sigma_k
        if self.alpha_max is not None:
            d["alpha_max"] = self.alpha_max
        if self.fspot:
            d["fspot"] = self.fspot
        return d

    @classmethod
    def from_hparam(cls, hparam: dict) -> "SpotEvolutionModel":
        """
        Construct a SpotEvolutionModel from a raw hparam dict.

        Accepts the same dict format that resolve_hparam accepts,
        including all envelope types and amplitude modes.
        """
        p = resolve_hparam(hparam)

        visibility = VisibilityFunction(p["peq"], p["kappa"], p["inc"])

        if "sigma_sn" in hparam and "n_sn" in hparam:
            envelope = SkewedGaussianEnvelope(
                p["sigma_sn"], p["n_sn"], p.get("lspot", 0.0))
        elif "tau_em" in hparam and "tau_dec" in hparam:
            envelope = TrapezoidAsymmetricEnvelope(
                p["lspot"], p["tau_em"], p["tau_dec"])
        else:
            envelope = TrapezoidSymmetricEnvelope(p["lspot"], p["tau_spot"])

        return cls(envelope, visibility, sigma_k=p["sigma_k"])

    def get_sympy(self, display=True, compute_symbolic=False):
        """
        Display sympy expressions for the full spot evolution model.

        Delegates to ``EnvelopeFunction.get_sympy()`` and
        ``VisibilityFunction.get_sympy()`` in sequence.

        Requires sympy (``pip install sympy``).

        Parameters
        ----------
        display : bool, optional
            If True (default), render equations as formatted LaTeX in a
            Jupyter notebook (via IPython.display) or print them as LaTeX
            strings in a plain terminal.
        compute_symbolic : bool, optional
            Passed through to ``EnvelopeFunction.get_sympy()``.  If True,
            attempt to derive Gamma_hat and R_Gamma symbolically from
            sympy_Gamma() when no explicit override exists.  Defaults to
            False.

        Returns
        -------
        dict
            ``{"envelope": envelope_exprs, "visibility": visibility_exprs,
               "latitude": latitude_exprs}``
            where each value is the dict returned by the respective
            ``get_sympy()`` call.
        """
        def _display_not_specified(label):
            if display:
                try:
                    from IPython.display import display as ipy_display, Math
                    ipy_display(Math(r"\textbf{" + label
                                     + r"} \text{ [not specified]}"))
                except ImportError:
                    print(f"{label} [not specified]")

        if self.envelope is not None:
            envelope_exprs = self.envelope.get_sympy(
                display=display, status=self._envelope_status,
                compute_symbolic=compute_symbolic)
        else:
            _display_not_specified("EnvelopeFunction")
            envelope_exprs = None

        if self.visibility is not None:
            visibility_exprs = self.visibility.get_sympy(
                display=display, status=self._visibility_status)
        else:
            _display_not_specified("VisibilityFunction")
            visibility_exprs = None

        latitude_exprs = self.latitude_distribution.get_sympy(
            display=display, status=self._latitude_status)
        return {"envelope": envelope_exprs, "visibility": visibility_exprs, "latitude": latitude_exprs}

    def __repr__(self) -> str:
        if self.envelope is not None:
            env_str = (f"{self.envelope.__class__.__name__}"
                       f"({self.envelope.param_dict})")
        else:
            env_str = "None"
        if self.visibility is not None:
            vis_str = (f"VisibilityFunction"
                       f"(peq={self.peq}, kappa={self.kappa}, inc={self.inc:.3f})")
        else:
            vis_str = "None"
        return (
            f"SpotEvolutionModel(\n"
            f"  envelope={env_str},\n"
            f"  visibility={vis_str},\n"
            f"  sigma_k={self.sigma_k}\n)"
        )
