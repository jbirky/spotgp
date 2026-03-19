"""
spot_model.py — VisibilityFunction and SpotEvolutionModel.

VisibilityFunction encapsulates the Fourier-series representation of the
stellar visibility function: how much flux a spot at latitude phi
contributes as the star rotates, decomposed into rotation harmonics.

SpotEvolutionModel combines an EnvelopeFunction and a VisibilityFunction
with an amplitude parameter (sigma_k) to fully describe the statistical
spot evolution model used by AnalyticKernel, NumericalKernel, GPSolver,
and LightcurveModel.
"""
from __future__ import annotations

import numpy as np

_UNSET = object()  # sentinel to distinguish "not passed" from explicit None
import jax
import jax.numpy as jnp

try:
    from .envelope import (
        EnvelopeFunction,
        TrapezoidSymmetricEnvelope,
        TrapezoidAsymmetricEnvelope,
        SkewedGaussianEnvelope,
        ExponentialEnvelope,
        _R_Gamma_symmetric,
        _R_Gamma_asymmetric,
    )
    from .params import resolve_hparam
except ImportError:
    from envelope import (
        EnvelopeFunction,
        TrapezoidSymmetricEnvelope,
        TrapezoidAsymmetricEnvelope,
        SkewedGaussianEnvelope,
        ExponentialEnvelope,
        _R_Gamma_symmetric,
        _R_Gamma_asymmetric,
    )
    from params import resolve_hparam

__all__ = [
    "LatitudeDistributionFunction",
    "VisibilityFunction",
    "EdgeOnVisibilityFunction",
    "SpotEvolutionModel",
    # low-level helpers re-exported for backward compat
    "_cn_general_jax",
    "_cn_squared_coefficients_jax",
    "_gauss_legendre_grid",
]


# ── Low-level JAX helpers (Fourier visibility coefficients) ─────────────────

def _safe_arccos(x):
    """arccos safe for autodiff at x = ±1."""
    return jnp.arccos(jnp.clip(x, -1.0 + 1e-7, 1.0 - 1e-7))


@jax.jit
def _cn_general_jax(n, inc, phi):
    """
    Fourier coefficient c_n of the visibility function.
    JAX-compatible; uses safe_arccos for finite gradients at boundaries.
    """
    a0 = jnp.cos(inc) * jnp.sin(phi)
    a1 = jnp.sin(inc) * jnp.cos(phi)

    safe_a1 = jnp.where(jnp.abs(a1) < 1e-15, 1.0, a1)
    ratio = -a0 / safe_a1

    always_visible = ratio <= -1.0
    never_visible = ratio >= 1.0
    tiny_a1 = jnp.abs(a1) < 1e-15

    theta_vis = jnp.where(
        tiny_a1, 0.0,
        jnp.where(always_visible, jnp.pi,
                  jnp.where(never_visible, 0.0,
                            _safe_arccos(ratio))))

    c0 = jnp.where(tiny_a1, a0,
                   (a0 * theta_vis + a1 * jnp.sin(theta_vis)) / jnp.pi)
    c0 = jnp.where(never_visible & ~tiny_a1, 0.0, c0)

    c1 = (a0 * jnp.sin(theta_vis)
          + a1 / 2 * (theta_vis + jnp.sin(theta_vis) * jnp.cos(theta_vis))) / jnp.pi
    c1 = jnp.where(tiny_a1 | never_visible, 0.0, c1)

    n_f = jnp.float64(n) if hasattr(jnp, 'float64') else jnp.float32(n)
    nm1 = n_f - 1
    np1 = n_f + 1
    safe_nm1 = jnp.where(jnp.abs(nm1) < 1e-15, 1.0, nm1)
    safe_np1 = jnp.where(jnp.abs(np1) < 1e-15, 1.0, np1)

    term1 = a0 * jnp.sin(n_f * theta_vis) / (n_f + 1e-30)
    term2 = a1 / 2 * (jnp.sin(safe_nm1 * theta_vis) / safe_nm1
                       + jnp.sin(safe_np1 * theta_vis) / safe_np1)
    cn_general = (term1 + term2) / jnp.pi
    cn_general = jnp.where(tiny_a1 | never_visible, 0.0, cn_general)

    return jnp.where(n == 0, c0, jnp.where(n == 1, c1, cn_general))


def _cn_squared_coefficients_jax(inc, phi, n_harmonics=2):
    """Compute |c_n|² for n = 0, 1, ..., n_harmonics."""
    ns = jnp.arange(n_harmonics + 1)
    cn_vals = jax.vmap(lambda n: _cn_general_jax(n, inc, phi))(ns)
    return cn_vals ** 2


def _gauss_legendre_grid(n, a, b):
    """
    Gauss-Legendre nodes and weights on [a, b].

    Returns
    -------
    nodes : jnp.ndarray, shape (n,)
    weights : jnp.ndarray, shape (n,)
    """
    nodes_ref, weights_ref = np.polynomial.legendre.leggauss(n)
    nodes = 0.5 * (b - a) * nodes_ref + 0.5 * (a + b)
    weights = 0.5 * (b - a) * weights_ref
    return jnp.array(nodes), jnp.array(weights)


# ── LatitudeDistributionFunction ────────────────────────────────────────────

class LatitudeDistributionFunction:
    """
    Base class for starspot latitude distributions.

    Defines the probability density p(phi) over stellar latitude phi and
    the latitude range over which spots are placed.

    The default implementation is a uniform distribution over
    [-pi/2, pi/2].  To define a custom distribution, subclass this class
    and override ``__call__`` and optionally ``lat_range``.

    Examples
    --------
    Equatorial band (spots confined to |phi| < 30 deg):

    >>> class EquatorialBand(LatitudeDistributionFunction):
    ...     @property
    ...     def lat_range(self):
    ...         return (-np.pi / 6, np.pi / 6)
    ...     def __call__(self, phi):
    ...         return 1.0

    Gaussian centred on the equator:

    >>> class GaussianLatitude(LatitudeDistributionFunction):
    ...     def __init__(self, sigma=np.pi / 6):
    ...         self.sigma = sigma
    ...     def __call__(self, phi):
    ...         return np.exp(-0.5 * (phi / self.sigma) ** 2)
    """

    @property
    def lat_range(self) -> tuple:
        """(min, max) latitude in radians."""
        return (-np.pi / 2, np.pi / 2)

    def __call__(self, phi: float) -> float:
        """
        Unnormalized probability density at latitude phi.

        Normalization is handled internally by the kernel integrator.

        Parameters
        ----------
        phi : float
            Stellar latitude [radians].

        Returns
        -------
        float
            Relative probability density at phi.
        """
        return 1.0

    def sympy_pdf(self):
        """
        Return the sympy expression for the latitude PDF p(phi).

        Subclasses should override this to provide their analytic form.
        The base implementation returns 1 (uniform distribution).

        Returns
        -------
        sympy.Expr or None
            Sympy expression for p(phi), or None if no analytic form exists.
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError(
                "sympy is required for get_sympy(). "
                "Install with: pip install sympy")
        return sp.Integer(1)

    def get_sympy(self, display=True, status=None):
        """
        Display the sympy expression for the latitude PDF p(phi).

        Requires sympy (``pip install sympy``).

        Parameters
        ----------
        display : bool, optional
            If True (default), render equations as formatted LaTeX in a
            Jupyter notebook (via IPython.display) or print them as LaTeX
            strings in a plain terminal.
        status : str or None, optional
            If provided, appended to the class name header in brackets,
            e.g. ``"default"`` renders as
            ``LatitudeDistributionFunction [default]``.

        Returns
        -------
        dict
            ``{"pdf": expr_or_None}``
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError(
                "sympy is required for get_sympy(). "
                "Install with: pip install sympy")

        expr = self.sympy_pdf()
        exprs = {"pdf": expr}

        if display:
            rhs = r"\text{[numerical]}" if expr is None else sp.latex(expr)
            status_tag = r" \text{[" + status + r"]}" if status else ""
            header = r"\textbf{" + type(self).__name__ + r"}" + status_tag
            try:
                from IPython.display import display as ipy_display, Math
                ipy_display(Math(header))
                ipy_display(Math(r"p(\phi) = " + rhs))
            except ImportError:
                status_str = f" [{status}]" if status else ""
                print(f"{type(self).__name__}{status_str}")
                print(f"  $p(\\phi) = {rhs}$")

        return exprs

    def __repr__(self) -> str:
        return (f"{type(self).__name__}("
                f"lat_range=[{self.lat_range[0]:.3f}, {self.lat_range[1]:.3f}])")


# ── VisibilityFunction ──────────────────────────────────────────────────────

class VisibilityFunction:
    """
    Stellar visibility function parameterized by rotation and inclination.

    The visibility function V(phi, lon) describes the flux contribution from
    a spot at latitude phi as the star rotates.  It is expanded in a Fourier
    series over rotation harmonics, with coefficients c_n(inc, phi).

    Parameters
    ----------
    peq : float
        Equatorial rotation period [days].
    kappa : float
        Differential rotation shear (dimensionless).
    inc : float
        Stellar inclination [radians].
    """

    def __init__(self, peq: float, kappa: float, inc: float):
        self.peq = float(peq)
        self.kappa = float(kappa)
        self.inc = float(inc)

    @property
    def param_dict(self) -> dict:
        """Visibility parameters as {name: value}."""
        return {"peq": self.peq, "kappa": self.kappa, "inc": self.inc}

    @property
    def param_keys(self) -> tuple:
        """Ordered parameter names."""
        return ("peq", "kappa", "inc")

    def omega0(self, phi):
        """Latitude-dependent rotation angular frequency [rad/day]."""
        return 2.0 * jnp.pi * (1.0 - self.kappa * jnp.sin(phi) ** 2) / self.peq

    def cn_squared(self, phi, n_harmonics: int = 3):
        """
        Squared Fourier coefficients |c_n|² at stellar latitude phi.

        Returns
        -------
        cn_sq : jnp.ndarray, shape (n_harmonics + 1,)
        """
        return _cn_squared_coefficients_jax(self.inc, phi, n_harmonics)

    def get_sympy(self, display=True, status=None):
        """
        Display the sympy expressions for the visibility function.

        Renders or prints LaTeX for:
          - omega_0(phi): latitude-dependent rotation angular frequency.
          - a_0, a_1, theta_v: intermediate visibility geometry variables.
          - c_0, c_1: special-case Fourier coefficients.
          - c_n: general Fourier coefficient (n >= 2).

        Intermediate symbols are introduced so each equation stays compact
        and human-readable.

        Requires sympy (``pip install sympy``).

        Parameters
        ----------
        display : bool, optional
            If True (default), render equations as formatted LaTeX in a
            Jupyter notebook (via IPython.display) or print them as LaTeX
            strings in a plain terminal.
        status : str or None, optional
            If provided, appended to the class name header in brackets,
            e.g. ``"user defined"`` renders as
            ``VisibilityFunction [user defined]``.

        Returns
        -------
        dict
            ``{"omega0": expr, "a0": expr, "a1": expr,
               "theta_v": expr, "c0": expr, "c1": expr, "cn": expr}``
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError(
                "sympy is required for get_sympy(). "
                "Install with: pip install sympy")

        phi   = sp.Symbol(r'\phi', real=True)
        inc   = sp.Symbol('i', positive=True)
        P_eq  = sp.Symbol(r'P_{\rm eq}', positive=True)
        kappa = sp.Symbol(r'\kappa', real=True)
        n     = sp.Symbol('n', positive=True, integer=True)

        # Intermediate geometry symbols (keeps c_n expressions readable)
        a0_sym  = sp.Symbol('a_0', real=True)
        a1_sym  = sp.Symbol('a_1', real=True)
        tv_sym  = sp.Symbol(r'\theta_v', nonnegative=True)

        # Definitions
        omega0   = 2 * sp.pi * (1 - kappa * sp.sin(phi)**2) / P_eq
        a0_def   = sp.cos(inc) * sp.sin(phi)
        a1_def   = sp.sin(inc) * sp.cos(phi)
        theta_def = sp.acos(-a0_sym / a1_sym)

        # Fourier coefficients in terms of the intermediate symbols
        c0 = (a0_sym * tv_sym + a1_sym * sp.sin(tv_sym)) / sp.pi
        c1 = (a0_sym * sp.sin(tv_sym)
              + a1_sym / 2 * (tv_sym + sp.sin(tv_sym) * sp.cos(tv_sym))) / sp.pi
        cn = (a0_sym * sp.sin(n * tv_sym) / n
              + a1_sym / 2 * (sp.sin((n - 1) * tv_sym) / (n - 1)
                              + sp.sin((n + 1) * tv_sym) / (n + 1))) / sp.pi

        exprs = {
            "omega0": omega0, "a0": a0_def, "a1": a1_def,
            "theta_v": theta_def, "c0": c0, "c1": c1, "cn": cn,
        }

        lhs = {
            "omega0":   r"\omega_0(\phi)",
            "a0":       r"a_0",
            "a1":       r"a_1",
            "theta_v":  r"\theta_v",
            "c0":       r"c_0",
            "c1":       r"c_1",
            "cn":       r"c_n \; (n \geq 2)",
        }

        if display:
            status_tag = r" \text{[" + status + r"]}" if status else ""
            header = r"\textbf{VisibilityFunction}" + status_tag
            try:
                from IPython.display import display as ipy_display, Math
                ipy_display(Math(header))
                for key, expr in exprs.items():
                    ipy_display(Math(lhs[key] + " = " + sp.latex(expr)))
            except ImportError:
                status_str = f" [{status}]" if status else ""
                print(f"VisibilityFunction{status_str}")
                for key, expr in exprs.items():
                    print(f"  ${lhs[key]} = {sp.latex(expr)}$")

        return exprs


class EdgeOnVisibilityFunction(VisibilityFunction):
    """
    Closed-form visibility for edge-on viewing (I = pi/2) with solid-body
    rotation (kappa = 0) and a uniform latitude distribution.

    For this special case, the latitude-averaged squared Fourier coefficients
    are known analytically (Eq. 68 of Birky et al.):

        <|c_0|^2> = 1 / (2 * pi^2)
        <|c_1|^2> = 1 / 16
        <|c_2|^2> = 1 / (9 * pi^2)

    and the rotation frequency is latitude-independent:
        omega_0 = 2 * pi / P_eq.

    This eliminates the need for numerical latitude quadrature, making
    kernel evaluation significantly faster.

    Parameters
    ----------
    peq : float
        Equatorial rotation period [days].
    """

    # Pre-computed latitude-averaged |c_n|^2 = g_n^2 / 2
    # where g_0 = 1/pi, g_1 = 1/4, g_2 = 1/(3*pi)
    _CN_SQ = None  # lazily built as JAX array

    def __init__(self, peq: float):
        super().__init__(peq=peq, kappa=0.0, inc=jnp.pi / 2)

    @property
    def param_dict(self) -> dict:
        return {"peq": self.peq}

    @property
    def param_keys(self) -> tuple:
        return ("peq",)

    def omega0(self, phi):
        """Rotation frequency (latitude-independent for kappa=0)."""
        return 2.0 * jnp.pi / self.peq

    def cn_squared(self, phi, n_harmonics: int = 3):
        """Latitude-averaged |c_n|^2 (independent of phi).

        Returns the closed-form coefficients for n = 0, 1, 2 and zero
        for higher harmonics.
        """
        cn_sq = jnp.zeros(n_harmonics + 1)
        pi2 = jnp.pi ** 2
        # n=0: g_0 = 1/pi  => g_0^2/2 = 1/(2*pi^2)
        cn_sq = cn_sq.at[0].set(1.0 / (2.0 * pi2))
        # n=1: g_1 = 1/4   => g_1^2/2 = 1/32
        if n_harmonics >= 1:
            cn_sq = cn_sq.at[1].set(1.0 / 32.0)
        # n=2: g_2 = 1/(3*pi) => g_2^2/2 = 1/(18*pi^2)
        if n_harmonics >= 2:
            cn_sq = cn_sq.at[2].set(1.0 / (18.0 * pi2))
        return cn_sq


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
            self.sigma_k = float(sigma_k)
        elif nspot_rate is not None and alpha_max is not None:
            self.sigma_k = (
                float(np.sqrt(float(nspot_rate)))
                * (1.0 - float(fspot))
                * float(alpha_max) ** 2
            )
        else:
            raise ValueError(
                "SpotEvolutionModel requires either sigma_k or "
                "(nspot_rate, fspot, alpha_max).")

    # ── Convenience accessors ───────────────────────────────────────────────

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
        return vis_keys + env_keys + ("sigma_k",)

    @property
    def theta0(self) -> np.ndarray:
        """Initial parameter vector from current model values."""
        vals = {}
        if self.visibility is not None:
            vals.update(self.visibility.param_dict)
        if self.envelope is not None:
            vals.update(self.envelope.param_dict)
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
        (pure visibility kernel).  The R_Gamma function is selected based
        on the envelope type and captured (together with any precomputed
        grids) in a closure so that the returned callable is safe to use
        inside jax.jit.
        """
        if self.envelope is None:
            def r_gamma(theta_arr, lag):  # noqa: ARG001
                return jnp.ones_like(jnp.asarray(lag))
            return r_gamma

        n_vis = len(self.visibility.param_keys) if self.visibility is not None else 0

        if isinstance(self.envelope, TrapezoidSymmetricEnvelope):
            def r_gamma(theta_arr, lag):
                lspot     = theta_arr[n_vis]        # index 3
                tau_spot  = theta_arr[n_vis + 1]    # index 4
                return _R_Gamma_symmetric(lag, lspot, tau_spot)

        elif isinstance(self.envelope, TrapezoidAsymmetricEnvelope):
            def r_gamma(theta_arr, lag):
                lspot   = theta_arr[n_vis]       # index 3
                tau_em  = theta_arr[n_vis + 1]   # index 4
                tau_dec = theta_arr[n_vis + 2]   # index 5
                te = jnp.minimum(tau_em, tau_dec)
                td = jnp.maximum(tau_em, tau_dec)
                return _R_Gamma_asymmetric(lag, lspot, te, td)

        elif isinstance(self.envelope, SkewedGaussianEnvelope):
            # sigma_sn and n_sn are in theta but R_Gamma uses the
            # precomputed interpolation grid (fixed at init time).
            lag_grid = self.envelope._R_lag_grid
            R_vals   = self.envelope._R_vals

            def r_gamma(theta_arr, lag):  # noqa: ARG001 (theta_arr unused)
                return jnp.interp(jnp.abs(lag), lag_grid, R_vals)

        elif isinstance(self.envelope, ExponentialEnvelope):
            def r_gamma(theta_arr, lag):
                tau_spot = theta_arr[n_vis]  # index 3 (no lspot for exponential)
                abs_lag = jnp.abs(lag)
                return (tau_spot + abs_lag) * jnp.exp(-abs_lag / tau_spot)

        else:
            # Generic fallback via precomputed grid (like skew-normal)
            import functools
            env = self.envelope
            tau_ref = env.tau_spot if env.tau_spot > 0 else 1.0
            env_np = lambda t_arr: np.asarray(env.Gamma(jnp.array(t_arr)))
            from .envelope import compute_R_Gamma_numerical
            lag_grid, R_vals = compute_R_Gamma_numerical(env_np, tau_ref)

            def r_gamma(theta_arr, lag):  # noqa: ARG001
                return jnp.interp(jnp.abs(lag), lag_grid, R_vals)

        return r_gamma

    # ── Bandwidth support ───────────────────────────────────────────────────

    def bandwidth_support(self, param_keys, bounds_arr) -> float:
        """
        Estimate the kernel support using upper bounds of parameters.

        Used by GPSolver._compute_bandwidth to determine the banded
        Cholesky bandwidth as a compile-time constant.

        Parameters
        ----------
        param_keys : sequence of str
            Parameter names in the same order as bounds_arr.
        bounds_arr : array_like, shape (n_params, 2)
            Lower and upper bounds for each parameter.
        """
        keys = list(param_keys)
        bounds_arr = np.asarray(bounds_arr)

        def upper(key, fallback):
            if key in keys:
                return float(bounds_arr[keys.index(key), 1])
            log_key = f"log_{key}"
            if log_key in keys:
                return 10.0 ** float(bounds_arr[keys.index(log_key), 1])
            return float(fallback)

        if self.envelope is None:
            return 0.0

        if isinstance(self.envelope, TrapezoidSymmetricEnvelope):
            return (upper("lspot", self.lspot)
                    + 2.0 * upper("tau_spot", self.tau_spot))

        elif isinstance(self.envelope, TrapezoidAsymmetricEnvelope):
            return (upper("lspot",   self.lspot)
                    + upper("tau_em",  self.envelope.tau_em)
                    + upper("tau_dec", self.envelope.tau_dec))

        elif isinstance(self.envelope, SkewedGaussianEnvelope):
            return 12.0 * upper("sigma_sn", self.envelope.sigma_sn)

        elif isinstance(self.envelope, ExponentialEnvelope):
            return 6.0 * upper("tau_spot", self.tau_spot)

        else:
            return self.envelope.kernel_support()

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
