"""
visibility.py — Stellar visibility functions for starspot models.

VisibilityFunction encapsulates the Fourier-series representation of the
stellar visibility function: how much flux a spot at latitude phi
contributes as the star rotates, decomposed into rotation harmonics.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

__all__ = [
    "VisibilityFunction",
    "EdgeOnVisibilityFunction",
    "FullGeometryVisibilityFunction",
    "LimbDarkenedVisibilityFunction",
    "FullGeometryLimbDarkenedVisibilityFunction",
    # low-level helpers re-exported for backward compat
    "_cn_general_jax",
    "_cn_squared_coefficients_jax",
    "_gauss_legendre_grid",
]


def _as_harmonic_orders(harmonics) -> tuple:
    """
    Normalize a harmonic specification to a tuple of non-negative ints.

    Accepts either a scalar ``n`` -- shorthand for the contiguous set
    ``(0, 1, ..., n)`` that the ``n_harmonics`` arguments have always meant --
    or an explicit sequence of orders such as ``[0, 2, 4]``.
    """
    arr = np.asarray(harmonics)
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(
            f"harmonics must be numeric, got dtype {arr.dtype}")
    if np.any(arr != np.round(arr)):
        raise ValueError(f"harmonic orders must be integers, got {harmonics}")
    if arr.ndim == 0:
        n = int(arr)
        if n < 0:
            raise ValueError(f"n_harmonics must be >= 0, got {n}")
        return tuple(range(n + 1))
    if arr.ndim > 1:
        raise ValueError(
            f"harmonics must be 1-D, got shape {arr.shape}")
    orders = tuple(int(n) for n in arr)
    if not orders:
        raise ValueError("harmonics must contain at least one order")
    if any(n < 0 for n in orders):
        raise ValueError(f"harmonic orders must be >= 0, got {orders}")
    if len(set(orders)) != len(orders):
        raise ValueError(f"harmonic orders must be unique, got {orders}")
    return orders


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

    # At tiny_a1 the spot sits at a constant angle from the line of sight
    # (pole-on viewing, or a spot at a rotational pole): cos(beta) = a0 for
    # all longitudes, so the mean visible projected area is max(a0, 0).
    # The clamp matters when a0 < 0 (spot hidden on the far hemisphere at
    # phi = -pi/2); without it c_0 picked up a spurious negative value.
    c0 = jnp.where(tiny_a1, jnp.maximum(a0, 0.0),
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
    """
    Compute ``|c_n|^2`` at the requested harmonic orders.

    ``n_harmonics`` is either an int ``n`` (orders 0, 1, ..., n) or an
    explicit sequence of orders, e.g. ``[0, 2, 4]``.  Element ``i`` of the
    result corresponds to order ``i`` in the resolved set.
    """
    ns = jnp.asarray(_as_harmonic_orders(n_harmonics))
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
    harmonics : sequence of int or int
        Rotation harmonic orders to retain, default ``(0, 1, 2)``.  The
        orders need not be contiguous -- ``[0, 2, 4]`` keeps only the even
        harmonics.  A scalar ``n`` is shorthand for ``(0, 1, ..., n)``.
        :meth:`cn_squared` returns one coefficient per order, in the order
        given here.
    """

    def __init__(self, peq: float, kappa: float, inc: float,
                 harmonics=(0, 1, 2)):
        self.peq = float(peq)
        self.kappa = float(kappa)
        self.inc = float(inc)
        self.harmonics = _as_harmonic_orders(harmonics)

    @property
    def param_dict(self) -> dict:
        """Visibility parameters as {name: value}."""
        return {"peq": self.peq, "kappa": self.kappa, "inc": self.inc}

    @property
    def param_keys(self) -> tuple:
        """Ordered parameter names."""
        return ("peq", "kappa", "inc")

    def _orders(self, n_harmonics=None) -> tuple:
        """
        Resolve the harmonic orders for a ``cn_squared``-style call.

        ``None`` selects the orders this instance was constructed with;
        anything else is interpreted by :func:`_as_harmonic_orders`, so
        callers holding an int ``n_harmonics`` (the kernel classes) keep
        the contiguous ``0..n`` behaviour.
        """
        if n_harmonics is None:
            return self.harmonics
        return _as_harmonic_orders(n_harmonics)

    def omega0(self, phi):
        """Latitude-dependent rotation angular frequency [rad/day]."""
        return 2.0 * jnp.pi * (1.0 - self.kappa * jnp.sin(phi) ** 2) / self.peq

    def cn_squared(self, phi, n_harmonics=None):
        """
        Squared Fourier coefficients ``|c_n|^2`` at stellar latitude phi.

        Parameters
        ----------
        phi : float
            Spot latitude [radians].
        n_harmonics : int or sequence of int, optional
            Overrides :attr:`harmonics` for this call; an int ``n`` means
            orders ``0..n``.  Defaults to :attr:`harmonics`.

        Returns
        -------
        cn_sq : jnp.ndarray, shape (n_orders,)
            One entry per resolved harmonic order.
        """
        return _cn_squared_coefficients_jax(
            self.inc, phi, self._orders(n_harmonics))

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
            Sympy expressions keyed by name::

                {"omega0": expr, "a0": expr, "a1": expr,
                 "theta_v": expr, "c0": expr, "c1": expr, "cn": expr}
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
    are known analytically.  The per-latitude coefficients g_n (Eq. 68 of
    Birky et al.) are:

        g_0 = 1/pi,  g_1 = 1/4,  g_2 = 1/(3*pi)

    and the latitude-averaged *squared* coefficients are
    ``<|c_n|^2> = g_n^2/2``::

        <|c_0|^2> = 1 / (2 * pi^2)
        <|c_1|^2> = 1 / 32
        <|c_2|^2> = 1 / (18 * pi^2)

    The rotation frequency is latitude-independent:
        omega_0 = 2 * pi / P_eq.

    This eliminates the need for numerical latitude quadrature, making
    kernel evaluation significantly faster.

    Parameters
    ----------
    peq : float
        Equatorial rotation period [days].
    harmonics : sequence of int or int
        Rotation harmonic orders to retain, default ``(0, 1, 2)``.  Orders
        above 2 have no power in this closed form and come back as zero.
    """

    # Pre-computed latitude-averaged |c_n|^2 = g_n^2 / 2
    # where g_0 = 1/pi, g_1 = 1/4, g_2 = 1/(3*pi).  All higher orders vanish.
    _CN_SQ = {
        0: 1.0 / (2.0 * np.pi ** 2),
        1: 1.0 / 32.0,
        2: 1.0 / (18.0 * np.pi ** 2),
    }

    def __init__(self, peq: float, harmonics=(0, 1, 2)):
        super().__init__(peq=peq, kappa=0.0, inc=jnp.pi / 2,
                         harmonics=harmonics)

    @property
    def param_dict(self) -> dict:
        return {"peq": self.peq}

    @property
    def param_keys(self) -> tuple:
        return ("peq",)

    def omega0(self, phi):
        """Rotation frequency (latitude-independent for kappa=0)."""
        return 2.0 * jnp.pi / self.peq

    def cn_squared(self, phi, n_harmonics=None):
        """Latitude-averaged ``|c_n|^2`` (independent of phi).

        Returns the closed-form coefficients for n = 0, 1, 2 and zero
        for higher harmonics.
        """
        ns = self._orders(n_harmonics)
        return jnp.array([self._CN_SQ.get(n, 0.0) for n in ns])


class FullGeometryVisibilityFunction(VisibilityFunction):
    """
    Exact projected spot area using the full piecewise geometry (Eq. 5
    of Birky et al.), without the small-spot approximation.

    The projected area of a circular spot with angular radius alpha at
    angle beta from the line of sight has three regimes:

    - **Fully visible** (0 < beta < pi/2 - alpha)::

        A = pi sin^2(alpha) cos(beta)

    - **Partially visible** (pi/2 - alpha < beta < pi/2 + alpha)::

        A = arccos[cos(alpha) csc(beta)]
            + cos(beta) sin^2(alpha) arccos[-cot(alpha) cot(beta)]
            - cos(alpha) sin(beta) sqrt(1 - cos^2(alpha) csc^2(beta))

    - **Hidden** (pi/2 + alpha < beta < pi)::

        A = 0

    The base ``VisibilityFunction`` uses the small-spot limit where
    A ~ pi alpha^2 cos(beta) and the partial-visibility window vanishes.
    This subclass retains the exact expressions for use in forward
    simulations with ``LightcurveModel``.

    The Fourier coefficients ``cn_squared`` are computed numerically by
    evaluating the projected area over one rotation period and taking the
    DFT, rather than using the analytic c_n formulas.

    Parameters
    ----------
    peq : float
        Equatorial rotation period [days].
    kappa : float
        Differential rotation shear (dimensionless).
    inc : float
        Stellar inclination [radians].
    alpha_ref : float
        Reference spot angular radius [radians] used for computing
        Fourier coefficients (default 0.1).
    n_lon : int
        Number of longitude points for the numerical DFT (default 512).
    harmonics : sequence of int or int
        Rotation harmonic orders to retain, default ``(0, 1, 2)``.
    """

    def __init__(self, peq: float, kappa: float, inc: float,
                 alpha_ref: float = 0.1, n_lon: int = 512,
                 harmonics=(0, 1, 2)):
        super().__init__(peq=peq, kappa=kappa, inc=inc, harmonics=harmonics)
        self.alpha_ref = float(alpha_ref)
        self.n_lon = int(n_lon)

    @staticmethod
    def projected_area(alpha, beta):
        """
        Exact projected area of a circular spot (Eq. 5 of Birky et al.).

        Implements the full piecewise function for a spot of angular
        radius ``alpha`` at angle ``beta`` from the line of sight.
        All three geometric cases (fully visible, partially occluded,
        hidden) are handled in a branchless JAX-compatible form.

        Parameters
        ----------
        alpha : array_like
            Spot angular radius [radians].
        beta : array_like
            Angle between spot normal and line of sight [radians].

        Returns
        -------
        A : jnp.ndarray
            Projected area (unnormalized; divide by pi for the fractional
            flux deficit).
        """
        alpha = jnp.asarray(alpha)
        beta = jnp.asarray(beta)

        cos_a = jnp.cos(alpha)
        sin_a = jnp.sin(alpha)
        cos_b = jnp.cos(beta)
        sin_b = jnp.sin(beta)

        # Guard against division by zero at beta = 0 or pi
        eps = 1e-30
        csc_b = 1.0 / (sin_b + eps)
        cot_b = cos_b / (sin_b + eps)
        cot_a = cos_a / (sin_a + eps)

        # Case 1: fully visible
        A_full = jnp.pi * sin_a ** 2 * cos_b

        # Case 2: partially visible (Eq. 5, middle branch)
        arg1 = jnp.clip(cos_a * csc_b, -1.0, 1.0)
        arg2 = jnp.clip(-cot_a * cot_b, -1.0, 1.0)
        sqrt_arg = jnp.clip(1.0 - cos_a ** 2 * csc_b ** 2, 0.0, None)

        A_partial = (jnp.arccos(arg1)
                     + cos_b * sin_a ** 2 * jnp.arccos(arg2)
                     - cos_a * sin_b * jnp.sqrt(sqrt_arg))

        # Select case based on beta relative to pi/2 ± alpha
        half_pi = jnp.pi / 2.0
        fully_visible = beta < (half_pi - alpha)
        hidden = beta > (half_pi + alpha)

        A = jnp.where(fully_visible, A_full,
                       jnp.where(hidden, 0.0, A_partial))

        # Zero out when spot has zero size
        A = jnp.where(alpha > 1e-15, A, 0.0)

        return A

    def cos_beta(self, phi, longitude):
        """
        Cosine of the angle between spot normal and line of sight (Eq. 6).

        Parameters
        ----------
        phi : float or array_like
            Spot latitude [radians].
        longitude : float or array_like
            Spot longitude relative to observer [radians].

        Returns
        -------
        cos_beta : jnp.ndarray
        """
        return (jnp.cos(self.inc) * jnp.sin(phi)
                + jnp.sin(self.inc) * jnp.cos(phi) * jnp.cos(longitude))

    def visibility_profile(self, phi, alpha, n_lon=None):
        """
        Compute the projected area as a function of longitude for a spot
        at latitude ``phi`` with angular radius ``alpha``.

        Parameters
        ----------
        phi : float
            Spot latitude [radians].
        alpha : float
            Spot angular radius [radians].
        n_lon : int, optional
            Number of longitude grid points (default: self.n_lon).

        Returns
        -------
        lon_grid : jnp.ndarray, shape (n_lon,)
            Longitude values in [0, 2*pi).
        A : jnp.ndarray, shape (n_lon,)
            Projected area at each longitude.
        """
        if n_lon is None:
            n_lon = self.n_lon
        lon_grid = jnp.linspace(0, 2 * jnp.pi, n_lon, endpoint=False)
        cos_b = self.cos_beta(phi, lon_grid)
        beta = jnp.arccos(jnp.clip(cos_b, -1.0, 1.0))
        A = self.projected_area(alpha, beta)
        return lon_grid, A

    def cn_squared(self, phi, n_harmonics=None):
        """
        Numerically computed squared Fourier coefficients from the full
        projected-area profile.

        Evaluates the exact projected area over one full rotation at
        latitude ``phi`` using the reference spot size ``alpha_ref``,
        then extracts harmonics via DFT.  The coefficients are
        normalized by the spot area ``pi * sin^2(alpha_ref)`` so they
        are independent of spot size (consistent with the base class).

        Parameters
        ----------
        phi : float
            Spot latitude [radians].
        n_harmonics : int or sequence of int, optional
            Overrides :attr:`harmonics` for this call; an int ``n`` means
            orders ``0..n``.

        Returns
        -------
        cn_sq : jnp.ndarray, shape (n_orders,)
        """
        ns = self._orders(n_harmonics)
        _, A = self.visibility_profile(phi, self.alpha_ref)
        # Normalize by the peak area (fully visible, cos_beta=1)
        norm = jnp.pi * jnp.sin(self.alpha_ref) ** 2
        norm = jnp.where(norm > 1e-30, norm, 1.0)
        A_norm = A / norm

        # DFT to extract Fourier coefficients
        fft_coeffs = jnp.fft.rfft(A_norm) / len(A_norm)
        # c_0 is the DC component, c_n for n>=1 are the cosine amplitudes
        cn = jnp.abs(fft_coeffs[jnp.asarray(ns)])
        return cn ** 2


class LimbDarkenedVisibilityFunction(VisibilityFunction):
    """
    Visibility function including stellar limb darkening.

    The base ``VisibilityFunction`` assumes a uniformly bright disk, so in
    the small-spot limit a spot's flux deficit follows the projected area
    alone,

        V(mu) = max(mu, 0),
        mu = cos(beta) = cos(i) sin(phi) + sin(i) cos(phi) cos(theta),

    where ``theta`` is the rotational longitude.  With limb darkening the
    deficit is weighted by the local specific intensity at the spot and
    normalized by the disk-integrated flux,

        V(mu) = mu I(mu) / F,    F = int_0^1 2 mu I(mu) dmu,

    so that V reduces *exactly* to the base class when I(mu) == 1.

    Two intensity laws are supported:

    - ``law="quadratic"``: I(mu)/I(1) = 1 - u1 (1 - mu) - u2 (1 - mu)^2,
      with F = 1 - u1/3 - u2/6.

    - ``law="claret"``: the four-coefficient nonlinear law
      I(mu)/I(1) = 1 - sum_k c_k (1 - mu^(k/2)),  k = 1..4,
      with F = 1 - sum_k c_k k / (k + 4).  This matches the coefficient
      convention used by ``LightcurveModel.limbc``.

    Unlike the uniform-disk case, ``c_n`` has no closed form for a general
    I(mu), so the coefficients are obtained by evaluating V over one full
    rotation and taking the DFT -- the same strategy used by
    ``FullGeometryVisibilityFunction``.  ``n_lon`` sets the longitude
    resolution; 512 gives ``|c_n|^2`` accurate to ~1e-7.

    Limb darkening redistributes power toward higher harmonics: the
    intensity weighting sharpens the visibility profile, so harmonics that
    vanish for a uniform disk acquire real power.  Prefer a larger
    ``n_harmonics`` on the kernel than you would use for the uniform-disk
    case.

    Parameters
    ----------
    peq : float
        Equatorial rotation period [days].
    kappa : float
        Differential rotation shear (dimensionless).
    inc : float
        Stellar inclination [radians].
    u : sequence of float
        Limb-darkening coefficients: ``(u1, u2)`` for ``law="quadratic"``,
        ``(c1, c2, c3, c4)`` for ``law="claret"``.
    law : {"quadratic", "claret"}
        Intensity law (default "quadratic").
    n_lon : int
        Number of longitude grid points for the DFT (default 512).
    harmonics : sequence of int or int
        Rotation harmonic orders to retain, default ``(0, 1, 2)``.  Because
        limb darkening pushes power into higher harmonics, prefer a wider
        set here than for the uniform-disk case.

    Examples
    --------
    >>> import numpy as np
    >>> from spotgp import (SpotEvolutionModel, TrapezoidSymmetricEnvelope,
    ...                     LimbDarkenedVisibilityFunction)
    >>> vis = LimbDarkenedVisibilityFunction(
    ...     peq=4.0, kappa=0.0, inc=np.pi / 3, u=(0.4, 0.2))
    >>> model = SpotEvolutionModel(
    ...     envelope=TrapezoidSymmetricEnvelope(lspot=5.0, tau_spot=2.0),
    ...     visibility=vis, sigma_k=0.01)
    """

    def __init__(self, peq, kappa, inc, u=(0.3, 0.2), law="quadratic",
                 n_lon=512, harmonics=(0, 1, 2)):
        super().__init__(peq=peq, kappa=kappa, inc=inc, harmonics=harmonics)
        if law not in ("quadratic", "claret"):
            raise ValueError(
                f"Unknown limb-darkening law: {law!r}. "
                "Use 'quadratic' or 'claret'.")
        if law == "quadratic" and len(u) != 2:
            raise ValueError(
                "quadratic law needs 2 coefficients (u1, u2), "
                f"got {len(u)}")
        if law == "claret" and len(u) != 4:
            raise ValueError(
                f"claret law needs 4 coefficients (c1..c4), got {len(u)}")
        self.law = law
        self.u = tuple(float(x) for x in u)
        self.n_lon = int(n_lon)

    @property
    def param_dict(self) -> dict:
        return {"peq": self.peq, "kappa": self.kappa, "inc": self.inc}

    @property
    def param_keys(self) -> tuple:
        return ("peq", "kappa", "inc")

    # ── Intensity profile ───────────────────────────────────────────────────

    def intensity(self, mu):
        """Normalized specific intensity I(mu)/I(1)."""
        if self.law == "quadratic":
            u1, u2 = self.u
            return 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu) ** 2
        total = 1.0
        for k, ck in enumerate(self.u, start=1):
            total = total - ck * (1.0 - mu ** (k / 2.0))
        return total

    @property
    def flux_norm(self):
        """Disk-integrated flux F = int_0^1 2 mu I(mu) dmu."""
        if self.law == "quadratic":
            u1, u2 = self.u
            return 1.0 - u1 / 3.0 - u2 / 6.0
        return 1.0 - sum(ck * k / (k + 4.0)
                         for k, ck in enumerate(self.u, start=1))

    # ── Visibility profile ──────────────────────────────────────────────────

    def visibility_profile(self, phi, inc=None, n_lon=None):
        """
        Limb-darkened visibility over one rotation at latitude ``phi``.

        ``inc`` may be supplied explicitly (e.g. a JAX tracer during
        fitting); it defaults to ``self.inc``.

        Returns
        -------
        lon : jnp.ndarray, shape (n_lon,)
        V : jnp.ndarray, shape (n_lon,)
        """
        inc = self.inc if inc is None else inc
        n_lon = self.n_lon if n_lon is None else n_lon
        lon = jnp.linspace(0.0, 2.0 * jnp.pi, n_lon, endpoint=False)
        mu = (jnp.cos(inc) * jnp.sin(phi)
              + jnp.sin(inc) * jnp.cos(phi) * jnp.cos(lon))
        # Clip to a strictly positive floor before evaluating I(mu).  The
        # Claret law's mu**(k/2) terms have infinite slope at mu = 0, and
        # jnp.where propagates gradients through *both* branches, so a floor
        # of exactly 0.0 yields inf * 0 = NaN in reverse-mode autodiff.
        mu_vis = jnp.clip(mu, 1e-12, 1.0)
        V = mu_vis * self.intensity(mu_vis) / self.flux_norm
        return lon, jnp.where(mu > 0.0, V, 0.0)

    # ── Fourier coefficients ────────────────────────────────────────────────

    def cn_squared(self, phi, n_harmonics=None):
        """Squared Fourier coefficients ``|c_n|^2`` at latitude phi."""
        return self.cn_sq_at(self.inc, phi, n_harmonics)

    def cn_sq_at(self, inc, phi, n_harmonics=None):
        """
        ``|c_n|^2`` at an explicitly supplied inclination.

        Separated from :meth:`cn_squared` so ``inc`` can be a JAX tracer,
        which is what lets gradients flow to ``inc`` during fitting.
        """
        ns = self._orders(n_harmonics)
        _, V = self.visibility_profile(phi, inc=inc)
        cn = jnp.abs(jnp.fft.rfft(V) / V.shape[0])
        return cn[jnp.asarray(ns)] ** 2

    def cn_sq_jax(self, theta_vis, phi, n_harmonics=None):
        """
        JAX-traceable ``|c_n|^2`` from the visibility slice of the theta vector.

        ``theta_vis`` is ``[peq, kappa, inc]`` -- the leading entries of
        ``theta_arr``, matching :attr:`param_keys`.  This method is the hook
        ``SpotEvolutionModel.get_cn_sq_func`` looks for; defining it is what
        makes a visibility subclass participate in the latitude-averaged
        kernel (``AnalyticKernel.kernel`` and the ``GPSolver``
        log-likelihood) rather than only the PSD / single-latitude paths.
        """
        return self.cn_sq_at(theta_vis[2], phi, n_harmonics)

    def get_sympy(self, display=True, status=None):
        """Limb-darkened c_n are numerical; there is no closed form."""
        raise NotImplementedError(
            "LimbDarkenedVisibilityFunction computes c_n numerically by DFT; "
            "there is no closed-form expression to render. Use "
            "visibility_profile() to inspect V(theta) instead.")


class FullGeometryLimbDarkenedVisibilityFunction(LimbDarkenedVisibilityFunction):
    """
    Visibility function with the full finite-spot geometry *and* limb
    darkening.

    Combines the two effects that the sibling subclasses treat separately:

    - :class:`FullGeometryVisibilityFunction` keeps the exact projected
      area of a spot with finite angular radius ``alpha`` -- including the
      partial-visibility regime at the limb -- but assumes a uniformly
      bright disk.
    - :class:`LimbDarkenedVisibilityFunction` weights the deficit by the
      local specific intensity ``I(mu)`` but takes the small-spot limit,
      evaluating ``mu`` only at the spot centre.

    Here the flux deficit is the surface integral of the projected,
    intensity-weighted area over the *visible part* of the spot cap.  With
    the cap parameterized by angular distance ``rho`` from the spot centre
    and azimuth ``psi`` (surface element ``sin(rho) drho dpsi``), a point's
    angle from the line of sight follows the spherical law of cosines,

        mu(rho, psi) = cos(beta) cos(rho) + sin(beta) sin(rho) cos(psi),

    where ``beta`` is the angle between the spot centre and the line of
    sight, ``cos(beta) = cos(i) sin(phi) + sin(i) cos(phi) cos(theta)``.
    The visibility is then

        V(phi, theta) = D(beta) / (pi sin^2(alpha) F),

        D(beta) = int_0^alpha int_0^2pi I(mu) max(mu, 0)
                  sin(rho) dpsi drho,

        F = int_0^1 2 mu I(mu) dmu.

    The ``max(mu, 0)`` mask reproduces the full piecewise geometry of
    Eq. 5 without any branch analysis: for ``I == 1`` the double integral
    equals the exact projected area (all three regimes), so this class
    reduces to :class:`FullGeometryVisibilityFunction`; for
    ``alpha -> 0`` it reduces to :class:`LimbDarkenedVisibilityFunction`;
    and with both simplifications it recovers the base class.  The
    normalization ``pi sin^2(alpha)`` (spot area, so V is per unit spot
    area) and ``F`` (disk-integrated flux) match those conventions.

    The double integral is evaluated on a fixed Gauss-Legendre grid in
    ``rho`` and a uniform (trapezoidal, exact for periodic integrands)
    grid in ``psi``; the Fourier coefficients ``|c_n|^2`` are then
    extracted by DFT over one rotation, exactly as in the two numeric
    sibling classes.  Everything is branchless jnp array math, so
    :meth:`cn_sq_at` remains differentiable in ``inc`` and the inherited
    ``cn_sq_jax`` hook makes the class participate in the
    latitude-averaged kernel (``AnalyticKernel`` covariance and the
    ``GPSolver`` log-likelihood), not only the forward-model paths.

    Note that, unlike the small-spot classes, the harmonic content now
    depends on the spot size: a larger ``alpha_ref`` smooths the
    visibility profile (the spot spends longer partially occulted at the
    limb) and shifts power between harmonics.  ``alpha_ref`` is treated
    as fixed configuration, like the limb-darkening coefficients.

    Parameters
    ----------
    peq : float
        Equatorial rotation period [days].
    kappa : float
        Differential rotation shear (dimensionless).
    inc : float
        Stellar inclination [radians].
    alpha_ref : float
        Spot angular radius [radians] used for the cap integral and the
        area normalization (default 0.1).  Must lie in (0, pi/2).
    u : sequence of float
        Limb-darkening coefficients: ``(u1, u2)`` for ``law="quadratic"``,
        ``(c1, c2, c3, c4)`` for ``law="claret"``.  ``u=(0.0, 0.0)``
        recovers the uniform-disk full-geometry case.
    law : {"quadratic", "claret"}
        Intensity law (default "quadratic").
    n_lon : int
        Number of longitude grid points for the DFT (default 512).
    n_rho, n_psi : int
        Quadrature resolution over the spot cap (default 32 x 64).  The
        integrand is smooth except along the visibility terminator
        crossing the cap, so the defaults give |c_n|^2 to ~1e-6; double
        them for spots larger than ~0.3 rad.
    harmonics : sequence of int or int
        Rotation harmonic orders to retain, default ``(0, 1, 2)``.  Both
        limb darkening and the finite spot size move power across
        harmonics, so prefer a wider set than for the base class.

    Examples
    --------
    >>> import numpy as np
    >>> from spotgp import FullGeometryLimbDarkenedVisibilityFunction
    >>> vis = FullGeometryLimbDarkenedVisibilityFunction(
    ...     peq=4.0, kappa=0.0, inc=np.pi / 3, alpha_ref=0.2, u=(0.4, 0.2))
    >>> lon, V = vis.visibility_profile(phi=np.deg2rad(30))
    """

    def __init__(self, peq, kappa, inc, alpha_ref=0.1, u=(0.3, 0.2),
                 law="quadratic", n_lon=512, n_rho=32, n_psi=64,
                 harmonics=(0, 1, 2)):
        super().__init__(peq=peq, kappa=kappa, inc=inc, u=u, law=law,
                         n_lon=n_lon, harmonics=harmonics)
        alpha_ref = float(alpha_ref)
        if not 0.0 < alpha_ref < np.pi / 2:
            raise ValueError(
                f"alpha_ref must lie in (0, pi/2), got {alpha_ref}")
        self.alpha_ref = alpha_ref
        self.n_rho = int(n_rho)
        self.n_psi = int(n_psi)

        # Fixed quadrature over the spot cap: Gauss-Legendre in rho on
        # [0, alpha_ref]; uniform psi with weight 2*pi/n_psi (the
        # trapezoid rule, exact for periodic integrands).  sin(rho) is
        # folded into the rho weights since it multiplies every integrand.
        rho, w_rho = _gauss_legendre_grid(self.n_rho, 0.0, alpha_ref)
        self._rho = rho                                  # (n_rho,)
        self._w_rho = w_rho * jnp.sin(rho)               # (n_rho,)
        self._psi = jnp.linspace(0.0, 2.0 * jnp.pi, self.n_psi,
                                 endpoint=False)         # (n_psi,)
        self._w_psi = 2.0 * jnp.pi / self.n_psi
        # Spot area, the per-unit-spot-area normalization of V
        self._spot_area = np.pi * float(np.sin(alpha_ref)) ** 2

    # ── Spot-cap integral ───────────────────────────────────────────────────

    def spot_deficit(self, cos_beta):
        """
        Limb-darkened flux deficit ``D(beta)`` of the finite spot.

        Integrates ``I(mu) mu`` over the visible part of the spot cap for
        each supplied ``cos(beta)``.  For ``I == 1`` this equals the exact
        projected area of :meth:`FullGeometryVisibilityFunction.projected_area`
        (all three visibility regimes emerge from the ``mu > 0`` mask).

        Parameters
        ----------
        cos_beta : array_like, shape (...,)
            Cosine of the angle between the spot centre and the line of
            sight.  May be a JAX tracer.

        Returns
        -------
        D : jnp.ndarray, shape (...,)
            Unnormalized deficit; divide by ``pi sin^2(alpha_ref)`` for
            the fractional-area convention and by :attr:`flux_norm` for
            the limb-darkening normalization.
        """
        cos_b = jnp.asarray(cos_beta)[..., None, None]   # (..., 1, 1)
        # Positive floor inside the clip: at cos_beta = +/-1 (spot centre
        # crossing disk centre or anticentre) sqrt(0) has an infinite
        # gradient, and the terms sin_b multiplies vanish there by psi
        # symmetry, so inf * 0 = NaN in autodiff.  Clip's zero gradient
        # outside its active range removes the inf; the sin_b-channel
        # contribution it drops is 0 at that point.
        sin_b = jnp.sqrt(jnp.clip(1.0 - cos_b ** 2, 1e-24, 1.0))
        cos_r = jnp.cos(self._rho)[:, None]              # (n_rho, 1)
        sin_r = jnp.sin(self._rho)[:, None]
        cos_p = jnp.cos(self._psi)[None, :]              # (1, n_psi)

        mu = cos_b * cos_r + sin_b * sin_r * cos_p       # (..., n_rho, n_psi)
        # Strictly positive floor before I(mu): the Claret law's mu**(k/2)
        # terms have infinite slope at mu = 0 and jnp.where propagates
        # gradients through both branches (same guard as the parent).
        mu_vis = jnp.clip(mu, 1e-12, 1.0)
        integrand = jnp.where(mu > 0.0,
                              self.intensity(mu_vis) * mu_vis, 0.0)
        # psi sum (uniform weight), then weighted rho sum
        return self._w_psi * jnp.einsum(
            "...rp,r->...", integrand, self._w_rho)

    # ── Visibility profile ──────────────────────────────────────────────────

    def visibility_profile(self, phi, inc=None, n_lon=None):
        """
        Finite-spot, limb-darkened visibility over one rotation.

        Same signature as the parent class, so the inherited DFT-based
        ``cn_squared`` / ``cn_sq_at`` / ``cn_sq_jax`` operate on this
        profile unchanged.  ``inc`` may be a JAX tracer.

        Returns
        -------
        lon : jnp.ndarray, shape (n_lon,)
        V : jnp.ndarray, shape (n_lon,)
            Deficit per unit spot area, ``D(beta) / (pi sin^2(alpha) F)``.
        """
        inc = self.inc if inc is None else inc
        n_lon = self.n_lon if n_lon is None else n_lon
        lon = jnp.linspace(0.0, 2.0 * jnp.pi, n_lon, endpoint=False)
        cos_b = (jnp.cos(inc) * jnp.sin(phi)
                 + jnp.sin(inc) * jnp.cos(phi) * jnp.cos(lon))
        V = self.spot_deficit(cos_b) / (self._spot_area * self.flux_norm)
        return lon, V

    def get_sympy(self, display=True, status=None):
        """The combined c_n are numerical; there is no closed form."""
        raise NotImplementedError(
            "FullGeometryLimbDarkenedVisibilityFunction computes c_n "
            "numerically (spot-cap quadrature + DFT); there is no "
            "closed-form expression to render. Use visibility_profile() "
            "to inspect V(theta) instead.")
