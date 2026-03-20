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


class FullGeometryVisibilityFunction(VisibilityFunction):
    """
    Exact projected spot area using the full piecewise geometry (Eq. 5
    of Birky et al.), without the small-spot approximation.

    The projected area of a circular spot with angular radius alpha at
    angle beta from the line of sight has three regimes:

    - **Fully visible** (0 < beta < pi/2 - alpha):
      A = pi sin^2(alpha) cos(beta)

    - **Partially visible** (pi/2 - alpha < beta < pi/2 + alpha):
      A = arccos[cos(alpha) csc(beta)]
          + cos(beta) sin^2(alpha) arccos[-cot(alpha) cot(beta)]
          - cos(alpha) sin(beta) sqrt(1 - cos^2(alpha) csc^2(beta))

    - **Hidden** (pi/2 + alpha < beta < pi):
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
    """

    def __init__(self, peq: float, kappa: float, inc: float,
                 alpha_ref: float = 0.1, n_lon: int = 512):
        super().__init__(peq=peq, kappa=kappa, inc=inc)
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

    def cn_squared(self, phi, n_harmonics: int = 3):
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
        n_harmonics : int
            Number of harmonics (default 3).

        Returns
        -------
        cn_sq : jnp.ndarray, shape (n_harmonics + 1,)
        """
        _, A = self.visibility_profile(phi, self.alpha_ref)
        # Normalize by the peak area (fully visible, cos_beta=1)
        norm = jnp.pi * jnp.sin(self.alpha_ref) ** 2
        norm = jnp.where(norm > 1e-30, norm, 1.0)
        A_norm = A / norm

        # DFT to extract Fourier coefficients
        fft_coeffs = jnp.fft.rfft(A_norm) / len(A_norm)
        # c_0 is the DC component, c_n for n>=1 are the cosine amplitudes
        cn = jnp.abs(fft_coeffs[:n_harmonics + 1])
        return cn ** 2
