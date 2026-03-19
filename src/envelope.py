"""
envelope.py — Spot envelope function hierarchy.

Defines the abstract base class EnvelopeFunction and four concrete
implementations: TrapezoidSymmetricEnvelope, TrapezoidAsymmetricEnvelope,
SkewedGaussianEnvelope, and ExponentialEnvelope.

To define a custom envelope, subclass EnvelopeFunction and implement:
  - tau_spot  (property)  : characteristic timescale [days]   [REQUIRED]
  - Gamma(t)         : normalized envelope, peak = 1      [REQUIRED]
  - param_dict       : {name: value} of free parameters  [optional, needed for GPSolver]
  - lspot (property) : plateau duration [days]            [optional, default 0]
  - R_Gamma(lag)     : autocorrelation                    [optional, default: FFT]
  - Gamma_hat(omega) : |FT[Gamma]|(omega)                 [optional, default: FFT]
  - kernel_support() : upper lag support                  [optional, default: lspot+6*tau_spot]
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

__all__ = [
    "EnvelopeFunction",
    "TrapezoidSymmetricEnvelope",
    "TrapezoidAsymmetricEnvelope",
    "SkewedGaussianEnvelope",
    "ExponentialEnvelope",
    # low-level helpers (re-exported for backward compat with analytic_kernel)
    "compute_R_Gamma_numerical",
    "_Gamma_hat",
    "_R_Gamma_symmetric",
    "_R_Gamma_asymmetric",
    "_skew_normal_envelope_func",
    "_compute_Gamma_hat_sq_numerical",
]


# ── Low-level JAX helpers ───────────────────────────────────────────────────

@jax.jit
def _Gamma_hat(omega, ell, tau_spot):
    """
    Fourier transform of the normalized squared envelope Gamma(t).

    Uses safe_w = max(|omega|, eps) to avoid 1/w^3 singularity at omega=0.
    """
    omega = jnp.asarray(omega, dtype=float)
    safe_w = jnp.where(jnp.abs(omega) > 1e-14, omega, 1.0)

    nz_result = (4 / (tau_spot**2 * safe_w**3) *
                 (tau_spot * safe_w * jnp.cos(safe_w * ell / 2)
                  + jnp.sin(safe_w * ell / 2)
                  - jnp.sin(safe_w * ell / 2 + safe_w * tau_spot)))

    zero_result = ell + 2 * tau_spot / 3
    return jnp.where(jnp.abs(omega) > 1e-14, nz_result, zero_result)


@jax.jit
def _R_Gamma_symmetric(lag, ell, tau_s):
    """
    Closed-form autocorrelation of the symmetric trapezoidal envelope.

    Piecewise degree-5 polynomial on [0, ell + 2*tau_s], zero beyond.
    Assumes ell/2 > tau_s.
    """
    t = jnp.abs(jnp.asarray(lag, dtype=float).ravel())

    R1 = (ell + 2*tau_s/5
           - 4*t**2 / (3*tau_s)
           + 2*t**3 / (3*tau_s**2)
           - t**5 / (15*tau_s**4))

    R2 = ell + 2*tau_s/3 - t

    R3 = (t**5 / (30*tau_s**4)
           - (ell + 2*tau_s) * t**4 / (6*tau_s**4)
           + (ell**2 + 4*ell*tau_s + 2*tau_s**2) * t**3 / (3*tau_s**4)
           - ell*(ell**2 + 6*ell*tau_s + 6*tau_s**2) * t**2 / (3*tau_s**4)
           + (ell**4 + 8*ell**3*tau_s + 12*ell**2*tau_s**2
              - 6*tau_s**4) * t / (6*tau_s**4)
           + (-ell**5 - 10*ell**4*tau_s - 20*ell**3*tau_s**2
              + 30*ell*tau_s**4 + 20*tau_s**5) / (30*tau_s**4))

    R4 = (ell + 2*tau_s - t)**5 / (30*tau_s**4)

    return jnp.where(t <= tau_s, R1,
           jnp.where(t <= ell, R2,
           jnp.where(t <= ell + tau_s, R3,
           jnp.where(t <= ell + 2*tau_s, R4,
                     0.0))))


@jax.jit
def _R_Gamma_asymmetric(lag, ell, te, td):
    """
    Closed-form autocorrelation of the asymmetric trapezoidal envelope.

    Assumes te <= td (enforced by caller via min/max swap).
    Six intervals on [0, ell + te + td], zero beyond.
    """
    t = jnp.abs(jnp.asarray(lag, dtype=float).ravel())

    td2 = td**2
    te2 = te**2
    td2te2 = td2 * te2
    ell2 = ell**2
    ell3 = ell**3

    R1 = (ell + (te + td) / 5
          - 2 * (1/te + 1/td) / 3 * t**2
          + (1/te2 + 1/td2) / 3 * t**3
          - (1/te**4 + 1/td**4) / 30 * t**5)

    R2 = (ell + te/3 + td/5
          - t / 2
          - 2 * t**2 / (3 * td)
          + t**3 / (3 * td2)
          - t**5 / (30 * td**4))

    R3 = ell + (te + td) / 3 - t

    R4 = (t**5 / (30 * td2te2)
          - (ell + td + te) * t**4 / (6 * td2te2)
          + (ell2 + 2*ell*td + 2*ell*te + 2*td*te) * t**3 / (3 * td2te2)
          - ell * (ell2 + 3*ell*td + 3*ell*te + 6*td*te) * t**2 / (3 * td2te2)
          + (ell**4 + 4*ell3*td + 4*ell3*te + 12*ell2*td*te
             - 6*td2te2) * t / (6 * td2te2)
          + (-ell**5 - 5*ell**4*td - 5*ell**4*te - 20*ell3*td*te
             + 30*ell*td2te2 + 10*td**3*te2 + 10*td2*te**3) / (30 * td2te2))

    R5 = (-t**3 / (3 * td2)
          + (ell + td + te/3) * t**2 / td2
          - (6*ell2 + 12*ell*td + 4*ell*te + 6*td2 + 4*td*te + te2) * t / (6 * td2)
          + (ell3/3 + ell2*td + ell2*te/3 + ell*td2 + 2*ell*td*te/3
             + ell*te2/6 + td**3/3 + td2*te/3 + td*te2/6 + te**3/30) / td2)

    D = ell + te + td - t
    R6 = D**5 / (30 * td2te2)

    return jnp.where(t <= te, R1,
           jnp.where(t <= td, R2,
           jnp.where(t <= ell, R3,
           jnp.where(t <= ell + te, R4,
           jnp.where(t <= ell + td, R5,
           jnp.where(t <= ell + te + td, R6,
                     0.0))))))


def compute_R_Gamma_numerical(envelope_func, tau_ref, n_grid=4096, extent=12.0):
    """
    Compute R_Gamma(lag) = ∫ Gamma(t) · Gamma(t + lag) dt via FFT.

    Parameters
    ----------
    envelope_func : callable
        f(t: np.ndarray) -> np.ndarray, the normalized envelope Gamma(t).
    tau_ref : float
        Reference timescale [days] setting grid extent and resolution.
    n_grid : int
        Number of time-grid points (default 4096).
    extent : float
        Grid half-width in units of tau_ref (default 12.0).

    Returns
    -------
    lag_grid : jnp.ndarray, shape (n_grid,)
    R_Gamma_vals : jnp.ndarray, shape (n_grid,)
    """
    T = float(extent) * float(tau_ref)
    t_np = np.linspace(-T, T, n_grid)
    dt = float(t_np[1] - t_np[0])

    env_np = np.asarray(envelope_func(t_np), dtype=np.float64)
    env_np = np.maximum(env_np, 0.0)

    env_fft = np.fft.rfft(env_np, n=2 * n_grid)
    R_vals = np.fft.irfft(np.abs(env_fft) ** 2, n=2 * n_grid)[:n_grid] * dt

    lag_grid = np.arange(n_grid, dtype=np.float64) * dt
    return jnp.array(lag_grid), jnp.array(R_vals)


def _compute_Gamma_hat_sq_numerical(envelope_func, tau_ref, n_grid=4096, extent=12.0):
    """
    Precompute |Gamma_hat(ω)|² for a numerical envelope (used by compute_psd).

    Returns
    -------
    omega_grid : jnp.ndarray, shape (n_grid + 1,)
    Gh_sq_vals : jnp.ndarray, shape (n_grid + 1,)
    """
    T = float(extent) * float(tau_ref)
    t_np = np.linspace(-T, T, n_grid)
    dt = float(t_np[1] - t_np[0])

    env_np = np.asarray(envelope_func(t_np), dtype=np.float64)
    env_np = np.maximum(env_np, 0.0)

    n_fft = 2 * n_grid
    env_fft = np.fft.rfft(env_np, n=n_fft) * dt
    Gh_sq = np.abs(env_fft) ** 2

    omega_grid = 2.0 * np.pi * np.fft.rfftfreq(n_fft, d=dt)
    return jnp.array(omega_grid), jnp.array(Gh_sq)


def _skew_normal_envelope_func(sigma_sn, n_sn):
    """
    Return a callable for the normalized skew-normal envelope.

    Implements Eq. (1) of Baranyi et al. (2021) A&A 653, A59:
        Gamma(t) ∝ exp(-t²/(2σ²)) · (1 + erf(n·t / (σ·√2)))

    n_sn < 0: rapid rise / slow decay.
    n_sn > 0: slow rise / rapid decay.
    n_sn = 0: symmetric Gaussian envelope.
    """
    from scipy.special import erf as _scipy_erf
    sigma = float(sigma_sn)
    n = float(n_sn)

    def _f(t):
        z = np.asarray(t, dtype=np.float64) / sigma
        env = np.exp(-z ** 2 / 2.0) * (1.0 + _scipy_erf(n * z / np.sqrt(2.0)))
        env = np.maximum(env, 0.0)
        peak = env.max()
        return env / peak if peak > 0.0 else env

    return _f


# ── Abstract base class ─────────────────────────────────────────────────────

class EnvelopeFunction(ABC):
    """
    Abstract base class for spot size envelope functions.

    To define a custom envelope, subclass this and implement:

      **Required:**
        - ``tau_spot`` (property) : characteristic timescale [days].  Used to set
          the extent of numerical integration grids.
        - ``Gamma(t)``       : normalized envelope, peak = 1.

      **Optional** (numerical defaults are provided):
        - ``lspot`` (property)    : plateau duration [days]. Default: 0.
        - ``param_dict`` (property): ``{name: value}`` dict of envelope
          parameters.  Needed for ``GPSolver`` to know which parameters to
          fit.  Default: ``{}`` (kernel evaluation still works; fitting will
          not expose envelope parameters).
        - ``R_Gamma(lag)``        : autocorrelation ∫ Γ(t)Γ(t+lag)dt.
          Default: computed via FFT from ``Gamma``.
        - ``Gamma_hat(omega)``    : |FT[Gamma]|(ω).
          Default: computed via FFT from ``Gamma``.
        - ``Gamma_hat_sq(omega)`` : |FT[Gamma]|²(ω).
          Default: ``Gamma_hat(omega) ** 2``.
        - ``kernel_support()``    : upper lag bound where R_Gamma is negligible.
          Default: ``lspot + 6 * tau_spot``.

    The numerical defaults are computed **lazily**: the FFT grids are built
    once on the first call and cached on the instance, so there is no cost for
    subclasses that override these methods.

    Example
    -------
    >>> class GaussianEnvelope(EnvelopeFunction):
    ...     def __init__(self, sigma):
    ...         self._sigma = float(sigma)
    ...     @property
    ...     def tau_spot(self):
    ...         return self._sigma
    ...     @property
    ...     def param_dict(self):
    ...         return {"tau_spot": self._sigma}
    ...     def Gamma(self, t):
    ...         import jax.numpy as jnp
    ...         return jnp.exp(-0.5 * (t / self._sigma) ** 2)
    """

    # ── Required ─────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def tau_spot(self) -> float:
        """Characteristic (scalar) timescale [days]."""

    @abstractmethod
    def Gamma(self, t):
        """
        Normalized spot-size envelope evaluated at relative times t.

        t = 0 is the center/peak of the envelope.  Must return values in
        [0, 1] with a peak of 1.  Should be JAX-compatible (jnp operations)
        so that it can be evaluated inside JIT-compiled code.
        """

    # ── Optional with defaults ────────────────────────────────────────────────

    @property
    def lspot(self) -> float:
        """Spot plateau duration [days]. Default 0 (no plateau)."""
        return 0.0

    @property
    def param_dict(self) -> dict:
        """
        Envelope parameters as ``{name: value}``.

        Override this to expose envelope parameters to ``GPSolver`` and
        ``SpotEvolutionModel``.  The default returns an empty dict, which means
        the envelope shape is fixed (not inferred during GP fitting).
        """
        return {}

    def kernel_support(self) -> float:
        """
        Upper bound on the lag support of R_Gamma [days].

        Used by ``GPSolver`` to compute the banded-Cholesky bandwidth.
        Default: ``lspot + 6 * tau_spot``.  Override for tighter bounds.
        """
        return self.lspot + 6.0 * self.tau_spot

    # ── Lazy numerical grid helpers ───────────────────────────────────────────

    def _ensure_numerical_grids(self):
        """
        Build R_Gamma and |Gamma_hat|² grids from ``Gamma`` via FFT.

        Called automatically by the default ``R_Gamma``, ``Gamma_hat``, and
        ``Gamma_hat_sq`` methods.  Results are cached on the instance so the
        FFT is only computed once.
        """
        if not hasattr(self, '_num_R_lag_grid'):
            env_func = lambda t_arr: np.asarray(
                self.Gamma(jnp.array(t_arr)), dtype=np.float64)
            self._num_R_lag_grid, self._num_R_vals = \
                compute_R_Gamma_numerical(env_func, self.tau_spot)
            self._num_Gh_omega_grid, self._num_Gh_sq_vals = \
                _compute_Gamma_hat_sq_numerical(env_func, self.tau_spot)

    # ── Default implementations ───────────────────────────────────────────────

    def R_Gamma(self, lag):
        """
        Autocorrelation R_Gamma(lag) = ∫ Gamma(t) · Gamma(t + lag) dt.

        Default: interpolated from an FFT-based precomputed grid.
        Override with an analytic expression for better performance.
        """
        self._ensure_numerical_grids()
        lag_abs = jnp.abs(jnp.asarray(lag, dtype=float).ravel())
        return jnp.interp(lag_abs, self._num_R_lag_grid, self._num_R_vals)

    def Gamma_hat(self, omega):
        """
        Fourier transform magnitude |FT[Gamma]|(omega).

        Default: interpolated from an FFT-based precomputed grid.
        Override with a closed-form expression for better performance.
        """
        self._ensure_numerical_grids()
        omega = jnp.asarray(omega, dtype=float)
        return jnp.interp(
            jnp.abs(omega),
            self._num_Gh_omega_grid,
            jnp.sqrt(self._num_Gh_sq_vals),
        )

    def Gamma_hat_sq(self, omega):
        """
        |FT[Gamma](omega)|² = Gamma_hat(omega)².

        Default: interpolated from an FFT-based precomputed grid.
        """
        self._ensure_numerical_grids()
        omega = jnp.asarray(omega, dtype=float)
        return jnp.interp(
            jnp.abs(omega),
            self._num_Gh_omega_grid,
            self._num_Gh_sq_vals,
        )

    # ── Sympy analytic expressions (optional overrides) ───────────────────────

    def sympy_Gamma(self):
        """
        Sympy expression for Gamma(t).

        Override in subclasses that have a closed-form envelope.
        Returns None if no analytic expression is available.
        """
        return None

    def sympy_Gamma_hat(self):
        """
        Sympy expression for Gamma_hat(omega) = FT[Gamma](omega).

        Override in subclasses with a closed-form Fourier transform.
        Returns None if no analytic form is available.
        """
        return None

    def sympy_R_Gamma(self):
        """
        Sympy expression for R_Gamma(tau) = integral Gamma(t) Gamma(t+tau) dt.

        Override in subclasses with a closed-form autocorrelation.
        Returns None if no analytic form is available.
        """
        return None

    def _compute_Gamma_hat_symbolic(self, gamma_expr):
        """Attempt to derive Gamma_hat from gamma_expr via symbolic integration."""
        if gamma_expr is None:
            return None
        try:
            import sympy as sp
            t     = sp.Symbol('t', real=True)
            omega = sp.Symbol(r'\omega', real=True)
            result = sp.integrate(
                gamma_expr * sp.exp(-sp.I * omega * t), (t, -sp.oo, sp.oo))
            return None if result.has(sp.Integral) else result
        except Exception:
            return None

    def _compute_R_Gamma_symbolic(self, gamma_expr):
        """Attempt to derive R_Gamma from gamma_expr via symbolic integration."""
        if gamma_expr is None:
            return None
        try:
            import sympy as sp
            t   = sp.Symbol('t', real=True)
            tau = sp.Symbol(r'\tau', real=True)
            result = sp.integrate(
                gamma_expr * gamma_expr.subs(t, t + tau), (t, -sp.oo, sp.oo))
            return None if result.has(sp.Integral) else result
        except Exception:
            return None

    def get_sympy(self, display=True, status=None, compute_symbolic=False):
        """
        Retrieve and display sympy expressions for Gamma, Gamma_hat, and R_Gamma.

        Prints or renders the LaTeX equation for each function that has a
        closed-form analytic expression, or '[numerical]' for functions that
        rely on FFT-based approximations.

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
            ``TrapezoidSymmetricEnvelope [user defined]``.
        compute_symbolic : bool, optional
            If True, attempt to derive Gamma_hat and R_Gamma symbolically
            from sympy_Gamma() when no explicit override exists.  Can be
            slow for complex envelopes.  Defaults to False.

        Returns
        -------
        dict
            ``{"Gamma": expr_or_None, "Gamma_hat": expr_or_None,
               "R_Gamma": expr_or_None}``
        """
        try:
            import sympy as sp
        except ImportError:
            raise ImportError(
                "sympy is required for get_sympy(). "
                "Install with: pip install sympy")

        gamma_expr = self.sympy_Gamma()

        # Use explicit override if subclass provides one; otherwise optionally
        # derive from sympy_Gamma via symbolic integration.
        if type(self).sympy_Gamma_hat is not EnvelopeFunction.sympy_Gamma_hat:
            gamma_hat_expr = self.sympy_Gamma_hat()
        elif compute_symbolic:
            gamma_hat_expr = self._compute_Gamma_hat_symbolic(gamma_expr)
        else:
            gamma_hat_expr = None

        if type(self).sympy_R_Gamma is not EnvelopeFunction.sympy_R_Gamma:
            R_gamma_expr = self.sympy_R_Gamma()
        elif compute_symbolic:
            R_gamma_expr = self._compute_R_Gamma_symbolic(gamma_expr)
        else:
            R_gamma_expr = None

        exprs = {
            "Gamma":     gamma_expr,
            "Gamma_hat": gamma_hat_expr,
            "R_Gamma":   R_gamma_expr,
        }

        lhs = {
            "Gamma":     r"\Gamma(t)",
            "Gamma_hat": r"\hat{\Gamma}(\omega)",
            "R_Gamma":   r"R_{\Gamma}(\tau)",
        }

        # Integral definitions shown when no closed-form analytic expression exists
        t      = sp.Symbol('t', real=True)
        omega  = sp.Symbol(r'\omega', real=True)
        tau    = sp.Symbol(r'\tau', real=True)
        Gamma  = sp.Function(r'\Gamma')
        integral_forms = {
            "Gamma_hat": sp.Integral(Gamma(t) * sp.exp(-sp.I * omega * t),
                                     (t, -sp.oo, sp.oo)),
            "R_Gamma":   sp.Integral(Gamma(t) * Gamma(t + tau),
                                     (t, 0, sp.oo)),
        }

        def _rhs_latex(key, expr):
            if expr is not None:
                return sp.latex(expr)
            integral = integral_forms.get(key)
            if integral is not None:
                return sp.latex(integral) + r" \quad \text{[numerical]}"
            return r"\text{[numerical]}"

        if display:
            status_tag = r" \text{[" + status + r"]}" if status else ""
            header = r"\textbf{" + type(self).__name__ + r"}" + status_tag
            try:
                from IPython.display import display as ipy_display, Math
                ipy_display(Math(header))
                for key, expr in exprs.items():
                    ipy_display(Math(lhs[key] + " = " + _rhs_latex(key, expr)))
            except ImportError:
                status_str = f" [{status}]" if status else ""
                print(f"{type(self).__name__}{status_str}")
                for key, expr in exprs.items():
                    print(f"  ${lhs[key]} = {_rhs_latex(key, expr)}$")

        return exprs

    def check_functions(self, n_pts=300, ax=None, show=False):
        """
        Compare analytic overrides of Gamma_hat and R_Gamma to the
        FFT-based numerical implementations.

        Checks whether the subclass has provided analytic overrides for
        ``Gamma_hat(omega)`` and/or ``R_Gamma(lag)``.  For each override
        found, evaluates both the analytic and numerical versions on a fine
        grid, plots the comparison, and computes the RMSE and max absolute
        error (both normalized by the numerical peak).

        Parameters
        ----------
        n_pts : int, optional
            Number of evaluation points on each grid.  Default 300.
        ax : matplotlib Axes or array of Axes, optional
            Axes to plot on.  If None, a new figure is created sized to
            the number of overridden functions.
        show : bool, optional
            If True, call ``plt.show()`` after plotting.  Default False.

        Returns
        -------
        errors : dict
            Keys are the names of overridden functions (``"Gamma_hat"``
            and/or ``"R_Gamma"``).  Each value is a dict with:

            - ``"rmse"``     : root-mean-square error (normalized by peak)
            - ``"max_err"``  : maximum absolute error (normalized by peak)
        """
        import matplotlib.pyplot as plt

        # Detect which methods the subclass has overridden
        has_analytic = {
            "Gamma_hat": type(self).Gamma_hat is not EnvelopeFunction.Gamma_hat,
            "R_Gamma":   type(self).R_Gamma   is not EnvelopeFunction.R_Gamma,
        }
        overridden = [name for name, flag in has_analytic.items() if flag]

        if not overridden:
            print(f"{type(self).__name__}: no analytic overrides found for "
                  "Gamma_hat or R_Gamma — nothing to check.")
            return {}

        # Ensure the numerical grids are built before we temporarily bypass them
        self._ensure_numerical_grids()

        # Evaluation grids
        lag_max   = self.kernel_support()
        omega_max = 2.0 * np.pi / self.tau_spot * 5.0
        grids = {
            "Gamma_hat": np.linspace(0.0, omega_max, n_pts),
            "R_Gamma":   np.linspace(0.0, lag_max,   n_pts),
        }

        # Numerical baselines (always from the FFT grids, bypassing any override)
        def _numerical_R_Gamma(lag_arr):
            lag_abs = np.abs(lag_arr)
            return np.interp(lag_abs, self._num_R_lag_grid, self._num_R_vals)

        def _numerical_Gamma_hat(omega_arr):
            return np.interp(
                np.abs(omega_arr),
                self._num_Gh_omega_grid,
                np.sqrt(self._num_Gh_sq_vals),
            )

        numerical_funcs = {
            "Gamma_hat": _numerical_Gamma_hat,
            "R_Gamma":   _numerical_R_Gamma,
        }
        analytic_funcs = {
            "Gamma_hat": lambda x: np.asarray(self.Gamma_hat(jnp.array(x))),
            "R_Gamma":   lambda x: np.asarray(self.R_Gamma(jnp.array(x))),
        }
        xlabels = {
            "Gamma_hat": r"$\omega$ [rad/day]",
            "R_Gamma":   r"Lag $\tau$ [days]",
        }
        ylabels = {
            "Gamma_hat": r"$\hat{\Gamma}(\omega)$",
            "R_Gamma":   r"$R_{\Gamma}(\tau)$",
        }

        n_panels = len(overridden)
        if ax is None:
            fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
            if n_panels == 1:
                axes = [axes]
        else:
            axes = np.atleast_1d(ax)

        import time
        errors = {}
        for panel_ax, name in zip(axes, overridden):
            x = grids[name]

            analytic_funcs[name](x[:10])  # warmup: trigger JAX JIT compilation

            t0 = time.perf_counter()
            y_analytic  = analytic_funcs[name](x)
            t_analytic  = time.perf_counter() - t0

            t0 = time.perf_counter()
            y_numerical = numerical_funcs[name](x)
            t_numerical = time.perf_counter() - t0

            peak = float(np.max(np.abs(y_numerical)))
            if peak == 0.0:
                peak = 1.0
            residuals   = y_analytic - y_numerical
            rmse        = float(np.sqrt(np.mean(residuals ** 2))) / peak
            max_err     = float(np.max(np.abs(residuals))) / peak
            errors[name] = {"rmse": rmse, "max_err": max_err}
            print(f"{name}  RMSE = {rmse:.2e},  max err = {max_err:.2e}")
            print(f"  analytic  time = {t_analytic * 1e3:.3f} ms")
            print(f"  numerical time = {t_numerical * 1e3:.3f} ms")

            panel_ax.plot(x, y_numerical, color="k",  lw=2,   label="Numerical (FFT)")
            panel_ax.plot(x, y_analytic,  color="r", lw=1.5, ls="--", label="Analytic (user)")
            panel_ax.set_xlabel(xlabels[name], fontsize=13)
            panel_ax.set_ylabel(ylabels[name], fontsize=13)
            panel_ax.set_title(
                f"{name}  —  RMSE = {rmse:.2e},  max err = {max_err:.2e}",
                fontsize=12,
            )
            panel_ax.legend(fontsize=11)

        if n_panels > 0 and ax is None:
            fig.suptitle(f"{type(self).__name__}.check_functions()", fontsize=13)
            fig.tight_layout()

        if show:
            plt.show()

        return errors


# ── Concrete implementations ────────────────────────────────────────────────

class TrapezoidSymmetricEnvelope(EnvelopeFunction):
    """
    Symmetric trapezoidal envelope.

    Shape: linear rise over tau_spot, plateau of lspot, linear decay over tau_spot.

    Parameters
    ----------
    lspot : float
        Plateau duration [days].
    tau_spot : float
        Rise/decay timescale [days].
    """

    def __init__(self, lspot: float, tau_spot: float):
        self._lspot = float(lspot)
        self._tau_spot = float(tau_spot)

    @property
    def tau_spot(self) -> float:
        return self._tau_spot

    @property
    def lspot(self) -> float:
        return self._lspot

    @property
    def param_dict(self) -> dict:
        return {"lspot": self._lspot, "tau_spot": self._tau_spot}

    def Gamma(self, t):
        t = jnp.asarray(t, dtype=float)
        half = self._lspot / 2.0
        tau_spot = self._tau_spot
        return jnp.where(
            t < -(half + tau_spot), 0.0,
            jnp.where(
                t < -half, (t + half + tau_spot) / tau_spot,
                jnp.where(
                    t <= half, 1.0,
                    jnp.where(
                        t < half + tau_spot, (half + tau_spot - t) / tau_spot,
                        0.0))))

    def Gamma_hat(self, omega):
        return _Gamma_hat(jnp.asarray(omega, dtype=float), self._lspot, self._tau_spot)

    def Gamma_hat_sq(self, omega):
        gh = _Gamma_hat(jnp.asarray(omega, dtype=float), self._lspot, self._tau_spot)
        return gh ** 2

    def R_Gamma(self, lag):
        return _R_Gamma_symmetric(jnp.asarray(lag), self._lspot, self._tau_spot)

    def kernel_support(self) -> float:
        return self._lspot + 2.0 * self._tau_spot

    def sympy_Gamma(self):
        import sympy as sp
        t    = sp.Symbol('t', real=True)
        ell  = sp.Symbol(r'\ell', positive=True)
        tau  = sp.Symbol(r'\tau_{\rm spot}', positive=True)
        half = ell / 2
        alpha_norm = sp.Piecewise(
            (sp.Integer(0),                        t < -(half + tau)),
            ((t + half + tau) / tau,               t < -half),
            (sp.Integer(1),                        t <= half),
            ((half + tau - t) / tau,               t < half + tau),
            (sp.Integer(0),                        True),
        )
        return alpha_norm ** 2

    def sympy_Gamma_hat(self):
        import sympy as sp
        omega = sp.Symbol(r'\omega', real=True)
        ell   = sp.Symbol(r'\ell', positive=True)
        tau   = sp.Symbol(r'\tau_{\rm spot}', positive=True)
        return (4 / (tau**2 * omega**3) * (
            tau * omega * sp.cos(omega * ell / 2)
            + sp.sin(omega * ell / 2)
            - sp.sin(omega * ell / 2 + omega * tau)))


class TrapezoidAsymmetricEnvelope(EnvelopeFunction):
    """
    Asymmetric trapezoidal envelope with distinct emergence and decay rates.

    Shape: linear rise over tau_em, plateau of lspot, linear decay over tau_dec.

    Parameters
    ----------
    lspot : float
        Plateau duration [days].
    tau_em : float
        Emergence timescale [days].
    tau_dec : float
        Decay timescale [days].
    """

    def __init__(self, lspot: float, tau_em: float, tau_dec: float):
        self._lspot = float(lspot)
        self._tau_em = float(tau_em)
        self._tau_dec = float(tau_dec)
        # Precompute numerical grids once (base-class defaults use these)
        self._ensure_numerical_grids()

    @property
    def tau_spot(self) -> float:
        return (self._tau_em + self._tau_dec) / 2.0

    @property
    def tau_em(self) -> float:
        return self._tau_em

    @property
    def tau_dec(self) -> float:
        return self._tau_dec

    @property
    def lspot(self) -> float:
        return self._lspot

    @property
    def param_dict(self) -> dict:
        return {
            "lspot": self._lspot,
            "tau_em": self._tau_em,
            "tau_dec": self._tau_dec,
        }

    def Gamma(self, t):
        t = jnp.asarray(t, dtype=float)
        half = self._lspot / 2.0
        te, td = self._tau_em, self._tau_dec
        return jnp.where(
            t < -(half + te), 0.0,
            jnp.where(
                t < -half, (t + half + te) / te,
                jnp.where(
                    t <= half, 1.0,
                    jnp.where(
                        t < half + td, (half + td - t) / td,
                        0.0))))

    def R_Gamma(self, lag):
        te = min(self._tau_em, self._tau_dec)
        td = max(self._tau_em, self._tau_dec)
        return _R_Gamma_asymmetric(jnp.asarray(lag), self._lspot, te, td)

    def kernel_support(self) -> float:
        return self._lspot + self._tau_em + self._tau_dec

    def sympy_Gamma(self):
        import sympy as sp
        t    = sp.Symbol('t', real=True)
        ell  = sp.Symbol(r'\ell', positive=True)
        te   = sp.Symbol(r'\tau_{\rm em}', positive=True)
        td   = sp.Symbol(r'\tau_{\rm dec}', positive=True)
        half = ell / 2
        alpha_norm = sp.Piecewise(
            (sp.Integer(0),              t < -(half + te)),
            ((t + half + te) / te,       t < -half),
            (sp.Integer(1),              t <= half),
            ((half + td - t) / td,       t < half + td),
            (sp.Integer(0),              True),
        )
        return alpha_norm ** 2


class SkewedGaussianEnvelope(EnvelopeFunction):
    """
    Skew-normal (Baranyi et al. 2021) envelope.

    Implements Eq. (1) of Baranyi et al. (2021) A&A 653, A59:
        Gamma(t) ∝ exp(-t²/(2σ²)) · (1 + erf(n·t / (σ·√2)))

    Parameters
    ----------
    sigma_sn : float
        Scale parameter [days].
    n_sn : float
        Skewness (dimensionless). n_sn < 0: rapid rise / slow decay;
        n_sn > 0: slow rise / rapid decay; n_sn = 0: Gaussian.
    lspot : float, optional
        Unused (required by base schema); set to 0 (default).
    """

    def __init__(self, sigma_sn: float, n_sn: float, lspot: float = 0.0):
        self._sigma_sn = float(sigma_sn)
        self._n_sn = float(n_sn)
        self._lspot = float(lspot)

        _env_func = _skew_normal_envelope_func(sigma_sn, n_sn)

        # Precompute R_Gamma and |Gamma_hat|² on fine grids for interpolation
        self._R_lag_grid, self._R_vals = compute_R_Gamma_numerical(
            _env_func, tau_ref=sigma_sn)
        self._Gh_omega_grid, self._Gh_sq_vals = _compute_Gamma_hat_sq_numerical(
            _env_func, tau_ref=sigma_sn)

        # Also precompute Gamma itself on a t-grid for JAX-traceable Gamma(t)
        T = 12.0 * sigma_sn
        t_grid_np = np.linspace(-T, T, 4096)
        gamma_np = _env_func(t_grid_np)
        self._t_grid = jnp.array(t_grid_np)
        self._Gamma_vals = jnp.array(gamma_np)

    @property
    def tau_spot(self) -> float:
        return self._sigma_sn

    @property
    def sigma_sn(self) -> float:
        return self._sigma_sn

    @property
    def n_sn(self) -> float:
        return self._n_sn

    @property
    def lspot(self) -> float:
        return self._lspot

    @property
    def param_dict(self) -> dict:
        return {
            "sigma_sn": self._sigma_sn,
            "n_sn": self._n_sn,
            "lspot": self._lspot,
        }

    def Gamma(self, t):
        """JAX-traceable Gamma(t) via interpolation from precomputed grid."""
        t = jnp.asarray(t, dtype=float)
        return jnp.interp(t, self._t_grid, self._Gamma_vals)

    def Gamma_hat(self, omega):
        omega = jnp.asarray(omega, dtype=float)
        return jnp.interp(
            jnp.abs(omega), self._Gh_omega_grid,
            jnp.sqrt(self._Gh_sq_vals))

    def Gamma_hat_sq(self, omega):
        omega = jnp.asarray(omega, dtype=float)
        return jnp.interp(jnp.abs(omega), self._Gh_omega_grid, self._Gh_sq_vals)

    def R_Gamma(self, lag):
        lag_abs = jnp.abs(jnp.asarray(lag, dtype=float).ravel())
        return jnp.interp(lag_abs, self._R_lag_grid, self._R_vals)

    def kernel_support(self) -> float:
        return 12.0 * self._sigma_sn


class ExponentialEnvelope(EnvelopeFunction):
    """
    Bilateral exponential (double-sided) envelope.

    Gamma(t) = exp(-|t| / tau_spot)

    This gives a spot that is at peak at t = 0 and decays symmetrically
    with characteristic timescale tau_spot.  There is no plateau (lspot = 0).

    Analytical results:
      Gamma_hat(omega)  = 2*tau_spot / (1 + (omega*tau_spot)²)   [Lorentzian]
      R_Gamma(lag)      = (tau_spot + |lag|) * exp(-|lag| / tau_spot)

    Parameters
    ----------
    tau_spot : float
        Decay timescale [days].
    """

    def __init__(self, tau_spot: float):
        self._tau_spot = float(tau_spot)

    @property
    def tau_spot(self) -> float:
        return self._tau_spot

    @property
    def lspot(self) -> float:
        return 0.0

    @property
    def param_dict(self) -> dict:
        return {"tau_spot": self._tau_spot}

    def Gamma(self, t):
        t = jnp.asarray(t, dtype=float)
        return jnp.exp(-jnp.abs(t) / self._tau_spot)

    def Gamma_hat(self, omega):
        """|FT[Gamma]| = 2*tau_spot / (1 + (omega*tau_spot)²) (Lorentzian)."""
        omega = jnp.asarray(omega, dtype=float)
        return 2.0 * self._tau_spot / (1.0 + (omega * self._tau_spot) ** 2)

    def Gamma_hat_sq(self, omega):
        gh = self.Gamma_hat(omega)
        return gh ** 2

    def R_Gamma(self, lag):
        """R_Gamma(lag) = (tau_spot + |lag|) * exp(-|lag| / tau_spot)."""
        abs_lag = jnp.abs(jnp.asarray(lag, dtype=float))
        return (self._tau_spot + abs_lag) * jnp.exp(-abs_lag / self._tau_spot)

    def kernel_support(self) -> float:
        return 6.0 * self._tau_spot

    def sympy_Gamma(self):
        import sympy as sp
        t   = sp.Symbol('t', real=True)
        tau = sp.Symbol(r'\tau_{\rm spot}', positive=True)
        return sp.exp(-sp.Abs(t) / tau)

    def sympy_Gamma_hat(self):
        import sympy as sp
        omega = sp.Symbol(r'\omega', real=True)
        tau   = sp.Symbol(r'\tau_{\rm spot}', positive=True)
        return 2 * tau / (1 + (omega * tau)**2)

    def sympy_R_Gamma(self):
        import sympy as sp
        tau_lag = sp.Symbol(r'\tau', real=True)
        tau     = sp.Symbol(r'\tau_{\rm spot}', positive=True)
        return (tau + sp.Abs(tau_lag)) * sp.exp(-sp.Abs(tau_lag) / tau)
