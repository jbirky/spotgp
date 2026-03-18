import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

try:
    from .params import (
        resolve_hparam,
        BASE_REQUIRED_KEYS as _REQUIRED_KEYS_BASE,
        _REQUIRED_KEYS, _AMPLITUDE_KEYS_SIGMA,
        _AMPLITUDE_KEYS_PHYSICAL_RATE, _AMPLITUDE_KEYS_PHYSICAL,
    )
except ImportError:
    from params import (
        resolve_hparam,
        BASE_REQUIRED_KEYS as _REQUIRED_KEYS_BASE,
        _REQUIRED_KEYS, _AMPLITUDE_KEYS_SIGMA,
        _AMPLITUDE_KEYS_PHYSICAL_RATE, _AMPLITUDE_KEYS_PHYSICAL,
    )

__all__ = ["AnalyticKernel", "compute_R_Gamma_numerical"]


@jax.jit
def _Gamma_hat(omega, ell, tau):
    """
    Fourier transform of the normalized squared envelope Gamma(t) = alpha^2(t)/alpha_max^2.
    Fully vectorized with JAX.

    The alpha_max dependence is absorbed into sigma_k^2.

    Uses safe_w = max(|omega|, eps) to avoid 1/w^3 singularity at omega=0.
    This ensures correct gradients through JAX autodiff (jnp.where alone
    does not prevent NaN gradients from the unused branch).
    """
    omega = jnp.asarray(omega, dtype=float)
    # Replace omega=0 with a safe value to avoid 0/0 in forward pass
    # (the result at omega~0 is overridden by the zero_result branch)
    safe_w = jnp.where(jnp.abs(omega) > 1e-14, omega, 1.0)

    nz_result = (4 / (tau**2 * safe_w**3) *
                 (tau * safe_w * jnp.cos(safe_w * ell / 2)
                  + jnp.sin(safe_w * ell / 2)
                  - jnp.sin(safe_w * ell / 2 + safe_w * tau)))

    zero_result = ell + 2 * tau / 3

    return jnp.where(jnp.abs(omega) > 1e-14, nz_result, zero_result)


@jax.jit
def _R_Gamma_symmetric(lag, ell, tau_s):
    """
    Normalized closed-form autocorrelation of Gamma(t) = alpha^2(t)/alpha_max^2.

    Piecewise degree-5 polynomial derived from direct convolution of the
    normalized squared trapezoidal envelope (see Appendix D, Eq. R_Gamma_closed).
    The alpha_max dependence is absorbed into sigma_k^2.

    Four intervals on [0, ell + 2*tau_s], zero beyond.
    Assumes ell/2 > tau_s (plateau longer than ramp).

    Parameters
    ----------
    lag : array_like
        Time lags (non-negative) [days].
    ell : float
        Spot plateau duration [days].
    tau_s : float
        Rise/decay timescale [days].
    """
    t = jnp.abs(jnp.asarray(lag, dtype=float).ravel())

    # Interval 1: 0 <= t <= tau_s
    R1 = (ell + 2*tau_s/5
           - 4*t**2 / (3*tau_s)
           + 2*t**3 / (3*tau_s**2)
           - t**5 / (15*tau_s**4))

    # Interval 2: tau_s <= t <= ell  (linear)
    R2 = ell + 2*tau_s/3 - t

    # Interval 3: ell <= t <= ell + tau_s  (degree-5 polynomial P3)
    R3 = (t**5 / (30*tau_s**4)
           - (ell + 2*tau_s) * t**4 / (6*tau_s**4)
           + (ell**2 + 4*ell*tau_s + 2*tau_s**2) * t**3 / (3*tau_s**4)
           - ell*(ell**2 + 6*ell*tau_s + 6*tau_s**2) * t**2 / (3*tau_s**4)
           + (ell**4 + 8*ell**3*tau_s + 12*ell**2*tau_s**2
              - 6*tau_s**4) * t / (6*tau_s**4)
           + (-ell**5 - 10*ell**4*tau_s - 20*ell**3*tau_s**2
              + 30*ell*tau_s**4 + 20*tau_s**5) / (30*tau_s**4))

    # Interval 4: ell + tau_s <= t <= ell + 2*tau_s
    R4 = (ell + 2*tau_s - t)**5 / (30*tau_s**4)

    # Select the correct interval for each lag value
    R = jnp.where(t <= tau_s, R1,
        jnp.where(t <= ell, R2,
        jnp.where(t <= ell + tau_s, R3,
        jnp.where(t <= ell + 2*tau_s, R4,
                  0.0))))

    return R


@jax.jit
def _R_Gamma_asymmetric(lag, ell, te, td):
    """
    Closed-form autocorrelation of Gamma(t) for asymmetric trapezoidal envelope.

    Generalizes _R_Gamma_symmetric to distinct emergence (te) and decay (td) timescales.
    Assumes te <= td (enforced by the caller via min/max swap).

    Six intervals on [0, ell + te + td], zero beyond.
    Assumes ell/2 > td (plateau longer than either ramp).

    Parameters
    ----------
    lag : array_like
        Time lags (non-negative) [days].
    ell : float
        Spot plateau duration [days].
    te : float
        Emergence timescale [days] (must be <= td).
    td : float
        Decay timescale [days].
    """
    t = jnp.abs(jnp.asarray(lag, dtype=float).ravel())

    td2 = td**2
    te2 = te**2
    td2te2 = td2 * te2
    ell2 = ell**2
    ell3 = ell**3

    # Interval 1: 0 <= t <= te
    R1 = (ell + (te + td) / 5
          - 2 * (1/te + 1/td) / 3 * t**2
          + (1/te2 + 1/td2) / 3 * t**3
          - (1/te**4 + 1/td**4) / 30 * t**5)

    # Interval 2: te <= t <= td
    R2 = (ell + te/3 + td/5
          - t / 2
          - 2 * t**2 / (3 * td)
          + t**3 / (3 * td2)
          - t**5 / (30 * td**4))

    # Interval 3: td <= t <= ell  (linear)
    R3 = ell + (te + td) / 3 - t

    # Interval 4: ell <= t <= ell + te  (degree-5 polynomial P4)
    R4 = (t**5 / (30 * td2te2)
          - (ell + td + te) * t**4 / (6 * td2te2)
          + (ell2 + 2*ell*td + 2*ell*te + 2*td*te) * t**3 / (3 * td2te2)
          - ell * (ell2 + 3*ell*td + 3*ell*te + 6*td*te) * t**2 / (3 * td2te2)
          + (ell**4 + 4*ell3*td + 4*ell3*te + 12*ell2*td*te
             - 6*td2te2) * t / (6 * td2te2)
          + (-ell**5 - 5*ell**4*td - 5*ell**4*te - 20*ell3*td*te
             + 30*ell*td2te2 + 10*td**3*te2 + 10*td2*te**3) / (30 * td2te2))

    # Interval 5: ell + te <= t <= ell + td  (degree-3 polynomial P5)
    R5 = (-t**3 / (3 * td2)
          + (ell + td + te/3) * t**2 / td2
          - (6*ell2 + 12*ell*td + 4*ell*te + 6*td2 + 4*td*te + te2) * t / (6 * td2)
          + (ell3/3 + ell2*td + ell2*te/3 + ell*td2 + 2*ell*td*te/3
             + ell*te2/6 + td**3/3 + td2*te/3 + td*te2/6 + te**3/30) / td2)

    # Interval 6: ell + td <= t <= ell + te + td
    D = ell + te + td - t
    R6 = D**5 / (30 * td2te2)

    # Select the correct interval for each lag value
    R = jnp.where(t <= te, R1,
        jnp.where(t <= td, R2,
        jnp.where(t <= ell, R3,
        jnp.where(t <= ell + te, R4,
        jnp.where(t <= ell + td, R5,
        jnp.where(t <= ell + te + td, R6,
                  0.0))))))

    return R


def _safe_arccos(x):
    """arccos that is safe for autodiff at x = +-1.
    Clamps input away from +-1 so that the gradient -1/sqrt(1-x^2)
    remains finite.
    """
    return jnp.arccos(jnp.clip(x, -1.0 + 1e-7, 1.0 - 1e-7))


@jax.jit
def _cn_general_jax(n, inc, phi):
    """
    Fourier coefficient c_n of the visibility function.
    JAX-compatible (no Python control flow on traced values).
    Uses safe_arccos to ensure finite gradients everywhere.
    """
    a0 = jnp.cos(inc) * jnp.sin(phi)
    a1 = jnp.sin(inc) * jnp.cos(phi)

    # Safe division: replace a1~0 with 1.0 to avoid 0/0
    safe_a1 = jnp.where(jnp.abs(a1) < 1e-15, 1.0, a1)
    ratio = -a0 / safe_a1

    # Determine visibility angle
    always_visible = ratio <= -1.0
    never_visible = ratio >= 1.0
    tiny_a1 = jnp.abs(a1) < 1e-15

    # Use safe_arccos for gradient-safe computation
    theta_vis = jnp.where(
        tiny_a1, 0.0,
        jnp.where(always_visible, jnp.pi,
                  jnp.where(never_visible, 0.0,
                            _safe_arccos(ratio))))

    # n == 0 case
    c0 = jnp.where(tiny_a1, a0,
                   (a0 * theta_vis + a1 * jnp.sin(theta_vis)) / jnp.pi)
    c0 = jnp.where(never_visible & ~tiny_a1, 0.0, c0)

    # n == 1 case
    c1 = (a0 * jnp.sin(theta_vis)
          + a1 / 2 * (theta_vis + jnp.sin(theta_vis) * jnp.cos(theta_vis))) / jnp.pi
    c1 = jnp.where(tiny_a1 | never_visible, 0.0, c1)

    # general n >= 2 case
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

    # Select based on n
    result = jnp.where(n == 0, c0, jnp.where(n == 1, c1, cn_general))
    return result


def _cn_squared_coefficients_jax(inc, phi, n_harmonics=2):
    """
    Compute |c_n|^2 for n = 0, 1, ..., n_harmonics using JAX.
    Vectorized over all harmonics at once.
    """
    ns = jnp.arange(n_harmonics + 1)
    # vmap over harmonic number
    cn_vals = jax.vmap(lambda n: _cn_general_jax(n, inc, phi))(ns)
    return cn_vals**2


def _gauss_legendre_grid(n, a, b):
    """
    Compute Gauss-Legendre nodes and weights on [a, b].

    Uses numpy for the root-finding (done once at init time),
    then stores as JAX arrays.

    Parameters
    ----------
    n : int
        Number of quadrature points.
    a, b : float
        Integration interval.

    Returns
    -------
    nodes : jnp.ndarray, shape (n,)
    weights : jnp.ndarray, shape (n,)
    """
    nodes_ref, weights_ref = np.polynomial.legendre.leggauss(n)
    # Map from [-1, 1] to [a, b]
    nodes = 0.5 * (b - a) * nodes_ref + 0.5 * (a + b)
    weights = 0.5 * (b - a) * weights_ref
    return jnp.array(nodes), jnp.array(weights)


def compute_R_Gamma_numerical(envelope_func, tau_ref, n_grid=4096, extent=12.0):
    """
    Compute R_Gamma(lag) = ∫ Gamma(t) · Gamma(t + lag) dt numerically via FFT.

    Use this whenever no closed-form autocorrelation is available for a
    custom envelope shape.  The result is stored as a pair of JAX arrays
    and can be evaluated at arbitrary lags with ``jnp.interp``::

        lag_grid, R_vals = compute_R_Gamma_numerical(my_envelope, tau_ref=tau)
        R_at_lags = jnp.interp(jnp.abs(lags), lag_grid, R_vals)

    Parameters
    ----------
    envelope_func : callable
        ``f(t: np.ndarray) -> np.ndarray``
        The normalized envelope Gamma(t) ≥ 0, evaluated on a 1-D numpy
        array of time values [days].  Negative values are clipped to zero.
        The envelope should be negligibly small at ±extent·tau_ref so
        that the FFT does not suffer from wrap-around artefacts.
    tau_ref : float
        Reference timescale [days] that sets the grid extent and resolution.
        A good choice is the half-width at half-maximum or the decay
        timescale of the envelope.
    n_grid : int, optional
        Number of time-grid points (default 4096).  Larger values give
        finer lag resolution and a larger maximum representable lag.
    extent : float, optional
        Grid half-width in units of tau_ref (default 12.0).  Increase if
        the envelope has a heavy tail that is non-negligible at
        ±extent·tau_ref.

    Returns
    -------
    lag_grid : jnp.ndarray, shape (n_grid,)
        Non-negative lag values [days], starting at 0.
    R_Gamma_vals : jnp.ndarray, shape (n_grid,)
        R_Gamma at each lag.  R_Gamma is symmetric, so only non-negative
        lags are stored; use ``jnp.abs(lag)`` when interpolating.
    """
    T = float(extent) * float(tau_ref)
    t_np = np.linspace(-T, T, n_grid)
    dt = float(t_np[1] - t_np[0])

    env_np = np.asarray(envelope_func(t_np), dtype=np.float64)
    env_np = np.maximum(env_np, 0.0)  # clip any numerical negatives

    # FFT-based autocorrelation, zero-padded to 2·n_grid to avoid aliasing
    env_fft = np.fft.rfft(env_np, n=2 * n_grid)
    R_vals = np.fft.irfft(np.abs(env_fft) ** 2, n=2 * n_grid)[:n_grid] * dt

    lag_grid = np.arange(n_grid, dtype=np.float64) * dt
    return jnp.array(lag_grid), jnp.array(R_vals)


def _skew_normal_envelope_func(sigma_sn, n_sn):
    """
    Return a callable for the normalized skew-normal envelope.

    Implements Eq. (1) of Baranyi et al. (2021) A&A 653, A59:

        Gamma(t) ∝ exp(-t²/(2σ²)) · (1 + erf(n·t / (σ·√2)))

    normalized so that the peak value equals 1.

    Parameters
    ----------
    sigma_sn : float
        Scale parameter [days].
    n_sn : float
        Skewness parameter (dimensionless).
        n_sn < 0: rapid rise / slow decay.
        n_sn > 0: slow rise / rapid decay.
        n_sn = 0: symmetric Gaussian envelope.

    Returns
    -------
    callable
        f(t: np.ndarray) -> np.ndarray
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


def _compute_Gamma_hat_sq_numerical(envelope_func, tau_ref, n_grid=4096, extent=12.0):
    """
    Precompute |Gamma_hat(ω)|² for a numerical envelope (used by compute_psd).

    Gamma_hat(ω) = ∫ Gamma(t) · exp(-i·ω·t) dt

    Parameters
    ----------
    envelope_func : callable
        Same signature as for compute_R_Gamma_numerical.
    tau_ref, n_grid, extent : same as compute_R_Gamma_numerical.

    Returns
    -------
    omega_grid : jnp.ndarray, shape (n_grid + 1,)
        Non-negative angular frequencies [rad/day].
    Gh_sq_vals : jnp.ndarray, shape (n_grid + 1,)
        |Gamma_hat(ω)|² at each frequency.
    """
    T = float(extent) * float(tau_ref)
    t_np = np.linspace(-T, T, n_grid)
    dt = float(t_np[1] - t_np[0])

    env_np = np.asarray(envelope_func(t_np), dtype=np.float64)
    env_np = np.maximum(env_np, 0.0)

    n_fft = 2 * n_grid
    # Multiply by dt to approximate the continuous FT integral
    env_fft = np.fft.rfft(env_np, n=n_fft) * dt
    Gh_sq = np.abs(env_fft) ** 2

    # Angular frequency grid [rad/day]
    omega_grid = 2.0 * np.pi * np.fft.rfftfreq(n_fft, d=dt)

    return jnp.array(omega_grid), jnp.array(Gh_sq)


class AnalyticKernel:
    """
    JAX-accelerated analytic GP kernel for stellar rotation variability.

    Key optimizations over numpy version:
    - R_Gamma computed via vectorized matrix multiply instead of per-lag loop
    - cn coefficients computed via vmap over harmonics
    - Latitude integration vectorized with vmap
    - Key functions JIT-compiled

    Parameters
    ----------
    hparam : dict
        Hyperparameters. Required keys: peq, kappa, inc, lspot.
        For the envelope shape, provide ONE of:
          - tau                   : symmetric trapezoidal envelope, OR
          - tau_em + tau_dec      : asymmetric trapezoidal envelope, OR
          - sigma_sn + n_sn       : skew-normal envelope (Baranyi et al. 2021
                                    eq. 1); lspot is required by schema but
                                    unused — set to 0.
        For the kernel amplitude, provide EITHER:
          - sigma_k : overall amplitude prefactor, OR
          - nspot_rate + fspot + alpha_max : physical parameterization.
    n_harmonics : int
        Number of harmonics to include (default 2).
    n_lat : int
        Number of latitude grid points (default 64).
    lat_range : tuple
        (min, max) latitude in radians (default (0, pi)).
    quadrature : str
        Latitude integration method: "trapezoid" or "gauss-legendre"
        (default "trapezoid").
    """

    def __init__(self, hparam, n_harmonics=3, n_lat=64,
                 lat_range=(-np.pi/2, np.pi/2), quadrature="trapezoid"):

        # Validate, resolve envelope timescale, and compute sigma_k
        self.hparam = resolve_hparam(hparam)

        self.peq = self.hparam["peq"]
        self.kappa = self.hparam["kappa"]
        self.inc = self.hparam["inc"]
        self.lspot = self.hparam["lspot"]
        self.sigma_k = self.hparam["sigma_k"]

        # Envelope type — detected from the *original* hparam keys
        if "sigma_sn" in hparam and "n_sn" in hparam:
            # ── Skew-normal (Baranyi et al. 2021, eq. 1) ──────────────────
            self.envelope_type = "skew_normal"
            self.sigma_sn = self.hparam["sigma_sn"]
            self.n_sn = self.hparam["n_sn"]
            self.tau = self.hparam["tau"]   # = sigma_sn, injected by resolve_hparam
            self.tau_em = self.tau
            self.tau_dec = self.tau
            self.asymmetric = False
            # Precompute R_Gamma and |Gamma_hat|² on fine grids for fast
            # interpolation at evaluation time.
            _env_func = _skew_normal_envelope_func(self.sigma_sn, self.n_sn)
            self._R_Gamma_lag_grid, self._R_Gamma_vals = (
                compute_R_Gamma_numerical(_env_func, tau_ref=self.sigma_sn))
            self._Gh_sq_omega_grid, self._Gh_sq_vals = (
                _compute_Gamma_hat_sq_numerical(_env_func, tau_ref=self.sigma_sn))

        elif "tau_em" in hparam and "tau_dec" in hparam:
            # ── Asymmetric trapezoid ───────────────────────────────────────
            self.envelope_type = "trapezoid_asymmetric"
            self.asymmetric = True
            self.tau_em = self.hparam["tau_em"]
            self.tau_dec = self.hparam["tau_dec"]
            self.tau = self.hparam["tau"]  # mean, injected by resolve_hparam
            # _R_Gamma_asymmetric requires te <= td
            self._te = min(self.tau_em, self.tau_dec)
            self._td = max(self.tau_em, self.tau_dec)

        else:
            # ── Symmetric trapezoid (default) ─────────────────────────────
            self.envelope_type = "trapezoid_symmetric"
            self.asymmetric = False
            self.tau = self.hparam["tau"]
            self.tau_em = self.tau
            self.tau_dec = self.tau

        self.n_harmonics = n_harmonics
        self.n_lat = n_lat
        self.lat_range = lat_range
        self.quadrature = quadrature

        # Precompute quadrature grid
        if quadrature == "gauss-legendre":
            self._quad_nodes, self._quad_weights = _gauss_legendre_grid(
                n_lat, lat_range[0], lat_range[1])
        elif quadrature == "trapezoid":
            self._quad_nodes = None
            self._quad_weights = None
        else:
            raise ValueError(
                f"Unknown quadrature method: {quadrature!r}. "
                "Use 'trapezoid' or 'gauss-legendre'.")

    def omega0(self, phi):
        """Latitude-dependent rotation frequency."""
        return 2 * jnp.pi * (1 - self.kappa * jnp.sin(phi)**2) / self.peq

    def R_Gamma(self, lag):
        """Autocorrelation of the squared envelope."""
        if self.envelope_type == "skew_normal":
            lag_abs = jnp.abs(jnp.asarray(lag, dtype=float).ravel())
            return jnp.interp(lag_abs, self._R_Gamma_lag_grid, self._R_Gamma_vals)
        if self.asymmetric:
            return _R_Gamma_asymmetric(
                jnp.asarray(lag), self.lspot, self._te, self._td)
        return _R_Gamma_symmetric(jnp.asarray(lag), self.lspot, self.tau)

    def cn_squared(self, phi):
        """Squared Fourier coefficients at latitude phi."""
        return _cn_squared_coefficients_jax(self.inc, phi, self.n_harmonics)

    def kernel_single_latitude(self, lag, phi):
        """Single-spot kernel at a fixed latitude."""
        lag = jnp.asarray(lag, dtype=float).ravel()
        R = self.R_Gamma(lag)
        cn_sq = self.cn_squared(phi)
        w0 = self.omega0(phi)

        ns = jnp.arange(1, len(cn_sq))
        cosine_terms = jnp.sum(cn_sq[1:] * jnp.cos(ns * w0 * lag[:, None]), axis=1)
        cosine_sum = cn_sq[0] + 2 * cosine_terms

        return R * cosine_sum

    def kernel(self, lag, lat_dist=None):
        """
        Full GP kernel averaged over latitude.

        Uses sequential latitude accumulation (``jax.lax.scan``) instead
        of ``jax.vmap`` so that only one lag-sized buffer is live at a
        time, reducing peak memory from O(n_lat * M) to O(M).

        Supports both 1D lag arrays and 2D lag matrices (for covariance).

        Parameters
        ----------
        lag : array_like
            Time lags [days]. Can be 1D or 2D.
        lat_dist : callable or None
            Latitude probability density. If None, uniform.

        Returns
        -------
        K : ndarray
            Kernel values at each lag (same shape as input).
        """
        lag = jnp.asarray(lag, dtype=float)
        orig_shape = lag.shape
        lag_flat = lag.ravel()

        if lat_dist is None:
            lat_dist = lambda phi: 1.0

        # Precompute R_Gamma once (on flat lags)
        R = self.R_Gamma(lag_flat)

        n_harmonics = self.n_harmonics

        # Single-latitude contribution (returns shape (M,))
        def _lat_contribution(phi):
            cn_sq = self.cn_squared(phi)
            w0 = self.omega0(phi)
            ns = jnp.arange(1, n_harmonics + 1)
            cosine_terms = jnp.sum(cn_sq[1:] * jnp.cos(ns * w0 * lag_flat[:, None]), axis=1)
            return cn_sq[0] + 2 * cosine_terms

        if self.quadrature == "gauss-legendre":
            phi_grid = self._quad_nodes
            quad_weights = self._quad_weights
            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            weights = user_weights * quad_weights
            norm = jnp.sum(weights)
        else:
            phi_min, phi_max = self.lat_range
            phi_grid = jnp.linspace(phi_min, phi_max, self.n_lat)
            dphi = phi_grid[1] - phi_grid[0]
            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            weights = user_weights * dphi
            norm = jnp.trapezoid(user_weights, phi_grid)

        # Accumulate weighted contributions one latitude at a time
        def _scan_body(K_acc, idx):
            phi = phi_grid[idx]
            w = weights[idx]
            contrib = _lat_contribution(phi)
            return K_acc + w * contrib, None

        K, _ = jax.lax.scan(_scan_body, jnp.zeros_like(lag_flat),
                             jnp.arange(len(phi_grid)))
        K = K / norm

        K = R * K * self.sigma_k**2

        return np.asarray(K.reshape(orig_shape))

    def kernel_solid_body(self, lag, lat_dist=None):
        """Kernel for solid-body rotation (kappa=0)."""
        lag = jnp.asarray(lag, dtype=float)

        if lat_dist is None:
            lat_dist = lambda phi: 1.0

        # Average |c_n|^2 over latitude via vmap
        if self.quadrature == "gauss-legendre":
            phi_grid = self._quad_nodes
            quad_weights = self._quad_weights

            all_cn_sq = jax.vmap(
                lambda phi: _cn_squared_coefficients_jax(self.inc, phi, self.n_harmonics)
            )(phi_grid)

            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            norm = jnp.sum(user_weights * quad_weights)

            cn_sq_avg = jnp.sum(
                user_weights[:, None] * quad_weights[:, None] * all_cn_sq, axis=0)
            cn_sq_avg = cn_sq_avg / norm
        else:
            phi_min, phi_max = self.lat_range
            phi_grid = jnp.linspace(phi_min, phi_max, self.n_lat)

            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            norm = jnp.trapezoid(user_weights, phi_grid)

            all_cn_sq = jax.vmap(
                lambda phi: _cn_squared_coefficients_jax(self.inc, phi, self.n_harmonics)
            )(phi_grid)

            cn_sq_avg = jnp.sum(user_weights[:, None] * all_cn_sq, axis=0)
            cn_sq_avg = cn_sq_avg * (phi_grid[1] - phi_grid[0]) / norm

        w0 = 2 * jnp.pi / self.peq
        R = self.R_Gamma(lag)

        ns = jnp.arange(1, len(cn_sq_avg))
        cosine_terms = jnp.sum(cn_sq_avg[1:] * jnp.cos(ns * w0 * lag[:, None]), axis=1)
        cosine_sum = cn_sq_avg[0] + 2 * cosine_terms

        return np.asarray(R * cosine_sum * self.sigma_k**2)

    def compute_psd(self, omega, lat_dist=None):
        """
        Analytic power spectral density, vectorized with JAX.

        Parameters
        ----------
        omega : array_like
            Angular frequencies [rad/day].
        lat_dist : callable or None
            Latitude probability density.

        Returns
        -------
        freq : ndarray
            Frequencies in cycles/day.
        power : ndarray
            PSD values.
        """
        omega = jnp.asarray(omega, dtype=float)

        if lat_dist is None:
            lat_dist = lambda phi: 1.0

        if self.envelope_type == "skew_normal":
            # Numerical |Gamma_hat(ω)|² via precomputed grid.
            # |FT(Gamma)|² is symmetric so we interpolate at |ω|.
            _Gh_sq_grid = self._Gh_sq_omega_grid
            _Gh_sq_vals = self._Gh_sq_vals

            def _Gh_sq(om):
                return jnp.interp(jnp.abs(om), _Gh_sq_grid, _Gh_sq_vals)

            def _psd_at_lat(phi):
                cn_sq = self.cn_squared(phi)
                w0 = self.omega0(phi)

                contrib = cn_sq[0] * _Gh_sq(omega)

                def _harmonic_contrib(n):
                    return cn_sq[n] * (_Gh_sq(omega - n * w0) + _Gh_sq(omega + n * w0))

                ns = jnp.arange(1, len(cn_sq))
                harmonic_contribs = jax.vmap(lambda n: _harmonic_contrib(n))(ns)
                return contrib + jnp.sum(harmonic_contribs, axis=0)

        else:
            def _psd_at_lat(phi):
                cn_sq = self.cn_squared(phi)
                w0 = self.omega0(phi)

                # n=0 term
                Gh_0 = _Gamma_hat(omega, self.lspot, self.tau)
                contrib = cn_sq[0] * Gh_0**2

                # n>=1 terms
                def _harmonic_contrib(n):
                    Gh_plus = _Gamma_hat(omega - n * w0, self.lspot, self.tau)
                    Gh_minus = _Gamma_hat(omega + n * w0, self.lspot, self.tau)
                    return cn_sq[n] * (Gh_plus**2 + Gh_minus**2)

                ns = jnp.arange(1, len(cn_sq))
                harmonic_contribs = jax.vmap(lambda n: _harmonic_contrib(n))(ns)
                return contrib + jnp.sum(harmonic_contribs, axis=0)

        if self.quadrature == "gauss-legendre":
            phi_grid = self._quad_nodes
            quad_weights = self._quad_weights

            all_contribs = jax.vmap(_psd_at_lat)(phi_grid)

            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            norm = jnp.sum(user_weights * quad_weights)

            psd = jnp.sum(user_weights[:, None] * quad_weights[:, None]
                          * all_contribs, axis=0)
            psd = psd / norm
        else:
            phi_min, phi_max = self.lat_range
            phi_grid = jnp.linspace(phi_min, phi_max, self.n_lat)
            dphi = phi_grid[1] - phi_grid[0]

            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            norm = jnp.trapezoid(user_weights, phi_grid)

            all_contribs = jax.vmap(_psd_at_lat)(phi_grid)
            psd = jnp.sum(user_weights[:, None] * all_contribs, axis=0)
            psd = psd * dphi / norm

        psd = psd * self.sigma_k**2

        self.psd_omega = np.asarray(omega)
        self.psd_freq = np.asarray(omega / (2 * jnp.pi))
        self.psd_power = np.asarray(psd)

        return self.psd_freq, self.psd_power

    def __call__(self, lag, **kwargs):
        """Evaluate the kernel at the given lags."""
        return self.kernel(lag, **kwargs)
