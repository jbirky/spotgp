import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

__all__ = ["AnalyticKernel"]

HPARAM_KEYS = ["peq", "kappa", "inc", "nspot", "lspot", "tau", "alpha_max", "sigma_k"]


@jax.jit
def _Gamma_hat(omega, ell, tau, alpha_max):
    """
    Fourier transform of the squared trapezoidal envelope Gamma(t) = alpha^2(t).
    Fully vectorized with JAX.

    Uses safe_w = max(|omega|, eps) to avoid 1/w^3 singularity at omega=0.
    This ensures correct gradients through JAX autodiff (jnp.where alone
    does not prevent NaN gradients from the unused branch).
    """
    omega = jnp.asarray(omega, dtype=float)
    # Replace omega=0 with a safe value to avoid 0/0 in forward pass
    # (the result at omega~0 is overridden by the zero_result branch)
    safe_w = jnp.where(jnp.abs(omega) > 1e-14, omega, 1.0)

    nz_result = (4 * alpha_max**2 / (tau**2 * safe_w**3) *
                 (tau * safe_w * jnp.cos(safe_w * ell / 2)
                  + jnp.sin(safe_w * ell / 2)
                  - jnp.sin(safe_w * ell / 2 + safe_w * tau)))

    zero_result = alpha_max**2 * (ell + 2 * tau / 3)

    return jnp.where(jnp.abs(omega) > 1e-14, nz_result, zero_result)


@jax.jit
def _R_Gamma(lag, ell, tau_s, alpha_max):
    """
    Closed-form autocorrelation of Gamma(t) = alpha^2(t).

    Piecewise degree-5 polynomial derived from direct convolution of the
    squared trapezoidal envelope (see Appendix D, Eq. R_Gamma_closed).

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
    alpha_max : float
        Peak angular radius [rad].
    """
    t = jnp.abs(jnp.asarray(lag, dtype=float).ravel())
    a4 = alpha_max**4

    # Interval 1: 0 <= t <= tau_s
    R1 = a4 * (ell + 2*tau_s/5
               - 4*t**2 / (3*tau_s)
               + 2*t**3 / (3*tau_s**2)
               - t**5 / (15*tau_s**4))

    # Interval 2: tau_s <= t <= ell  (linear)
    R2 = a4 * (ell + 2*tau_s/3 - t)

    # Interval 3: ell <= t <= ell + tau_s  (degree-5 polynomial P3)
    R3 = a4 * (t**5 / (30*tau_s**4)
               - (ell + 2*tau_s) * t**4 / (6*tau_s**4)
               + (ell**2 + 4*ell*tau_s + 2*tau_s**2) * t**3 / (3*tau_s**4)
               - ell*(ell**2 + 6*ell*tau_s + 6*tau_s**2) * t**2 / (3*tau_s**4)
               + (ell**4 + 8*ell**3*tau_s + 12*ell**2*tau_s**2
                  - 6*tau_s**4) * t / (6*tau_s**4)
               + (-ell**5 - 10*ell**4*tau_s - 20*ell**3*tau_s**2
                  + 30*ell*tau_s**4 + 20*tau_s**5) / (30*tau_s**4))

    # Interval 4: ell + tau_s <= t <= ell + 2*tau_s
    R4 = a4 * (ell + 2*tau_s - t)**5 / (30*tau_s**4)

    # Select the correct interval for each lag value
    R = jnp.where(t <= tau_s, R1,
        jnp.where(t <= ell, R2,
        jnp.where(t <= ell + tau_s, R3,
        jnp.where(t <= ell + 2*tau_s, R4,
                  0.0))))

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
    hparam : dict or list
        Hyperparameters: peq, kappa, inc, nspot, lspot, tau, alpha_max, sigma_k.
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

    def __init__(self, hparam, n_harmonics=2, n_lat=64,
                 lat_range=(0, np.pi), quadrature="trapezoid"):

        if isinstance(hparam, dict):
            missing = set(HPARAM_KEYS) - set(hparam.keys())
            if missing:
                raise ValueError(f"hparam dict is missing keys: {missing}")
            self.hparam = {k: hparam[k] for k in HPARAM_KEYS}
        else:
            if len(hparam) != len(HPARAM_KEYS):
                raise ValueError(
                    f"hparam list must have {len(HPARAM_KEYS)} elements: {HPARAM_KEYS}")
            self.hparam = dict(zip(HPARAM_KEYS, hparam))

        self.peq = self.hparam["peq"]
        self.kappa = self.hparam["kappa"]
        self.inc = self.hparam["inc"]
        self.nspot = self.hparam["nspot"]
        self.lspot = self.hparam["lspot"]
        self.tau = self.hparam["tau"]
        self.alpha_max = self.hparam["alpha_max"]
        self.sigma_k = self.hparam["sigma_k"]

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
        """Autocorrelation of the squared envelope (closed-form piecewise polynomial)."""
        return _R_Gamma(jnp.asarray(lag), self.lspot, self.tau, self.alpha_max)

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

        Vectorized latitude integration using JAX operations.
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

        # Vectorized latitude contribution
        def _lat_contribution(phi):
            cn_sq = self.cn_squared(phi)
            w0 = self.omega0(phi)
            ns = jnp.arange(1, n_harmonics + 1)
            cosine_terms = jnp.sum(cn_sq[1:] * jnp.cos(ns * w0 * lag_flat[:, None]), axis=1)
            return cn_sq[0] + 2 * cosine_terms

        if self.quadrature == "gauss-legendre":
            phi_grid = self._quad_nodes
            quad_weights = self._quad_weights

            all_contributions = jax.vmap(_lat_contribution)(phi_grid)

            # Apply lat_dist weights
            user_weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            norm = jnp.sum(user_weights * quad_weights)

            K = jnp.sum(user_weights[:, None] * quad_weights[:, None]
                        * all_contributions, axis=0)
            K = K / norm
        else:
            # Trapezoid rule
            phi_min, phi_max = self.lat_range
            phi_grid = jnp.linspace(phi_min, phi_max, self.n_lat)
            dphi = phi_grid[1] - phi_grid[0]

            all_contributions = jax.vmap(_lat_contribution)(phi_grid)

            weights = jnp.array([lat_dist(float(p)) for p in phi_grid])
            norm = jnp.trapezoid(weights, phi_grid)

            K = jnp.sum(weights[:, None] * all_contributions, axis=0)
            K = K * dphi / norm

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

        def _psd_at_lat(phi):
            cn_sq = self.cn_squared(phi)
            w0 = self.omega0(phi)

            # n=0 term
            Gh_0 = _Gamma_hat(omega, self.lspot, self.tau, self.alpha_max)
            contrib = cn_sq[0] * Gh_0**2

            # n>=1 terms
            def _harmonic_contrib(n):
                Gh_plus = _Gamma_hat(omega - n * w0, self.lspot, self.tau, self.alpha_max)
                Gh_minus = _Gamma_hat(omega + n * w0, self.lspot, self.tau, self.alpha_max)
                return cn_sq[n] * (Gh_plus**2 + Gh_minus**2)

            ns = jnp.arange(1, len(cn_sq))
            harmonic_contribs = jax.vmap(lambda n: _harmonic_contrib(n))(ns)
            contrib = contrib + jnp.sum(harmonic_contribs, axis=0)

            return contrib

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
