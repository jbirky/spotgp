import numpy as np

__all__ = ["AnalyticKernel"]

# Required keys for all modes (alpha_max only needed for physical-param amplitude)
_REQUIRED_KEYS = {"peq", "kappa", "inc", "lspot", "tau"}

# Two valid modes for specifying amplitude:
#   Mode 1: provide sigma_k directly
#   Mode 2: provide nspot and fspot (sigma_k computed from Eq. kernel_Nspot)
_AMPLITUDE_KEYS_SIGMA = {"sigma_k"}
_AMPLITUDE_KEYS_PHYSICAL = {"nspot", "fspot", "alpha_max"}


def _Gamma_hat(omega, ell, tau):
    """
    Fourier transform of the normalized squared envelope Gamma(t) = alpha^2(t)/alpha_max^2.
    Vectorized with NumPy.

    The alpha_max dependence is absorbed into sigma_k^2.

    Uses safe_w = max(|omega|, eps) to avoid 1/w^3 singularity at omega=0.
    """
    omega = np.asarray(omega, dtype=float)
    safe_w = np.where(np.abs(omega) > 1e-14, omega, 1.0)

    nz_result = (4 / (tau**2 * safe_w**3) *
                 (tau * safe_w * np.cos(safe_w * ell / 2)
                  + np.sin(safe_w * ell / 2)
                  - np.sin(safe_w * ell / 2 + safe_w * tau)))

    zero_result = ell + 2 * tau / 3

    return np.where(np.abs(omega) > 1e-14, nz_result, zero_result)


def _R_Gamma(lag, ell, tau_s):
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
    t = np.abs(np.asarray(lag, dtype=float).ravel())

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
    R = np.where(t <= tau_s, R1,
        np.where(t <= ell, R2,
        np.where(t <= ell + tau_s, R3,
        np.where(t <= ell + 2*tau_s, R4,
                  0.0))))

    return R


def _cn_general(n, inc, phi):
    """
    Fourier coefficient c_n of the visibility function.
    Scalar inputs (n is an integer, inc and phi are floats).
    """
    a0 = np.cos(inc) * np.sin(phi)
    a1 = np.sin(inc) * np.cos(phi)

    if abs(a1) < 1e-15:
        # pole/equator case
        if n == 0:
            return a0
        else:
            return 0.0

    ratio = -a0 / a1

    if ratio >= 1.0:
        # never visible
        return 0.0
    elif ratio <= -1.0:
        theta_vis = np.pi
    else:
        theta_vis = np.arccos(np.clip(ratio, -1.0 + 1e-7, 1.0 - 1e-7))

    if n == 0:
        return (a0 * theta_vis + a1 * np.sin(theta_vis)) / np.pi
    elif n == 1:
        return (a0 * np.sin(theta_vis)
                + a1 / 2 * (theta_vis + np.sin(theta_vis) * np.cos(theta_vis))) / np.pi
    else:
        n_f = float(n)
        nm1 = n_f - 1
        np1 = n_f + 1
        term1 = a0 * np.sin(n_f * theta_vis) / n_f
        term2 = a1 / 2 * (np.sin(nm1 * theta_vis) / nm1
                           + np.sin(np1 * theta_vis) / np1)
        return (term1 + term2) / np.pi


def _cn_squared_coefficients(inc, phi, n_harmonics=2):
    """
    Compute |c_n|^2 for n = 0, 1, ..., n_harmonics using a Python loop.
    """
    cn_vals = np.array([_cn_general(n, inc, phi) for n in range(n_harmonics + 1)])
    return cn_vals**2


def _gauss_legendre_grid(n, a, b):
    """
    Compute Gauss-Legendre nodes and weights on [a, b].

    Parameters
    ----------
    n : int
        Number of quadrature points.
    a, b : float
        Integration interval.

    Returns
    -------
    nodes : ndarray, shape (n,)
    weights : ndarray, shape (n,)
    """
    nodes_ref, weights_ref = np.polynomial.legendre.leggauss(n)
    # Map from [-1, 1] to [a, b]
    nodes = 0.5 * (b - a) * nodes_ref + 0.5 * (a + b)
    weights = 0.5 * (b - a) * weights_ref
    return nodes, weights


class AnalyticKernel:
    """
    NumPy analytic GP kernel for stellar rotation variability.

    Same interface as the JAX version but uses NumPy loops for latitude
    integration instead of jax.lax.scan / vmap.

    Parameters
    ----------
    hparam : dict
        Hyperparameters. Required keys: peq, kappa, inc, lspot, tau.
        For the kernel amplitude, provide EITHER:
          - sigma_k : overall amplitude prefactor, OR
          - nspot + fspot + alpha_max : number of spots, spot contrast,
            and max spot radius, from which sigma_k is computed as
            sqrt(N_spot) * (1 - f_spot) * alpha_max^2 / pi.
    n_harmonics : int
        Number of harmonics to include (default 2).
    n_lat : int
        Number of latitude grid points (default 64).
    lat_range : tuple
        (min, max) latitude in radians (default (-pi/2, pi/2)).
    quadrature : str
        Latitude integration method: "trapezoid" or "gauss-legendre"
        (default "trapezoid").
    """

    def __init__(self, hparam, n_harmonics=3, n_lat=64,
                 lat_range=(-np.pi/2, np.pi/2), quadrature="trapezoid"):

        if not isinstance(hparam, dict):
            raise TypeError("hparam must be a dict")

        missing = _REQUIRED_KEYS - set(hparam.keys())
        if missing:
            raise ValueError(f"hparam dict is missing required keys: {missing}")

        has_sigma = "sigma_k" in hparam
        has_physical = "nspot" in hparam and "fspot" in hparam and "alpha_max" in hparam

        if not has_sigma and not has_physical:
            raise ValueError(
                "hparam must contain either 'sigma_k' or all of "
                "'nspot', 'fspot', and 'alpha_max'")

        self.hparam = dict(hparam)

        self.peq = self.hparam["peq"]
        self.kappa = self.hparam["kappa"]
        self.inc = self.hparam["inc"]
        self.lspot = self.hparam["lspot"]
        self.tau = self.hparam["tau"]

        if has_sigma:
            self.sigma_k = self.hparam["sigma_k"]
        else:
            nspot = self.hparam["nspot"]
            fspot = self.hparam["fspot"]
            alpha_max = self.hparam["alpha_max"]
            self.sigma_k = np.sqrt(nspot) * (1 - fspot) * alpha_max**2 / np.pi
            self.hparam["sigma_k"] = self.sigma_k

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
        return 2 * np.pi * (1 - self.kappa * np.sin(phi)**2) / self.peq

    def R_Gamma(self, lag):
        """Autocorrelation of the squared envelope (closed-form piecewise polynomial)."""
        return _R_Gamma(np.asarray(lag), self.lspot, self.tau)

    def cn_squared(self, phi):
        """Squared Fourier coefficients at latitude phi."""
        return _cn_squared_coefficients(self.inc, phi, self.n_harmonics)

    def kernel_single_latitude(self, lag, phi):
        """Single-spot kernel at a fixed latitude."""
        lag = np.asarray(lag, dtype=float).ravel()
        R = self.R_Gamma(lag)
        cn_sq = self.cn_squared(phi)
        w0 = self.omega0(phi)

        ns = np.arange(1, len(cn_sq))
        cosine_terms = np.sum(cn_sq[1:] * np.cos(ns * w0 * lag[:, None]), axis=1)
        cosine_sum = cn_sq[0] + 2 * cosine_terms

        return R * cosine_sum

    def kernel(self, lag, lat_dist=None):
        """
        Full GP kernel averaged over latitude.

        Uses a Python for loop over latitude points (sequential accumulation),
        keeping memory usage at O(M) instead of O(n_lat * M).

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
        lag = np.asarray(lag, dtype=float)
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
            ns = np.arange(1, n_harmonics + 1)
            cosine_terms = np.sum(cn_sq[1:] * np.cos(ns * w0 * lag_flat[:, None]), axis=1)
            return cn_sq[0] + 2 * cosine_terms

        if self.quadrature == "gauss-legendre":
            phi_grid = self._quad_nodes
            quad_weights = self._quad_weights
            user_weights = np.array([lat_dist(float(p)) for p in phi_grid])
            weights = user_weights * quad_weights
            norm = np.sum(weights)
        else:
            phi_min, phi_max = self.lat_range
            phi_grid = np.linspace(phi_min, phi_max, self.n_lat)
            dphi = phi_grid[1] - phi_grid[0]
            user_weights = np.array([lat_dist(float(p)) for p in phi_grid])
            weights = user_weights * dphi
            norm = np.trapezoid(user_weights, phi_grid)

        # Accumulate weighted contributions one latitude at a time
        K = np.zeros_like(lag_flat)
        for i, phi in enumerate(phi_grid):
            K += weights[i] * _lat_contribution(float(phi))
        K = K / norm

        K = R * K * self.sigma_k**2

        return K.reshape(orig_shape)

    def kernel_solid_body(self, lag, lat_dist=None):
        """Kernel for solid-body rotation (kappa=0)."""
        lag = np.asarray(lag, dtype=float)

        if lat_dist is None:
            lat_dist = lambda phi: 1.0

        # Average |c_n|^2 over latitude
        if self.quadrature == "gauss-legendre":
            phi_grid = self._quad_nodes
            quad_weights = self._quad_weights

            all_cn_sq = np.array([
                _cn_squared_coefficients(self.inc, float(phi), self.n_harmonics)
                for phi in phi_grid
            ])

            user_weights = np.array([lat_dist(float(p)) for p in phi_grid])
            norm = np.sum(user_weights * quad_weights)

            cn_sq_avg = np.sum(
                user_weights[:, None] * quad_weights[:, None] * all_cn_sq, axis=0)
            cn_sq_avg = cn_sq_avg / norm
        else:
            phi_min, phi_max = self.lat_range
            phi_grid = np.linspace(phi_min, phi_max, self.n_lat)

            user_weights = np.array([lat_dist(float(p)) for p in phi_grid])
            norm = np.trapezoid(user_weights, phi_grid)

            all_cn_sq = np.array([
                _cn_squared_coefficients(self.inc, float(phi), self.n_harmonics)
                for phi in phi_grid
            ])

            cn_sq_avg = np.sum(user_weights[:, None] * all_cn_sq, axis=0)
            cn_sq_avg = cn_sq_avg * (phi_grid[1] - phi_grid[0]) / norm

        w0 = 2 * np.pi / self.peq
        R = self.R_Gamma(lag)

        ns = np.arange(1, len(cn_sq_avg))
        cosine_terms = np.sum(cn_sq_avg[1:] * np.cos(ns * w0 * lag[:, None]), axis=1)
        cosine_sum = cn_sq_avg[0] + 2 * cosine_terms

        return R * cosine_sum * self.sigma_k**2

    def compute_psd(self, omega, lat_dist=None):
        """
        Analytic power spectral density, vectorized with NumPy.

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
        omega = np.asarray(omega, dtype=float)

        if lat_dist is None:
            lat_dist = lambda phi: 1.0

        def _psd_at_lat(phi):
            cn_sq = self.cn_squared(phi)
            w0 = self.omega0(phi)

            # n=0 term
            Gh_0 = _Gamma_hat(omega, self.lspot, self.tau)
            contrib = cn_sq[0] * Gh_0**2

            # n>=1 terms
            for n in range(1, len(cn_sq)):
                Gh_plus = _Gamma_hat(omega - n * w0, self.lspot, self.tau)
                Gh_minus = _Gamma_hat(omega + n * w0, self.lspot, self.tau)
                contrib = contrib + cn_sq[n] * (Gh_plus**2 + Gh_minus**2)

            return contrib

        if self.quadrature == "gauss-legendre":
            phi_grid = self._quad_nodes
            quad_weights = self._quad_weights

            all_contribs = np.array([_psd_at_lat(float(phi)) for phi in phi_grid])

            user_weights = np.array([lat_dist(float(p)) for p in phi_grid])
            norm = np.sum(user_weights * quad_weights)

            psd = np.sum(user_weights[:, None] * quad_weights[:, None]
                          * all_contribs, axis=0)
            psd = psd / norm
        else:
            phi_min, phi_max = self.lat_range
            phi_grid = np.linspace(phi_min, phi_max, self.n_lat)
            dphi = phi_grid[1] - phi_grid[0]

            user_weights = np.array([lat_dist(float(p)) for p in phi_grid])
            norm = np.trapezoid(user_weights, phi_grid)

            all_contribs = np.array([_psd_at_lat(float(phi)) for phi in phi_grid])
            psd = np.sum(user_weights[:, None] * all_contribs, axis=0)
            psd = psd * dphi / norm

        psd = psd * self.sigma_k**2

        self.psd_omega = omega
        self.psd_freq = omega / (2 * np.pi)
        self.psd_power = psd

        return self.psd_freq, self.psd_power

    def __call__(self, lag, **kwargs):
        """Evaluate the kernel at the given lags."""
        return self.kernel(lag, **kwargs)
