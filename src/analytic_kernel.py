import numpy as np
from scipy import integrate

__all__ = ["AnalyticKernel"]

HPARAM_KEYS = ["peq", "kappa", "inc", "nspot", "lspot", "tau", "alpha_max"]


def _Gamma_hat(omega, ell, tau, alpha_max):
    """
    Fourier transform of the squared trapezoidal envelope Gamma(t) = alpha^2(t).

    Parameters
    ----------
    omega : array_like
        Angular frequencies.
    ell : float
        Plateau duration of the spot.
    tau : float
        Rise/decay timescale.
    alpha_max : float
        Peak angular radius.

    Returns
    -------
    result : ndarray
        Real-valued FT evaluated at each omega.
    """
    omega = np.asarray(omega, dtype=float)
    result = np.zeros_like(omega)
    nz = np.abs(omega) > 1e-14
    w = omega[nz]
    result[nz] = (4 * alpha_max**2 / (tau**2 * w**3) *
                  (tau * w * np.cos(w * ell / 2)
                   + np.sin(w * ell / 2)
                   - np.sin(w * ell / 2 + w * tau)))
    result[~nz] = alpha_max**2 * (ell + 2 * tau / 3)
    return result


def _R_Gamma(lag, ell, tau, alpha_max, n_omega=4096, omega_max=None):
    """
    Autocorrelation of Gamma(t) = alpha^2(t) via inverse FT of |Gamma_hat|^2.

    R_Gamma(tau) = (1/2pi) int |Gamma_hat(omega)|^2 e^{i omega tau} d omega

    Since Gamma_hat is real, |Gamma_hat|^2 = Gamma_hat^2 and R_Gamma is even.

    Parameters
    ----------
    lag : array_like
        Time lags at which to evaluate the autocorrelation.
    ell : float
        Plateau duration.
    tau : float
        Rise/decay timescale.
    alpha_max : float
        Peak angular radius.
    n_omega : int
        Number of frequency grid points for numerical integration.
    omega_max : float or None
        Upper frequency limit. If None, set automatically from tau.

    Returns
    -------
    R : ndarray
        Autocorrelation values at each lag.
    """
    lag = np.asarray(lag, dtype=float)
    if omega_max is None:
        omega_max = 20 * np.pi / max(tau, 0.1)
    omega_grid = np.linspace(0, omega_max, n_omega)
    Gh = _Gamma_hat(omega_grid, ell, tau, alpha_max)
    Gh_sq = Gh**2

    R = np.empty_like(lag)
    for i, t in enumerate(lag.ravel()):
        # R(t) = (1/2pi) int_{-inf}^{inf} |Gh|^2 e^{i w t} dw
        # Since |Gh|^2 is even: R(t) = (1/pi) int_0^inf |Gh|^2 cos(w t) dw
        integrand = Gh_sq * np.cos(omega_grid * t)
        R.flat[i] = np.trapezoid(integrand, omega_grid) / np.pi
    return R


def _cn_general(n, inc, phi):
    """
    Fourier coefficient c_n of the visibility function Pi(t) = max{cos(beta), 0}
    for general inclination I and spot latitude phi.

    Parameters
    ----------
    n : int
        Harmonic number.
    inc : float
        Stellar inclination in radians.
    phi : float
        Spot latitude in radians.

    Returns
    -------
    cn : float
        The n-th Fourier coefficient (real).
    """
    a0 = np.cos(inc) * np.sin(phi)
    a1 = np.sin(inc) * np.cos(phi)

    # Always-visible spot (a0 >= a1 > 0)
    if np.abs(a1) < 1e-15:
        if n == 0:
            return a0
        else:
            return 0.0

    ratio = -a0 / a1
    if np.abs(ratio) >= 1.0:
        # Spot is always visible (ratio <= -1) or never visible (ratio >= 1)
        if ratio >= 1.0:
            return 0.0
        else:
            # Always visible: theta_vis = pi
            theta_vis = np.pi
    else:
        theta_vis = np.arccos(ratio)

    if n == 0:
        return (a0 * theta_vis + a1 * np.sin(theta_vis)) / np.pi
    elif abs(n) == 1:
        # Limit of general formula as n->1: sin((n-1)θ)/(n-1) -> θ_vis
        return (a0 * np.sin(theta_vis)
                + a1 / 2 * (theta_vis + np.sin(theta_vis) * np.cos(theta_vis))) / np.pi
    else:
        term1 = a0 * np.sin(n * theta_vis) / n
        nm1 = n - 1
        np1 = n + 1
        # Handle n=1 case already covered above, so nm1 != 0 and np1 != 0
        term2 = a1 / 2 * (np.sin(nm1 * theta_vis) / nm1
                          + np.sin(np1 * theta_vis) / np1)
        return (term1 + term2) / np.pi


def _cn_squared_coefficients(inc, phi, n_harmonics=2):
    """
    Compute |c_n|^2 for n = 0, 1, ..., n_harmonics.

    Parameters
    ----------
    inc : float
        Stellar inclination in radians.
    phi : float
        Spot latitude in radians.
    n_harmonics : int
        Maximum harmonic order.

    Returns
    -------
    cn_sq : ndarray of shape (n_harmonics + 1,)
        Squared Fourier coefficients [|c_0|^2, |c_1|^2, ..., |c_N|^2].
    """
    cn_sq = np.empty(n_harmonics + 1)
    for n in range(n_harmonics + 1):
        cn_sq[n] = _cn_general(n, inc, phi) ** 2
    return cn_sq


class AnalyticKernel:
    """
    Analytic GP kernel for stellar rotation variability due to starspots.

    The kernel is (Eq. 33 of the paper):

        K(tau) = (N_spot / pi^2) * R_Gamma(tau)
                 * integral over latitude of
                   [sum_n |c_n(Phi)|^2 cos(n omega_0(Phi) tau)] p(Phi) dPhi

    where the amplitude prefactor N_spot * alpha_max^2 / pi^2 is absorbed
    into R_Gamma via alpha_max.

    Parameters
    ----------
    hparam : dict or list
        Hyperparameters. Keys (or positional order):
        peq, kappa, inc, nspot, lspot, tau, alpha_max.
        - peq: equatorial rotation period [days]
        - kappa: differential rotation shear
        - inc: stellar inclination [radians]
        - nspot: number of spots (amplitude scaling)
        - lspot: spot plateau duration [days]
        - tau: spot rise/decay timescale [days]
        - alpha_max: peak spot angular radius [radians]
    n_harmonics : int
        Number of harmonics to include (default 2).
    n_lat : int
        Number of latitude grid points for numerical integration (default 64).
    lat_range : tuple
        (min, max) latitude in radians (default (0, pi/2) for one hemisphere,
        assuming symmetric spot distribution).
    n_omega : int
        Number of frequency grid points for R_Gamma computation (default 4096).
    """

    def __init__(self, hparam, n_harmonics=2, n_lat=64,
                 lat_range=(0, np.pi), n_omega=4096):

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

        self.n_harmonics = n_harmonics
        self.n_lat = n_lat
        self.lat_range = lat_range
        self.n_omega = n_omega

    def omega0(self, phi):
        """Latitude-dependent rotation frequency."""
        return 2 * np.pi * (1 - self.kappa * np.sin(phi)**2) / self.peq

    def R_Gamma(self, lag):
        """Autocorrelation of the squared envelope Gamma(t) = alpha^2(t)."""
        return _R_Gamma(lag, self.lspot, self.tau, self.alpha_max,
                        n_omega=self.n_omega)

    def cn_squared(self, phi):
        """Squared Fourier coefficients |c_n|^2 at latitude phi."""
        return _cn_squared_coefficients(self.inc, phi, self.n_harmonics)

    def kernel_single_latitude(self, lag, phi):
        """
        Single-spot kernel at a fixed latitude (Eq. 27):

            k(tau) = R_Gamma(tau) * [|c_0|^2 + 2 sum_{n>=1} |c_n|^2 cos(n w0 tau)]
        """
        lag = np.asarray(lag, dtype=float)
        R = self.R_Gamma(lag)
        cn_sq = self.cn_squared(phi)
        w0 = self.omega0(phi)

        cosine_sum = cn_sq[0] + 2 * sum(
            cn_sq[n] * np.cos(n * w0 * lag) for n in range(1, len(cn_sq)))

        return R * cosine_sum

    def kernel(self, lag, lat_dist=None):
        """
        Full GP kernel averaged over latitude (Eq. 33):

            K(tau) = (N_spot / pi^2) * integral [k(tau; Phi) p(Phi)] dPhi

        Parameters
        ----------
        lag : array_like
            Time lags [days].
        lat_dist : callable or None
            Latitude probability density p(Phi). If None, uses a uniform
            distribution (cos(Phi) weighting for isotropic on the sphere).

        Returns
        -------
        K : ndarray
            Kernel values at each lag.
        """
        lag = np.asarray(lag, dtype=float)

        if lat_dist is None:
            lat_dist = lambda phi: 1.0

        phi_min, phi_max = self.lat_range
        phi_grid = np.linspace(phi_min, phi_max, self.n_lat)
        dphi = phi_grid[1] - phi_grid[0]

        # Precompute R_Gamma once (independent of latitude)
        R = self.R_Gamma(lag)

        # Normalisation of p(Phi) over the integration range
        weights = np.array([lat_dist(p) for p in phi_grid])
        norm = np.trapezoid(weights, phi_grid)

        # Integrate the cosine sum over latitude
        K = np.zeros_like(lag)
        for j, phi in enumerate(phi_grid):
            cn_sq = self.cn_squared(phi)
            w0 = self.omega0(phi)

            cosine_sum = cn_sq[0] + 2 * sum(
                cn_sq[n] * np.cos(n * w0 * lag) for n in range(1, len(cn_sq)))

            K += weights[j] * cosine_sum

        K = K * dphi / norm  # trapezoidal integration
        K = R * K * self.nspot / np.pi**2

        return K

    def kernel_solid_body(self, lag, lat_dist=None):
        """
        Kernel for solid-body rotation (kappa=0), Eq. 28:

            K(tau) = (N_spot / pi^2) * R_Gamma(tau)
                     * [<|c_0|^2> + 2 sum <|c_n|^2> cos(n w0 tau)]

        where <|c_n|^2> is the latitude-averaged squared coefficient.
        """
        lag = np.asarray(lag, dtype=float)

        if lat_dist is None:
            lat_dist = lambda phi: 1.0

        phi_min, phi_max = self.lat_range
        phi_grid = np.linspace(phi_min, phi_max, self.n_lat)

        weights = np.array([lat_dist(p) for p in phi_grid])
        norm = np.trapezoid(weights, phi_grid)

        # Average |c_n|^2 over latitude
        cn_sq_avg = np.zeros(self.n_harmonics + 1)
        for j, phi in enumerate(phi_grid):
            cn_sq_avg += weights[j] * self.cn_squared(phi)
        cn_sq_avg = cn_sq_avg * (phi_grid[1] - phi_grid[0]) / norm

        w0 = 2 * np.pi / self.peq
        R = self.R_Gamma(lag)

        cosine_sum = cn_sq_avg[0] + 2 * sum(
            cn_sq_avg[n] * np.cos(n * w0 * lag)
            for n in range(1, len(cn_sq_avg)))

        return R * cosine_sum * self.nspot / np.pi**2

    def compute_psd(self, omega, lat_dist=None):
        """
        Analytic power spectral density (Fourier transform of the kernel).

        The PSD is the latitude-averaged energy spectral density:

            S(omega) = (N_spot / pi^2) * integral over Phi of
                       [sum_n |c_n(Phi)|^2 |Gamma_hat(omega - n*w0(Phi))|^2]
                       * p(Phi) dPhi

        where Gamma_hat is the Fourier transform of the squared spot-size
        envelope, and the sum runs over n = 0, +-1, +-2, ... including
        both positive and negative harmonics.

        Parameters
        ----------
        omega : array_like
            Angular frequencies [rad/day] at which to evaluate the PSD.
        lat_dist : callable or None
            Latitude probability density p(Phi). If None, uses uniform.

        Returns
        -------
        S : ndarray
            Power spectral density at each omega.
        """
        omega = np.asarray(omega, dtype=float)

        if lat_dist is None:
            lat_dist = lambda phi: 1.0

        phi_min, phi_max = self.lat_range
        phi_grid = np.linspace(phi_min, phi_max, self.n_lat)
        dphi = phi_grid[1] - phi_grid[0]

        weights = np.array([lat_dist(p) for p in phi_grid])
        norm = np.trapezoid(weights, phi_grid)

        psd = np.zeros_like(omega)
        for j, phi in enumerate(phi_grid):
            cn_sq = self.cn_squared(phi)
            w0 = self.omega0(phi)

            # n = 0 term: |c_0|^2 |Gamma_hat(omega)|^2
            Gh_0 = _Gamma_hat(omega, self.lspot, self.tau, self.alpha_max)
            contrib = cn_sq[0] * Gh_0**2

            # n >= 1 terms: |c_n|^2 [|Gamma_hat(omega - n*w0)|^2
            #                        + |Gamma_hat(omega + n*w0)|^2]
            for n in range(1, len(cn_sq)):
                Gh_plus = _Gamma_hat(omega - n * w0, self.lspot,
                                     self.tau, self.alpha_max)
                Gh_minus = _Gamma_hat(omega + n * w0, self.lspot,
                                      self.tau, self.alpha_max)
                contrib += cn_sq[n] * (Gh_plus**2 + Gh_minus**2)

            psd += weights[j] * contrib

        psd = psd * dphi / norm
        psd = psd * self.nspot / np.pi**2
        
        self.psd_omega = omega
        self.psd_freq = omega / (2 * np.pi)  # convert to cycles/day
        self.psd_power = psd

        return self.psd_freq, self.psd_power

    def __call__(self, lag, **kwargs):
        """Evaluate the kernel at the given lags."""
        return self.kernel(lag, **kwargs)
