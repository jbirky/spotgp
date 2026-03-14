import numpy as np
import time
from scipy import interpolate

try:
    import matplotlib.pyplot as plt
    from matplotlib import rc
    plt.style.use('classic')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    rc('figure', facecolor='w')
    rc('xtick', labelsize=24)
    rc('ytick', labelsize=24)
except:
    print("Unable to import matplotlib")

try:
    from .starspot import LightcurveModel
    from .psd import compute_psd
except ImportError:
    from starspot import LightcurveModel
    from psd import compute_psd

__all__ = ["NumericalKernel", "generate_sims", "avg_covariance_tlag"]

# Required keys for all modes
_REQUIRED_KEYS = {"peq", "kappa", "inc", "lspot", "tau", "alpha_max"}

# Two valid modes for specifying amplitude:
#   Mode 1: provide sigma_k directly
#   Mode 2: provide nspot and fspot (sigma_k computed from Eq. kernel_Nspot)
_AMPLITUDE_KEYS_SIGMA = {"sigma_k"}
_AMPLITUDE_KEYS_PHYSICAL = {"nspot", "fspot"}


# ======================================================================
# Compute lightcurve simulations and autocovariance for given parameters
# ======================================================================

def generate_sims(theta, nsim=1e3, **kwargs):
    """
    Generate synthetic lightcurves for a given set of parameters.

    Args:
        theta (tuple): Tuple containing peq, kappa, inc, and nspot parameters.
        nsim (int, optional): Number of simulations. Defaults to 1000.
        **kwargs: Additional arguments for the LightcurveModel class.

    Returns:
        numpy.ndarray: Array of synthetic lightcurves.
    """
    peq, kappa, inc, nspot = theta

    fluxes = []
    for _ in range(int(nsim)):
        sp = LightcurveModel(peq, kappa, inc, nspot, **kwargs)
        fluxes.append(sp.flux)

    return np.array(fluxes)


def avg_covariance_tlag(K):

    return np.array([np.mean(np.diagonal(K, offset=ti)) for ti in range(len(K))])


def plot_covariance(fluxes):
    """
    Plot the covariance matrix.

    Args:
        fluxes (numpy.ndarray): Array of fluxes.

    Returns:
        matplotlib.figure.Figure: The figure containing the covariance matrix plot.
    """
    K = np.cov(fluxes.T)
    fig, ax = plt.subplots(figsize=[12,12])
    pl = ax.matshow(K, cmap='binary_r', interpolation='none')
    plt.colorbar(pl, fraction=0.046, pad=0.04)
    plt.close()

    return fig

# ======================================================================
# Gaussian Process with numerical kernel
# ======================================================================

class NumericalKernel(object):
    """
    Gaussian Process for Stellar Rotation

    Parameters:
    - hparam: dict of hyperparameters.
              Required keys: peq, kappa, inc, lspot, tau, alpha_max.
              For the kernel amplitude, provide EITHER:
                - sigma_k : overall amplitude prefactor, OR
                - nspot + fspot : number of spots and spot contrast, from which
                  sigma_k is computed as sqrt(N_spot) * (1 - f_spot) / pi.
              Note: nspot is always required for the numerical simulations.
    - tsim: simulation time (default: 20)
    - tsamp: time sampling (default: 0.05)
    - nsim: number of simulations (default: 1e3)
    - verbose: whether to print verbose output (default: True)

    """
    def __init__(self, hparam, tsim=20, tsamp=0.05, nsim=1e3, verbose=True):

        t0 = time.time()

        if not isinstance(hparam, dict):
            raise TypeError("hparam must be a dict")

        missing = _REQUIRED_KEYS - set(hparam.keys())
        if missing:
            raise ValueError(f"hparam dict is missing required keys: {missing}")

        if "nspot" not in hparam:
            raise ValueError("nspot is required for numerical simulations")

        has_sigma = "sigma_k" in hparam
        has_physical = "nspot" in hparam and "fspot" in hparam

        if not has_sigma and not has_physical:
            raise ValueError(
                "hparam must contain either 'sigma_k' or both 'nspot' and 'fspot'")

        self.hparam = dict(hparam)

        self.peq       = self.hparam["peq"]
        self.kappa     = self.hparam["kappa"]
        self.inc       = self.hparam["inc"]
        self.nspot     = self.hparam["nspot"]
        self.lspot     = self.hparam["lspot"]
        self.tau       = self.hparam["tau"]
        self.alpha_max = self.hparam["alpha_max"]

        if has_sigma:
            self.sigma_k = self.hparam["sigma_k"]
        else:
            nspot = self.hparam["nspot"]
            fspot = self.hparam["fspot"]
            self.sigma_k = np.sqrt(nspot) * (1 - fspot) / np.pi
            self.hparam["sigma_k"] = self.sigma_k

        self.verbose   = verbose
        self.tsim      = tsim

        # create kernel function with these hyperparameters
        self.tarr = np.arange(0, tsim, tsamp)
        self.autocov, self.fluxes = self._compute_autocovariance(self.hparam, tsim=tsim, tsamp=tsamp, nsim=nsim)
        self.autocor = self.autocov / self.autocov[0]
        self.kernel_function = interpolate.interp1d(self.tarr, self.autocor)
        self.tsamp = tsamp

        if self.verbose:
            print(f"Kernel init time: {np.round(time.time() - t0, 2)}")

    def _generate_fluxes(self, theta, tsim=50, tsamp=0.05, nsim=1e3):
        """Generate raw flux simulations from hyperparameters.

        Accepts theta as a dict (any key order) or a positional list.

        Returns
        -------
        fluxes : ndarray of shape (nsim, ntime)
            Simulated lightcurves.
        """
        if isinstance(theta, dict):
            peq, kappa, inc, nspot = theta["peq"], theta["kappa"], theta["inc"], theta["nspot"]
            lspot, tau, alpha = theta["lspot"], theta["tau"], theta["alpha_max"]
        else:
            peq, kappa, inc, nspot, lspot, tau, alpha = theta

        fluxes = generate_sims(np.array([peq, kappa, inc, nspot]),
                               nsim, tem=tau, tdec=tau, alpha_max=alpha,
                               lspot=lspot, tsim=tsim, tsamp=tsamp)
        return fluxes

    def _compute_covariance_matrix(self, theta, tsim=50, tsamp=0.05, nsim=1e3):
        """Compute the covariance matrix from simulated lightcurves."""
        fluxes = self._generate_fluxes(theta, tsim=tsim, tsamp=tsamp, nsim=nsim)
        return np.cov(fluxes.T)

    def _compute_autocovariance(self, theta, tsim=50, tsamp=0.1, nsim=1e3):
        fluxes = self._generate_fluxes(theta, tsim=tsim, tsamp=tsamp, nsim=nsim)
        cov_matrix = np.cov(fluxes.T)
        avg_cov = avg_covariance_tlag(cov_matrix)
        return avg_cov, fluxes

    def _compute_autocorrelation(self, theta, tsim=50, tsamp=0.1, nsim=1e3):
        fluxes = self._generate_fluxes(theta, tsim=tsim, tsamp=tsamp, nsim=nsim)
        cov_matrix = np.cov(fluxes.T)
        avg_cov = avg_covariance_tlag(cov_matrix)
        avg_cor = avg_cov / np.var(fluxes)
        return avg_cor

    def get_acf(self):

        return self.tarr, self.autocor

    def compute_psd(self, tarr=None, freq_min=None, freq_max=None, normalization="psd", nsims=100):

        if tarr is None:
            tarr = self.tarr

        idx = np.random.choice(np.arange(len(self.fluxes)), size=nsims, replace=False)
        random_fluxes = self.fluxes[idx]

        psd_list = []
        for yt in random_fluxes:
            psd_freq, psd_power = compute_psd(yt, t=tarr, normalization=normalization,
                                              freq_min=freq_min, freq_max=freq_max)
            psd_list.append(psd_power)
        psd_list = np.array(psd_list)

        self.psd_freq = np.array(psd_freq)
        self.psd_power = np.median(np.stack(psd_list), axis=0)

        return self.psd_freq, self.psd_power

    def plot_autocorrelation(self):
        """
        Plot the autocorrelation function.

        Returns:
        - fig: matplotlib Figure object
        """

        fig = plt.figure(figsize=[12,6])
        plt.plot(self.tarr, self.autocor)
        plt.plot(self.tarr, self.kernel_function(self.tarr), linestyle="--")
        for ii in range(int(self.tsim / self.peq)+1):
            plt.axvline(ii*self.peq, color="k", alpha=0.2)
        plt.xlabel("Time lag", fontsize=25)
        plt.ylabel("Autocorrelation", fontsize=25)
        plt.xlim(min(self.tarr), max(self.tarr))
        plt.minorticks_on()
        plt.close()

        return fig
