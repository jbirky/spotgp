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

import starspot

__all__ = ["RotationGP", 
           "compute_covariance_matrix", 
           "compute_autocovariance",
           "compute_ft"]


def compute_covariance_matrix(theta, tsim=50, tsamp=0.05, nsim=1e3):

    peq, kappa, inc, nspot, lspot, tau, alpha = theta
    fluxes = starspot.generate_sims(np.array([peq, kappa, inc, nspot]), 
                                    nsim, tem=tau, tdec=tau, alpha_max=alpha, 
                                    lspot=lspot, tsim=tsim, tsamp=tsamp)
    
    return np.cov(fluxes.T)


def compute_autocovariance(theta, tsim=50, tsamp=0.1, nsim=1e3):

    fluxes = compute_covariance_matrix(theta, tsim=tsim, tsamp=tsamp, nsim=nsim)
    avg_cov = starspot.avg_covariance_tlag(np.cov(fluxes.T))

    return avg_cov, fluxes

def compute_autocorrelation(theta, tsim=50, tsamp=0.1, nsim=1e3):

    fluxes = compute_covariance_matrix(theta, tsim=tsim, tsamp=tsamp, nsim=nsim)
    avg_cov = starspot.avg_covariance_tlag(np.cov(fluxes.T))
    avg_cor = avg_cov / np.var(fluxes.T)

    return avg_cor


def compute_ft(time_series, tarr, log_power=False):

    ft = np.abs(np.fft.fft(time_series))
    freq = np.fft.fftfreq(len(time_series)) * 2*np.pi / (tarr[1] - tarr[0])
    order = np.argsort(freq)

    nsamples = len(time_series)
    ft *= 2 / nsamples

    if log_power == True:
        return freq[order], np.log(ft[order])
    else:
        return freq[order], ft[order]


class RotationGP(object):
    """
    Gaussian Process for Stellar Rotation

    Parameters:
    - hparam: list of hyperparameters [peq, kappa, inc, nspot, lspot, tau, alpha_max]
    - tsim: simulation time (default: 20)
    - tsamp: time sampling (default: 0.05)
    - nsim: number of simulations (default: 1e3)
    - verbose: whether to print verbose output (default: True)

    """
    def __init__(self, hparam, tsim=20, tsamp=0.05, nsim=1e3, verbose=True):
        
        t0 = time.time()
        
        # create kernel function with these hyperparameters
        self.tarr = np.arange(0,tsim,tsamp)
        self.autocov, self.fluxes = compute_autocovariance(hparam, tsim=tsim, tsamp=tsamp, nsim=nsim)
        self.kernel = interpolate.interp1d(self.tarr, self.autocov)
        
        self.hparam = hparam
        [self.peq, self.kappa, self.inc, self.nspot, self.lspot, self.tau, self.alpha_max] = hparam
        self.verbose = verbose
        self.tsim = tsim
        self.tsamp = tsamp
        
        if self.verbose:
            print(f"Kernel init time: {np.round(time.time() - t0, 2)}")
        
    def train(self, xtrain, ytrain):
        """
        Train the Gaussian Process model.

        Args:
        - xtrain: array-like, shape (n_samples,): training input data
        - ytrain: array-like, shape (n_samples,): training target data
        """
        
        t0 = time.time()
        
        delta_matrix = np.abs(xtrain[:, np.newaxis] - xtrain)
        self.Kmat = self.kernel(delta_matrix)
        
        # Pseudo-inverse
        self.Kinv = np.linalg.pinv(self.Kmat)
        self.alpha = self.Kinv @ ytrain
        
        # Compute log-likelihood 
        _ , logdet_K = np.linalg.slogdet(self.Kmat)
        self.lnlike = -1/2 * (ytrain.T @ self.alpha + logdet_K + xtrain.shape[0] * np.log(2*np.pi))
        
        self.xtrain = xtrain
        self.ytrain = ytrain
        
        if self.verbose:
            print(f"Train time: {np.round(time.time() - t0, 2)}")

    def predict(self, xtest):
        """
        Make predictions using the trained Gaussian Process model.

        Args:
        - xtest: array-like, shape (n_samples,): test input data

        Returns:
        - ypred: array-like, shape (n_samples,): predicted target data
        - yvar: array-like, shape (n_samples,): predictive variance
        """
        
        t0 = time.time()
        
        kvec = self.kernel(np.abs(self.xtrain - xtest[:, np.newaxis]))
        ypred = kvec @ self.alpha
        yvar = np.diag(kvec @ self.Kinv @ kvec.T)
        
        self.xtest = xtest
        self.ypred = ypred
        self.yvar = yvar
        self.ystd = np.sqrt(yvar)
        
        if self.verbose:
            print(f"Prediction time: {np.round(time.time() - t0, 2)}")

        return ypred, yvar
    
    def plot_autocovariance(self):
        """
        Plot the autocovariance function.
        
        Returns:
        - fig: matplotlib Figure object
        """
        
        fig = plt.figure(figsize=[12,6])
        plt.plot(self.tarr, self.autocov)
        plt.plot(self.tarr, self.kernel(self.tarr), linestyle="--")
        for ii in range(int(self.tsim / self.peq)+1):
            plt.axvline(ii*self.peq, color="k", alpha=0.2)
        plt.xlabel("Time lag", fontsize=25)
        plt.ylabel("Autocovariance", fontsize=25)
        plt.xlim(min(self.tarr), max(self.tarr))
        plt.minorticks_on()
        plt.close()

        return fig

    def plot_prediction(self):
        """
        Plot the prediction results.

        Returns:
        - fig: matplotlib Figure object
        """

        fig = plt.figure(figsize=[16,6])
        plt.scatter(self.xtrain, self.ytrain, edgecolor="none", facecolor="k", s=30)
        plt.plot(self.xtest, self.ytest, color="k", alpha=0.8, linewidth=1.5)
        plt.plot(self.xtest, self.ypred, color="r", alpha=0.8, linewidth=2, label=int(np.round(self.lnlike)))
        plt.fill_between(self.xtest, self.ypred - self.ystd, self.ypred + self.ystd, color="r", alpha=0.1)
        plt.xlim(min(self.xtest), max(self.xtest))
        plt.xlabel("Time [d]", fontsize=25)
        plt.ylabel("Flux", fontsize=25)
        plt.legend(loc="upper left", fontsize=22)
        plt.minorticks_on()
        plt.close()

        return fig