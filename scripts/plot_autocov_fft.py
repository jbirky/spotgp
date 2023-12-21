import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('classic')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('figure', facecolor='w')
rc('xtick', labelsize=24)
rc('ytick', labelsize=24)

import sys
sys.path.append("../src")
import starspot


def compute_cov_mat(theta, tsim=50, tsamp=0.1, nsim=1e3):
    peq, kappa, inc, nspot, lspot, tau, alpha = theta
    fluxes = starspot.generate_sims(np.array([peq, kappa, inc, nspot]), nsim, tem=tau, tdec=tau, alpha_max=alpha, lspot=lspot, tsim=tsim, tsamp=tsamp)
    return np.cov(fluxes.T)


def compute_cov(theta, tsim=50, tsamp=0.1, nsim=1e3):
    fluxes = compute_cov_mat(theta, tsim=tsim, tsamp=tsamp, nsim=nsim)
    avg_cov = starspot.avg_covariance_tlag(np.cov(fluxes.T))
    return avg_cov


def compute_ft(time_series):
    ft = np.fft.fft(time_series)
    freq = np.fft.fftfreq(len(time_series))
    order = np.argsort(freq)
    return freq[order], ft[order]


def plot_autocov(theta_var, tarr, index=0, tsim=100, tsamp=0.01, nsim=1e3, var="var"):

    autocov = np.empty((len(theta_var), len(tarr)))
    for ii, tt in enumerate(theta_var):
        autocov[ii] = compute_cov(tt, tsim=tsim, tsamp=tsamp, nsim=nsim)

    peq = theta_var[0][0]

    fig1 = plt.figure(figsize=[12,6])
    for ii, tt in enumerate(theta_var):
        plt.plot(tarr, autocov[ii], label=r"$%s=%s$"%(var, tt[index]))
    for ii in range(int(tsim / peq)+1):
        plt.axvline(ii*peq, color="k", alpha=0.2)
    plt.xlabel("Time lag", fontsize=25)
    plt.ylabel("Autocovariance", fontsize=25)
    plt.legend(loc="upper right", fontsize=20)
    plt.xlim(0, 50)
    plt.minorticks_on()
    plt.close()

    # --------------------------------
    fig2 = plt.figure(figsize=[12,6])
    for ii, tt in enumerate(theta_var):
        freq, ft = compute_ft(autocov[ii])
        plt.plot(freq * 2*np.pi / tsamp, ft, label=r"$%s=%s$"%(var, tt[index]))

    plt.axvline(2*np.pi / peq, color="k", linestyle="--")
    plt.axvline(-2*np.pi / peq, color="k", linestyle="--")
    plt.axvline(2*np.pi / peq * 2, color="k", linestyle="--")
    plt.axvline(-2*np.pi / peq * 2, color="k", linestyle="--")

    plt.xlabel("Frequency", fontsize=25)
    plt.ylabel("Amplitude", fontsize=25)
    plt.legend(loc="upper right", fontsize=20)
    plt.xlim(-5, 5)
    plt.minorticks_on()
    plt.close()

    return fig1, fig2, autocov, freq, ft


# ====================================================
nsim = 1e3
tsim = 100
tsamp = 0.01
tarr = np.arange(0,tsim,tsamp)

labels = ["P_{eq}", "\kappa", "inc", "nspot", "lspot", "\tau", "\alpha"]

theta_base = np.array([
    [3.0, 0.0, np.pi/2, 10, 20, 1., 1.],
    [3.0, 0.0, np.pi/2, 10, 20, 1., 1.],
    [3.0, 0.0, np.pi/2, 10, 20, 1., 1.]
])

peq_var = [1, 5, 10]
kappa_var = [-0.3, 0, 0.3]
inc_var = [np.pi/6, np.pi/4, np.pi/2]
nspot_var = [5, 10, 20]
lspot_var = [10, 20, 30]
tau_var = [1, 5, 10]
alpha_var = [.01, .05, .1]

var_vals = np.array([peq_var, kappa_var, inc_var, nspot_var, lspot_var, tau_var, alpha_var])

for ii in range(len(var_vals)):
    print(labels[ii])
    theta_var = theta_base.copy()
    theta_var[:,0] = var_vals[ii]

    fig1, fig2, autocov, freq, ft = plot_autocov(theta_var, tarr, index=ii, tsim=tsim, tsamp=tsamp, nsim=1e3, var=labels[ii])

    fig1.savefig("../results/autocov_%s.png"%labels[ii], bbox_inches="tight")
    fig2.savefig("../results/ft_%s.png"%labels[ii], bbox_inches="tight")
    np.savez("../results/autocov_%s.npz"%labels[ii], autocov=autocov, tarr=tarr, theta_var=theta_var, freq=freq, ft=ft)
