import numpy as np
import tqdm
from functools import partial
import multiprocessing as mp
import warnings

import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('classic')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('figure', facecolor='w')
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

__all__ = ["StarSpot",
           "generate_sims",
           "avg_covariance_tlag",
           "generate_training_sample",
           "plot_lightcurve",
           "plot_covariance"]


class StarSpot(object):
    """
    Class representing a star with spots and its lightcurve.

    Args:
        peq (float): Equatorial period of the star.
        kappa (float): Differential rotation shear.
        inc (float): Inclination of the star.
        nspot (int): Number of spots.
        tem (float or list or np.ndarray, optional): Emergence timescale of the spots. Defaults to 1.
        tdec (float or list or np.ndarray, optional): Decay timescale of the spots. Defaults to 2.
        alpha_max (float or list or np.ndarray, optional): Maximum angular area of the spots. Defaults to 0.1.
        fspot (float or list or np.ndarray, optional): Spot contrast fraction. Defaults to 0.
        lspot (float or list or np.ndarray, optional): Spot lifetime. Defaults to 2.
        long (list, optional): Range of spot longitudes. Defaults to [0, 2*np.pi].
        lat (list, optional): Range of spot latitudes. Defaults to [0, np.pi].
        tsim (float, optional): End simulation time. Defaults to 28.
        tsamp (float, optional): Sampling cadence. Defaults to 0.02.
        limb_darkening (bool, optional): Flag to enable star and spot limb darkening. Defaults to False.

    Attributes:
        tsim (float): End simulation time.
        tsamp (float): Sampling cadence.
        t (numpy.ndarray): Time array from 0 to tsim with tsamp cadence.
        peq (float): Equatorial period of the star.
        kappa (float): Differential rotation shear.
        inc (float): Inclination of the star.
        nspot (int): Number of spots.
        tem (float or numpy.ndarray): Emergence timescale of the spots.
        tdec (float or numpy.ndarray): Decay timescale of the spots.
        alpha_max (float or numpy.ndarray): Maximum angular area of the spots.
        fspot (float or numpy.ndarray): Spot contrast fraction.
        lspot (float or numpy.ndarray): Spot lifetime.
        long (list): Range of spot longitudes.
        lat (list): Range of spot latitudes.
        tmax (numpy.ndarray): Array of random spot emergence times.
        limb_darkening (bool): Flag to enable star and spot limb darkening.
        limbc (numpy.ndarray): Star limb darkening coefficients.
        limbd (numpy.ndarray): Spot limb darkening coefficients (same as limbc by default).
        dspots (numpy.ndarray): Flux removed from each spot in the lightcurve.
        dlimb (float): Flux removed from stellar limb darkening.
        flux (numpy.ndarray): Total remaining flux in the lightcurve.
    """
    def __init__(self, peq, kappa, inc, nspot, 
                 tem=1, 
                 tdec=2, 
                 alpha_max=0.1, 
                 fspot=0, 
                 lspot=2,
                 long=[0,2*np.pi], 
                 lat=[0,np.pi], 
                 tsim=28, 
                 tsamp=0.02,
                 limb_darkening=False):
        
        # simulation parameters
        self.tsim = tsim            # end simulation time
        self.tsamp = tsamp          # sampling cadence
        self.t = np.arange(0, self.tsim, self.tsamp)
        
        # star properties
        self.peq = peq              # equitorial period
        self.kappa = kappa          # differential rotation shear
        self.inc = inc              # inclination
        self.nspot = int(nspot)     # number of spots

        # spot properties
        self.tem = self.assign_property(tem)              # emergence timescale
        self.tdec = self.assign_property(tdec)            # decay timescale
        self.alpha_max = self.assign_property(alpha_max)  # max angular area
        self.fspot = self.assign_property(fspot)          # spot contrast fraction
        self.lspot = self.assign_property(lspot)          # spot lifetime
        self.long = self.assign_property(long)            # spot longitude
        self.lat = self.assign_property(lat)              # spot latitude
        self.tmax = np.random.uniform(0, self.tsim, self.nspot)

        # star and spot limb darkening (kipping 2012, claret 2011)
        self.limb_darkening = limb_darkening
        self.limbc = np.array([0.3999, 0.4269, -0.0227, -0.0839])
        self.limbd = self.limbc

        # compute lightcurve
        self.dflux()

    def assign_property(self, var):

        if (type(var) == int) or (type(var) == float):
            assign = var
        elif (type(var) == list) or (type(var) == np.ndarray): 
            assign = np.random.uniform(var[0], var[1], self.nspot)
        else:
            msg  = "Invalid datatype for model parameter. "
            msg += "Valid types: int, float, list, np.ndarray"
            raise TypeError(msg)
        
        return assign
        
    def alphak(self, tmaxk):
        
        dt1 = self.t - tmaxk + self.lspot/2 + self.tem
        dt2 = self.t - tmaxk + self.lspot/2
        dt3 = self.t - tmaxk - self.lspot/2
        dt4 = self.t - tmaxk - self.lspot/2 - self.tdec

        alphak  = (dt1 * np.heaviside(dt1, 1) - dt2 * np.heaviside(dt2, 1)) / self.tem
        alphak += -(dt3 * np.heaviside(dt3, 1) - dt4 * np.heaviside(dt4, 1)) / self.tdec
        alphak *= self.alpha_max
        
        return alphak
    
    def betak(self, longk, latk, tmaxk):
        
        longk_t = longk + 2*np.pi/self.peq * (1 + self.kappa * np.sin(latk)**2) * (self.t - tmaxk)
        
        cosb  = np.cos(self.inc) * np.sin(latk) 
        cosb += np.sin(self.inc) * np.cos(latk) * np.cos(longk_t)
        betak_t = np.arccos(cosb)
        
        return betak_t, longk_t
    
    def stellar_limb(self):

        if self.limb_darkening == True:
            ncoeff = len(self.limbc)
            flimb = np.sum([n*self.limbc[n] / (n + ncoeff) for n in range(ncoeff)])
        else:
            flimb = 0

        return flimb
    
    def spot_limb(self, alpha, beta):

        if self.limb_darkening == True:
            ncoeff = len(self.limbc)
            zeta_n = zeta(beta - alpha)
            zeta_p = zeta(beta + alpha)

            terms = np.zeros((ncoeff, alpha.shape[0]))
            for ii in range(ncoeff):
                t1 = ncoeff * (self.limbc[ii] - self.limbd[ii]*self.fspot) / (ii + ncoeff)
                t2 = (zeta_n**((ii+4)/2) - zeta_p**((ii+4)/2)) / (zeta_n**2 - zeta_p**2)
                term = t1 * t2
                term[np.isnan(term)] = 0
                terms[ii] = term
            factor = np.sum(terms, axis=0)

        else:
            factor = (1 - self.fspot)

        return factor
        
    def dflux_k(self, longk, latk, tmaxk):
        
        warnings.simplefilter("ignore")
        betak_t  = self.betak(longk, latk, tmaxk)[0]
        alphak_t = self.alphak(tmaxk)
        
        cosa = np.cos(alphak_t)
        sina = np.sin(alphak_t)
        cota = 1/np.tan(alphak_t)
        cosb = np.cos(betak_t)
        sinb = np.sin(betak_t)
        cscb = 1/sinb
        cotb = 1/np.tan(betak_t)
        
        Ak  = np.emath.arccos(cosa * cscb).real
        Ak += cosb * sina**2 * np.emath.arccos(-cota * cotb).real
        Ak += -cosa * sinb * np.emath.sqrt(1 - cosa**2 * cscb**2).real

        dspot = Ak.real / np.pi * self.spot_limb(alphak_t, betak_t)

        return dspot
    
    def dflux(self):
        
        df = []
        for ii in range(self.nspot):
            dfk = self.dflux_k(self.long[ii], self.lat[ii], self.tmax[ii])
            df.append(dfk)
        
        # flux removed from spots
        self.dspots = np.array(df)

        # flux removed from stellar limb darkening
        self.dlimb = self.stellar_limb()

        # total remaining flux
        self.flux = 1 - self.dlimb - np.sum(self.dspots, axis=0)
        
        
def zeta(x):
    """
    Calculate the function zeta(x) for spot limb darkening.

    Args:
        x (float or numpy.ndarray): Input value(s).

    Returns:
        float or numpy.ndarray: The value of zeta(x).
    """

    return np.cos(x) * np.heaviside(x,1) * np.heaviside(np.pi/2 - x,1) + np.heaviside(-x,1)
    

def generate_sims(theta, nsim=1e3, **kwargs):
    """
    Generate synthetic lightcurves for a given set of parameters.

    Args:
        theta (tuple): Tuple containing peq, kappa, inc, and nspot parameters.
        nsim (int, optional): Number of simulations. Defaults to 1000.
        **kwargs: Additional arguments for the StarSpot class.

    Returns:
        numpy.ndarray: Array of synthetic lightcurves.
    """
    peq, kappa, inc, nspot = theta
    
    fluxes = []
    for _ in range(int(nsim)):
        sp = StarSpot(peq, kappa, inc, nspot, **kwargs)
        sp.dflux()
        fluxes.append(sp.flux)
    
    return np.array(fluxes)


def avg_covariance_tlag(K):
    
    return np.array([np.mean(np.diagonal(K, offset=ti)) for ti in range(len(K))])


def generate_training_sample(thetas, nsim=int(1e3), ncore=10, **kwargs):
    """
    Generate a training sample of covariance matrices for a set of parameters.

    Args:
        thetas (numpy.ndarray): Array of parameter sets.
        nsim (int, optional): Number of simulations per parameter set. Defaults to 1000.
        ncore (int, optional): Number of CPU cores to use for parallel processing. Defaults to 10.
        **kwargs: Additional arguments for the generate_sims function.

    Returns:
        numpy.ndarray: Array of covariance matrices.
    """
    gen = partial(generate_sims, nsim=nsim, **kwargs)
    with mp.Pool(ncore) as p:
        covs = []
        for fluxes in tqdm.tqdm(p.imap(func=gen, iterable=thetas), total=len(thetas)):
            K = np.cov(fluxes.T)
            covs.append(avg_covariance_tlag(K))
        
    return np.array(covs)


def plot_lightcurve(sp, show_spots=True):
    """
    Plot the lightcurve.

    Args:
        sp (StarSpot): The StarSpot object containing the lightcurve.
        show_spots (bool, optional): Whether to show individual spots. Defaults to True.

    Returns:
        matplotlib.figure.Figure: The figure containing the lightcurve plot.
    """
    flux = sp.flux + sp.dlimb
    fig = plt.figure(figsize=[16,6])
    if show_spots == True:
        for ii in range(sp.nspot):
            plt.plot(sp.t, 1-sp.dspots[ii], alpha=0.5)
    plt.plot(sp.t, flux, color="k")
    plt.ylim(min(flux)-2e-3, 1+1e-3)
    plt.xlim(sp.t[0], sp.t[-1])
    plt.minorticks_on()
    plt.ticklabel_format(axis='both', style='', useOffset=False)
    plt.close()

    return fig


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