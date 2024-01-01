import numpy as np
from scipy.optimize import minimize

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
import rotation_kernel as rk

__all__ = ["format_title",
           "TrainKernel"]


keys_trend = ["a10", "a20", "c10", "c20"]
keys_cos = ["a11", "a21", "c11", "c21"]
keys_sin = ["b11", "b21"]
keys_cos2 = ["a12", "a22", "c12", "c22"]


def get_p0(pdict, key_list):
    
    return [pdict.get(key) for key in key_list]


def format_title(variable_dict, sig_fig=4):
        
    formatted_string = ""
    for var_name, var_value in variable_dict.items():
        formatted_string += f"{var_name} = {np.round(var_value, sig_fig)}\n"
        
    return formatted_string


class TrainKernel(object):

    def __init__(self, hyperparam, tsim=1000, tsamp=0.1, nsim=5e2, tcut=100, fit_ft=False):

        self.hyperparam = hyperparam

        self.peq = hyperparam["peq"]
        self.omega = (2*np.pi) / self.peq

        self.gp = rk.RotationGP(hyperparam.values(), tsim=tsim, tsamp=tsamp, nsim=nsim)

        tfine = np.arange(0,max(self.gp.tarr),1e-3)
        autocov = self.gp.kernel(tfine)
        autocor = autocov / np.var(self.gp.fluxes)

        # max time lag to use for kernel
        self.tcut = tcut
        idx = np.where(tfine < tcut)

        self.ydata, self.xdata = autocor[idx], tfine[idx]
        self.freq, self.power = rk.compute_ft(self.ydata, self.xdata)

        # Options 
        self.fit_ft = fit_ft

    def kernel_trend(self, x, par):
        
        term1 = -par[0] * np.exp(-x / par[2]**2)
        term2 = par[1] * np.exp(-x**2 / par[3]**2)
        
        return term1 + term2

    def kernel_cos(self, x, par):
        
        term1 = par[0] * np.exp(-x / par[2]**2) * np.cos(self.omega * x)
        term2 = par[1] * np.exp(-x**2 / par[3]**2) * np.cos(self.omega * x)
        
        return term1 + term2 

    def kernel_cos_second(self, x, par):
        
        term1 = par[0] * np.exp(-x / par[2]**2) * np.cos(2 * self.omega * x)
        term2 = par[1] * np.exp(-x**2 / par[3]**2) * np.cos(2 * self.omega * x)
        
        return term1 + term2 

    def kernel_sin(self, x, par):
        
        term1 = par[0] * np.exp(-x / par[2]**2) * np.sin(self.omega * x)
        term2 = par[1] * np.exp(-x**2 / par[3]**2) * np.sin(self.omega * x)
        
        return term1 + term2 

    def kernel_sin_cos(self, x, par):
        
        term1 = par[0] * np.exp(-x / par[4]**2) * np.sin(self.omega * x)
        term2 = par[1] * np.exp(-x**2 / par[5]**2) * np.sin(self.omega * x)
        
        term3 = par[2] * np.exp(-x / par[4]**2) * np.cos(self.omega * x)
        term4 = par[3] * np.exp(-x**2 / par[5]**2) * np.cos(self.omega * x)
        
        return term1 + term2 + term3 + term4
        
    # ===================================

    def loss_trend(self, params):
        
        y_pred = self.kernel_trend(self.xdata, params)
        error = np.mean((y_pred - self.ydata)**2)
        
        if self.fit_ft:
            power_pred = rk.compute_ft(y_pred, self.xdata)[1]
            error += np.mean((np.real(power_pred) - np.real(self.power))**2)
        
        return error

    def loss_cos(self, params, trend_param):
        
        y_trend = self.kernel_trend(self.xdata, trend_param)
        y_pred = self.kernel_cos(self.xdata, params) 
        error = np.mean((y_pred + y_trend - self.ydata)**2)
        
        if self.fit_ft:
            power_pred = rk.compute_ft(y_pred, self.xdata)[1]
            error += np.mean((np.real(power_pred) - np.real(self.power))**2)
        
        return error

    def loss_cos_second(self, params, trend_param, cos_param):
        
        y_trend = self.kernel_trend(self.xdata, trend_param)
        y_cos = self.kernel_cos(self.xdata, cos_param)  
        y_pred = self.kernel_cos_second(self.xdata, params)
        error = np.mean((y_pred + y_trend + y_cos - self.ydata)**2)
        
        if self.fit_ft:
            power_pred = rk.compute_ft(y_pred, self.xdata)[1]
            error += np.mean((np.real(power_pred) - np.real(self.power))**2)
        
        return error

    def loss_sin(self, params, trend_param, cos_param):
        
        y_trend = self.kernel_trend(self.xdata, trend_param)
        y_cos = self.kernel_cos(self.xdata, cos_param)  
        y_pred = self.kernel_sin(self.xdata, np.append(params, cos_param[2:]))
        error = np.mean((y_pred + y_trend + y_cos - self.ydata)**2)
        
        if self.fit_ft:
            power_pred = rk.compute_ft(y_pred, self.xdata)[1]
            error += np.mean((np.real(power_pred) - np.real(self.power))**2)
        
        return error

    def loss_sin_cos(self, params, trend_param):
        
        y_trend = self.kernel_trend(self.xdata, trend_param)  
        y_pred = self.kernel_sin_cos(self.xdata, params)
        error = np.mean((y_pred + y_trend - self.ydata)**2)
        
        if self.fit_ft:
            power_pred = rk.compute_ft(y_pred, self.xdata)[1]
            error += np.mean((np.real(power_pred) - np.real(self.power))**2)
        
        return error

    # ===================================

    def fit_kernel_trend(self, p0=None):
        
        lower_bounds = np.zeros_like(p0)
        upper_bounds = np.ones_like(p0) * np.inf
        bounds = np.array([lower_bounds, upper_bounds]).T

        result = minimize(self.loss_trend, p0, method="trust-constr", bounds=bounds)
        
        return result.x

    def fit_kernel_cos(self, trend_param, p0=None):

        lower_bounds = np.zeros_like(p0)
        upper_bounds = np.ones_like(p0) * np.inf
        bounds = np.array([lower_bounds, upper_bounds]).T

        result = minimize(self.loss_cos, p0, args=(trend_param), method="trust-constr", bounds=bounds)
        
        return result.x

    def fit_kernel_cos_second(self, trend_param, cos_param, p0=None):

        lower_bounds = np.zeros_like(p0)
        upper_bounds = np.ones_like(p0) * np.inf
        bounds = np.array([lower_bounds, upper_bounds]).T

        result = minimize(self.loss_cos_second, p0, args=(trend_param, cos_param), method="trust-constr", bounds=bounds)
        
        return result.x

    def fit_kernel_sin(self, trend_param, cos_param, p0=None):

        result = minimize(self.loss_sin, p0, args=(trend_param, cos_param), method="trust-constr")
        
        return result.x

    def fit_kernel_sin_cos(self, trend_param, p0=None):

        result = minimize(self.loss_sin_cos, p0, args=(trend_param), method="trust-constr")
        
        return result.x

    # ===================================

    def fit_model(self, pinit):
        
        ptrend = self.fit_kernel_trend(p0=get_p0(pinit, keys_trend))
        pcos = self.fit_kernel_cos(ptrend, p0=get_p0(pinit, keys_cos))
        psin = self.fit_kernel_sin(ptrend, pcos, p0=get_p0(pinit, keys_sin))  
        psin_cos = self.fit_kernel_sin_cos(ptrend, p0=np.append(psin, pcos))
        pcos2 = self.fit_kernel_cos_second(ptrend, pcos, p0=get_p0(pinit, keys_cos2))

        res_trend = dict(zip(keys_trend, ptrend))
        res_sin_cos = dict(zip(keys_sin + keys_cos, psin_cos))
        res_cos2 = dict(zip(keys_cos2, pcos2))
        
        plist = [res_trend, res_sin_cos, res_cos2]
        pdict = {**res_trend, **res_sin_cos, **res_cos2}
    
        return plist, pdict

    def predict(self, pfit):

        ptrend = get_p0(pfit, keys_trend)
        psin_cos = get_p0(pfit, keys_sin + keys_cos)
        pcos2 = get_p0(pfit, keys_cos2)
        
        ytrend = self.kernel_trend(self.xdata, ptrend)
        ysin_cos = self.kernel_sin_cos(self.xdata, psin_cos)
        ycos2 = self.kernel_cos_second(self.xdata, pcos2)
        ypred = ytrend + ysin_cos + ycos2

        power_pred = rk.compute_ft(ypred, self.xdata)[1]
        
        return ypred, power_pred

    def plot_kernel_fit(self, ypred, text_blocks=[""], log_power=False):

        fig, axes = plt.subplots(2, 1, figsize=(20, 16))

        axes[0].plot(self.xdata, self.ydata, color="k", label=format_title(self.hyperparam))
        axes[0].plot(self.xdata, ypred, color="r", linewidth=3, alpha=0.5)
        for ii in range(30):
            axes[0].axvline(self.peq * ii, color="k", linestyle="--", alpha=0.3)
        axes[0].axhline(0, color="k", linestyle="--", alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=25, frameon=False)
        axes[0].set_xlim(0, self.tcut)
        axes[0].set_xlabel("Time lag [days]", fontsize=30)
        axes[0].set_ylabel("Autocorrelation", fontsize=30)
        axes[0].minorticks_on()

        freq_pred, power_pred = rk.compute_ft(ypred, self.xdata)
        
        if log_power == True:
            axes[1].plot(self.freq, np.log(self.power), color="k")
            axes[1].plot(freq_pred, np.log(power_pred), color="r", linewidth=3, alpha=0.5)
            axes[1].set_ylabel("log(Power)", fontsize=30)
        else:
            axes[1].plot(self.freq, self.power, color="k")
            axes[1].plot(freq_pred, power_pred, color="r", linewidth=3, alpha=0.5)
            axes[1].set_ylabel("Power", fontsize=30)

        axes[1].axvline(0, color="k", linestyle="--", alpha=0.3)
        axes[1].axvline(-self.omega, color="k", linestyle="--", alpha=0.3)
        axes[1].axvline(self.omega, color="k", linestyle="--", alpha=0.3)
        axes[1].axvline(-2 * self.omega, color="k", linestyle="--", alpha=0.3)
        axes[1].axvline(2 * self.omega, color="k", linestyle="--", alpha=0.3)
        axes[1].set_xlim(-2.5 * self.omega, 2.5 * self.omega)
        axes[1].set_xlabel(r"Frequency [$2\pi \cdot \mathrm{days}^{-1}$]", fontsize=30)
        axes[1].minorticks_on()
        
        for ii, text in enumerate(text_blocks):
            axes[1].text(0.02+.15*ii, 0.97, text, transform=plt.gca().transAxes, va="top", fontsize=22, color='red')

        plt.subplots_adjust(hspace=0.2)
        plt.close()

        return fig