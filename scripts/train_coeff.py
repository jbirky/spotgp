import numpy as np
import pandas as pd
import tqdm
import time
import multiprocessing as mp

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
import kernel_fourier as kf


hyperparam = {
    'peq': 6.,
    'kappa': 0.,
    'inc': np.pi/2,
    'nspot': 20,
    'lspot': 20,
    'tau': 10,
    'alpha_max': .01
}

train_kernel = kf.TrainKernel(hyperparam)

pinit = {"a10": 0.1, "a20": 0.5, "c10": hyperparam["tau"], "c20": hyperparam["lspot"],
         "a11": 1.0, "a21": 0.1, "b11": 0, "b21": 0, "c11": hyperparam["tau"], "c21": hyperparam["tau"], 
         "a12": 0, "a22": 0, "c12": hyperparam["tau"], "c22": hyperparam["lspot"]}

plist, pdict = train_kernel.fit_model(pinit)
ypred = train_kernel.predict(pdict)

train_kernel.plot_kernel_fit(ypred, text_blocks=[kf.format_title(par) for par in plist])