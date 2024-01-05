import numpy as np
import pandas as pd
import tqdm
import multiprocessing as mp
import os
import sys
sys.path.append("../src")
import kernel_fourier as kf

os.nice(10)

# ================================================
# Configuration 

ncore = 64
trial = "4"
save_outputs = False
save_plots = False

df = pd.read_csv(f"../files/training_parameters{trial}.csv")
df = df[['peq', 'kappa', 'inc', 'nspot', 'lspot', 'tau', 'alpha_max']]
train_dict = df.to_dict(orient='records')

save_file = f"../files/train_parameters_fit_coefficients{trial}.csv"
data_dir = f"../results/data{trial}/"
plot_dir = f"../results/plots{trial}/"

for path in [plot_dir, data_dir]:
    if not os.path.exists(path):
        os.makedirs(path)

# ================================================

def fit_kernel_parameters(ii):

    opt_bounds = {
        "a00": [-0.5, 0.5],
        "a10": [-10, 0], 
        "a20": [0, 10], 
        "c10": [0, np.inf], 
        "c20": [0, np.inf], 
        "b11": [-10, 10],
        "b21": [-10, 10],
        "a11": [0, 10], 
        "a21": [-10, 10],
        "c11": [0, np.inf], 
        "c21": [0, np.inf], 
        "a12": [0, 10],
        "a22": [0, 10],
        "c12": [0, np.inf], 
        "c22": [0, np.inf], 
    }
    
    # initialize numerical kernel
    hyperparam = train_dict[ii]
    train_kernel = kf.TrainKernel(hyperparam, tsim=500, tsamp=0.1, nsim=5e2, tcut=200, 
                                fit_ft=True, log_params=False, log_power=False,
                                fit_orders=[0,1], fit_sin=True, opt_bounds=opt_bounds)
    
    pinit = {
        "a10": train_kernel.ydata[-1], 
        "a20": (max(train_kernel.ydata) - min(train_kernel.ydata))/2, 
        "c10": 1000, 
        "c20": hyperparam["lspot"] + 0.5*hyperparam["tau"],
        "a11": 1, 
        "a21": -0.5, 
        "b11": 0, 
        "b21": 0, 
        "c11": 9/80*(hyperparam["lspot"] + hyperparam["tau"]) - 1/4, 
        "c21": 9/40*(hyperparam["lspot"] + hyperparam["tau"]) - 1/2, 
        "a12": 0, 
        "a22": 0, 
        "c12": 10, 
        "c22": 10
    }
        
    # fit model coefficients to kernel
    plist, pdict = train_kernel.fit_model(pinit)
    ypred, power_pred = train_kernel.predict(pdict)

    # compute error
    ymse = np.mean((ypred - train_kernel.ydata)**2)
    pmse = np.mean((power_pred - train_kernel.power)**2)

    str_index = f"{ii}".rjust(5, "0")
    if save_outputs == True:
        # Save training data
        np.savez(os.path.join(data_dir, f"sim_{str_index}"), 
                 fluxes=train_kernel.gp.fluxes,
                 xdata=train_kernel.xdata, ydata=train_kernel.ydata,
                 freq=train_kernel.freq, power=train_kernel.power,
                 ypred=ypred, power_pred=power_pred)

    if save_plots == True:
        # Save figure
        if (ii % 10) == 0:
            fig = train_kernel.plot_kernel_fit(ypred, text_blocks=[kf.format_title(par) for par in plist])
            fig.savefig(os.path.join(plot_dir, f"sim_{str_index}.png"), bbox_inches="tight")
    
    summary = {"sim_id": ii, "ymse": ymse, "pmse": pmse}

    return {**summary, **hyperparam, **pdict}


# ================================================

if __name__ == "__main__":

    index_array = np.arange(len(train_dict))

    with mp.Pool(ncore) as p:
        outputs = []
        for result in tqdm.tqdm(p.imap(func=fit_kernel_parameters, iterable=index_array), total=len(index_array)):
            outputs.append(result)
        outputs = np.array(outputs)

    # merge all output dictionaries into one
    result_dict = {}
    for d in outputs:
        for key, value in d.items():
            if key not in result_dict:
                result_dict[key] = []
            result_dict[key].append(value)
            
    # save results to csv file 
    result_df = pd.DataFrame(result_dict)
    result_df.to_csv(save_file, index=False)