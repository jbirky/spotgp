import numpy as np
import pandas as pd
import tqdm
import multiprocessing as mp
import os
import sys
sys.path.append("../src")
import kernel_fourier as kf


# ================================================
# Configuration 

ncore = 48
os.nice(10)

df = pd.read_csv("../files/training_parameters.csv.gz")
df = df[['peq', 'kappa', 'inc', 'nspot', 'lspot', 'tau', 'alpha_max']]
train_dict = df.to_dict(orient='records')

# ================================================

def fit_kernel_parameters(ii, results_dir="../results/"):
    
    hyperparam = train_dict[ii]
    train_kernel = kf.TrainKernel(hyperparam, tsim=1000, tsamp=0.1, nsim=5e2, tcut=200, fit_ft=False)
    
    pinit = {"a10": 0.1, "a20": 0.5, "c10": hyperparam["tau"] + hyperparam["lspot"], "c20": hyperparam["lspot"],
             "a11": 1.0, "a21": 0.1, "b11": 0, "b21": 0, "c11": hyperparam["tau"], "c21": hyperparam["tau"], 
             "a12": 0, "a22": 0, "c12": hyperparam["tau"], "c22": hyperparam["lspot"]}

    plist, pdict = train_kernel.fit_model(pinit)
    ypred, power_pred = train_kernel.predict(pdict)
    
    # Save figure
    str_index = f"{ii}".rjust(5, "0")
    fig = train_kernel.plot_kernel_fit(ypred, text_blocks=[kf.format_title(par) for par in plist])
    fig.savefig(os.path.join(results_dir, f"plots/sim_{str_index}.png"), bbox_inches="tight")
    
    # Save training data
    np.savez(os.path.join(results_dir, f"data/sim_{str_index}.png"), xdata=train_kernel.ydata, ydata=train_kernel.ydata,
         freq=train_kernel.freq, power=train_kernel.power,
         ypred=ypred, power_pred=power_pred)
    
    return {**{"sim_id": ii}, **hyperparam, **pdict}


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
    result_df.to_csv("../files/train_parameters_fit_coefficients.csv", index=False)