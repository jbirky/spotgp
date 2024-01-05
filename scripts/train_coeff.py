import numpy as np
import pandas as pd
import tqdm
import multiprocessing as mp
import os
import argparse
import h5py
import shutil
import sys
sys.path.append("../src")

# not working on hyak. override any matplotlib.pyplot imports
sys.modules["matplotlib.pyplot"] = None
import kernel_fourier as kf

# ================================================
# Configuration 

ncore = mp.cpu_count()
print("number of cores:", ncore)

parser = argparse.ArgumentParser()
parser.add_argument("--trial", type=int, required=True)
parser.add_argument("--nsample", type=int, default=int(1e3))
parser.add_argument("--file_dir", type=str, default="../mox_hyak")
parser.add_argument("--save_outputs", type=bool, default=False)
args = parser.parse_args()

trial = args.trial
file_dir = args.file_dir
nsample = args.nsample
save_outputs = args.save_outputs

save_file_initial = f"{file_dir}/training_parameters{trial}.csv"
save_file_final = f"{file_dir}/train_parameters_fit_coefficients{trial}.csv"
data_dir = f"{file_dir}/results/data{trial}/"

# create necessary directories
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
if save_outputs == True:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

# ================================================
# Create training sample

# set random seed
np.random.seed(trial)

tsample = {}
tsample["sim_id"] = np.arange(0, nsample)
tsample["peq"] = np.random.uniform(1, 25, nsample)
tsample["kappa"] = np.random.uniform(-1, 1, nsample)
tsample["inc"] = np.pi/2 * np.ones(nsample)
tsample["nspot"] = 10 * np.ones(nsample)
tsample["lspot"] = np.random.uniform(0, 50, nsample)
tsample["tau"] = np.random.uniform(0, 50, nsample)
tsample["alpha_max"] = 10**(np.random.uniform(-3, -1, nsample))

train_sample = pd.DataFrame(data=tsample)
train_sample.to_csv(save_file_initial, index=False)

df = train_sample[['peq', 'kappa', 'inc', 'nspot', 'lspot', 'tau', 'alpha_max']]
train_dict = df.to_dict(orient='records')

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
    train_kernel = kf.TrainKernel(hyperparam, tsim=1000, tsamp=0.1, nsim=5e2, tcut=200, 
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
                 autocorr=train_kernel.ydata)

    summary = {"sim_id": ii, "ymse": ymse, "pmse": pmse}

    return {**summary, **hyperparam, **pdict}


# ================================================

def save_npz_to_hdf5(npz_folder, hdf5_filename, N):
    """
    Reads data from multiple .npz files and saves it to a single compressed HDF5 file.

    :param npz_folder: Folder containing the .npz files.
    :param hdf5_filename: Name of the output HDF5 file.
    :param N: Number of .npz files to read.
    """
    with h5py.File(hdf5_filename, 'w') as hdf:
        for ii in range(N):
            str_index = f"{ii}".rjust(5, "0")
            npz_file = os.path.join(npz_folder, f'sim_{str_index}.npz')
            with np.load(npz_file) as data:
                hdf.create_dataset(f'sim_{str_index}', data=data["autocorr"], compression='gzip')


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
    result_df.to_csv(save_file_final, index=False)

    # compress saved data files
    compress_file = f"{file_dir}/training_data{trial}.hdf5"
    print(f"compressing data to {compress_file}...")
    save_npz_to_hdf5(data_dir, compress_file, nsample)

    shutil.rmtree(data_dir)

    print("job complete")