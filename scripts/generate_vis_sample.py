import numpy as np 
import pandas as pd
import os
import sys
sys.path.append("../src")
import starspot
from preprocess import format_data

np.random.seed(10)
os.nice(10)

savedir = "../training"
nsim = int(1e4)         # number of sims averaged for each training sample
tsim = 28
tsamp = 0.02
tarr = np.arange(0, tsim, tsamp)
feat = ["peq", "kappa", "inc", "nspot"]

thetas = np.array([
    [1.0, 0.0, np.pi/2, 10],
    [10.0, 0.0, np.pi/2, 10],
    [5.0, 0.0, np.pi/2, 10],
    [5.0, 0.5, np.pi/2, 10],
    [5.0, 0.0, np.pi/6, 10],
    [5.0, 0.0, np.pi/4, 10],
    [5.0, 0.0, np.pi/2, 5],
    [5.0, 0.0, np.pi/2, 15],
])

# compute covariances
covs = starspot.generate_training_sample(thetas,  
                                         tem=2, 
                                         lspot=10,
                                         tdec=2, 
                                         alpha_max=0.1, 
                                         fspot=0, 
                                         long=[0,2*np.pi], 
                                         lat=[0,np.pi], 
                                         tsim=tsim, 
                                         tsamp=tsamp,
                                         limb_darkening=True,
                                         ncore=len(thetas))

# format data into training matrices
Xtrain, Ytrain = format_data(tarr, thetas, covs, len(feat))

# save training products
df = pd.DataFrame(data=thetas, columns=feat)
df.to_csv(os.path.join(savedir, "vis_features.csv"))
np.savez(os.path.join(savedir, "vis_covariances.npz"), covs=covs)
np.savez(os.path.join(savedir, "vis_data.npz"), tarr=tarr, Xtrain=Xtrain, Ytrain=Ytrain)