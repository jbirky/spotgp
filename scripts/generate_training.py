import numpy as np 
import pandas as pd
import os
import sys
sys.path.append("../src")
from sklearn.decomposition import PCA
import starspot

np.random.seed(10)
os.nice(10)

savedir = "training"
ntrain = int(1e3)       # number of training samples
nsim = int(1e4)         # number of sims averaged for each training sample
ncore = 48

# generate table of random parameters
dtable = {}
dtable["peq"] = np.random.uniform(1.0, 10.0, ntrain)
dtable["kappa"] = np.random.uniform(0., 0.7, ntrain)
dtable["inc"] = np.random.uniform(0., np.pi/2, ntrain)
dtable["nspot"] = np.array(np.round(np.random.uniform(1, 20, ntrain), 0), dtype=int)
feat_df = pd.DataFrame(data=dtable)
thetas = feat_df.values.T

# compute covariances
covs = starspot.generate_training_sample(thetas,  
                                         tem=2, 
                                         tdec=2, 
                                         alpha_max=0.1, 
                                         fspot=0, 
                                         lspot=10,
                                         long=[0,2*np.pi], 
                                         lat=[0,np.pi], 
                                         tsim=28, 
                                         tsamp=0.02,
                                         limb_darkening=True,
                                         ncore=ncore)

# run PCA on covariances
pca = PCA(n_components=5)
pc = pca.fit_transform(covs)
pca_df = pd.DataFrame(data=pc, columns=["pc1", "pc2", "pc3", "pc4", "pc5"])
df = pd.concat([feat_df, pca_df], axis=1)

# save training products
df.to_csv(os.path.join(savedir, "test_features.csv"))
np.savez(os.path.join(savedir, "test_covariances.npz"), covs)