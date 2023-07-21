import numpy as np 
import pandas as pd
import os
import sys
sys.path.append("../src")
from sklearn.decomposition import PCA
import starspot

np.random.seed(10)

savedir = "training"
ntrain = int(1e3)
nsim = int(5e3)
ncore = 18

# generate table of random parameters
th1 = np.random.uniform(1.0, 10.0, ntrain)
th2 = np.random.uniform(0., 0.7, ntrain)
th3 = np.random.uniform(0., np.pi/2, ntrain)
th4 = np.array(np.round(np.random.uniform(1, 20, ntrain), 0), dtype=int)
thetas = np.array([th1, th2, th3, th4]).T

features = ["peq", "kappa", "inc", "nspot"]
feat_df = pd.DataFrame(data=thetas, columns=features)

# compute covariances
covs = starspot.generate_training_sample(thetas, ncore=ncore, tem=2, tdec=5, lspot=10)

# run PCA on covariances
pca = PCA(n_components=3)
pc = pca.fit_transform(covs)
pca_df = pd.DataFrame(data=pc, columns=["pc1", "pc2", "pc3"])
df = pd.concat([feat_df, pca_df], axis=1)

# save training products
df.to_csv(os.path.join(savedir, "test_features.csv"))
np.savez(os.path.join(savedir, "test_covariances.npz"), covs)