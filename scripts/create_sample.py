import numpy as np
import pandas as pd

np.random.seed(100)

# nsample = int(1e4)

# tsample = {}
# tsample["sim_id"] = np.arange(0, nsample)
# tsample["peq"] = np.random.uniform(0.1, 25, nsample)
# tsample["kappa"] = np.random.uniform(-1, 1, nsample)
# tsample["inc"] = np.arccos(np.random.uniform(0, 1, nsample))
# tsample["nspot"] = np.random.randint(10, 1000, nsample)
# tsample["lspot"] = np.random.uniform(0, 50, nsample)
# tsample["tau"] = np.random.uniform(0, 50, nsample)
# tsample["alpha_max"] = np.exp(np.random.uniform(-3, -1, nsample))

# train_sample = pd.DataFrame(data=tsample)
# train_sample.to_csv("../files/training_parameters.csv", index=False)

# =================================================

# nsample = 500

# tsample = {}
# tsample["sim_id"] = np.arange(0, nsample)
# tsample["peq"] = np.random.uniform(0.1, 25, nsample)
# tsample["kappa"] = np.zeros(nsample)
# tsample["inc"] = np.ones(nsample) * np.pi/2
# tsample["nspot"] = np.random.randint(10, 1000, nsample)
# tsample["lspot"] = np.random.uniform(0, 50, nsample)
# tsample["tau"] = np.random.uniform(0, 50, nsample)
# tsample["alpha_max"] = np.exp(np.random.uniform(-3, -1, nsample))

# train_sample = pd.DataFrame(data=tsample)
# train_sample.to_csv("../files/training_parameters2.csv", index=False)

# =================================================

# nsample = int(1e3)

# tsample = {}
# tsample["sim_id"] = np.arange(0, nsample)
# tsample["peq"] = np.random.uniform(0.1, 25, nsample)
# tsample["kappa"] = np.random.uniform(-1, 1, nsample)
# tsample["inc"] = np.arccos(np.random.uniform(0, 1, nsample))
# tsample["nspot"] = np.random.randint(10, 1000, nsample)
# tsample["lspot"] = np.random.uniform(0, 50, nsample)
# tsample["tau"] = np.random.uniform(0, 50, nsample)
# tsample["alpha_max"] = 10**(np.random.uniform(-3, -1, nsample))

# train_sample = pd.DataFrame(data=tsample)
# train_sample.to_csv("../files/training_parameters3.csv", index=False)

# =================================================

nsample = int(1e3)

tsample = {}
tsample["sim_id"] = np.arange(0, nsample)
tsample["peq"] = np.random.uniform(0.1, 25, nsample)
tsample["kappa"] = np.random.uniform(-1, 1, nsample)
tsample["inc"] = np.arccos(np.random.uniform(0, 1, nsample))
tsample["nspot"] = np.random.randint(10, 100, nsample)
tsample["lspot"] = np.random.uniform(0, 50, nsample)
tsample["tau"] = np.random.uniform(0, 50, nsample)
tsample["alpha_max"] = 10**(np.random.uniform(-3, -1, nsample))

train_sample = pd.DataFrame(data=tsample)
train_sample.to_csv("../files/training_parameters4.csv", index=False)