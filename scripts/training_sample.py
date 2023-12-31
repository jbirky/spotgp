import numpy as np
import pandas as pd

nsample = int(1e4)

tsample = {}
tsample["sim_id"] = np.arange(0, nsample)
tsample["peq"] = np.random.uniform(0.1, 30, nsample)
tsample["kappa"] = np.random.uniform(-1, 1, nsample)
tsample["inc"] = np.random.uniform(0, np.pi/2, nsample)
tsample["nspot"] = np.random.randint(1, 100, nsample)
tsample["lspot"] = np.random.uniform(0, 100, nsample)
tsample["tau"] = np.random.uniform(0, 100, nsample)
tsample["alpha_max"] = np.exp(np.random.uniform(-3, 0, nsample))

train_sample = pd.DataFrame(data=tsample)
train_sample.to_csv("../training/training_parameters.csv")