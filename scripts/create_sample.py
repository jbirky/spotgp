import numpy as np
import pandas as pd

np.random.seed(100)

nsample = int(1e3)

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
train_sample.to_csv("../mox_hyak/training_parameters1.csv", index=False)