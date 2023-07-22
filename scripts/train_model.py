import numpy as np
import torch
import random
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

import sys
sys.path.append("../src")
from nnkernel import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--reload", dest="reload", action='store', type=bool, default=False)
# args = parser.parse_args()
# print(args.reload)
reload = False

# ====================================================

# Load training dataset
train_scaled = np.load("../training/test_scaled.npz")
dataset = CustomDataset(train_scaled["X"], train_scaled["Y"])

# Training configuration
config = {
  "num_inputs": train_scaled["X"].shape[1],
  "num_outputs": train_scaled["Y"].shape[1],
  "ncovs": 1000,
  "ntlags": 1400,
  "train_fraction": 0.8,
  "epochs": 100,
  "batch_size": 128,
  "num_hidden": 3,
  "dim_hidden": 32,
  "activation": torch.nn.Tanh(),
  "dropout_rate": 0.0, 
  "loss_criteria": torch.nn.MSELoss(),
  "optimizer": torch.optim.Adam,
  "learning_rate": 0.1,
  "device": device,
  "ncpu": 32,
  "save_weights": "weights/model_state",
}

# ====================================================

# Split the dataset into train and validation sets
train_size = int(config["train_fraction"] * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader instances to handle batching
train_loader = DataLoader(train_dataset, 
                          batch_size=config["batch_size"], 
                          num_workers=config["ncpu"],
                          shuffle=True)

val_loader = DataLoader(val_dataset, 
                        batch_size=config["batch_size"],
                        num_workers=config["ncpu"])

# ====================================================

# Initialize network
model = Feedforward(config["num_inputs"],
                    config["num_outputs"],
                    num_hidden=config["num_hidden"], 
                    dim_hidden=config["dim_hidden"], 
                    act=config["activation"],
                    dropout=torch.nn.Dropout(config["dropout_rate"]))
model.to(device)

# if (reload == True) and (os.path.isfile(config["save_weights"])):
#   print(f"Reloading model: {config['save_weights']}")
#   model.load_state_dict(torch.load(config["save_weights"]))

# ====================================================

# Learning rate scheduler
optimizer = config["optimizer"](model.parameters(), lr=config["learning_rate"])
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True)

# Initialize lightning learner module
learn = Learner(model, 
                optimizer=config["optimizer"], 
                lr=config["learning_rate"],
                scheduler=scheduler,
                loss_fn=config["loss_criteria"],
                trainloader=train_loader,
                valloader=val_loader)
 
# Initialize wandb logger
wandb_logger = WandbLogger(project="starspot", config=config)

# Train model
trainer = Trainer(max_epochs=int(config["epochs"]),
                  accelerator=device.type,
                  devices=torch.cuda.device_count(),
                  strategy=DDPStrategy(find_unused_parameters=False),
                  logger=wandb_logger)
trainer.fit(learn)

# save model weights
torch.save(model.state_dict(), config["save_weights"])

# ====================================================

nplots = 10
plot_ids = random.sample(list(np.arange(0,config["ncovs"])), nplots)

fig, ax = plt.subplots(nplots, 1, figsize=(12, 4*nplots), sharex=True)
plt.subplots_adjust(hspace=0)

for ii in range(nplots):
  idx = np.array(np.arange(config["ntlags"]*plot_ids[ii], config["ntlags"]*(plot_ids[ii]+1)), dtype=int)
  xeval, yeval = dataset.__getitem__(idx)
  tplot, ytrue, nnsol = predict_autocovariance(model, xeval, yeval)

  ax[ii].plot(tplot, ytrue, linestyle='--', color="k")
  ax[ii].plot(tplot, nnsol, color="r", linewidth=4, alpha=.4)
  ax[ii].set_xlim(min(tplot), max(tplot))
  ax[ii].set_xlabel("Time (scaled)", fontsize=25)

ax[-1].xaxis.set_minor_locator(AutoMinorLocator())
plt.close()
fig.savefig(f"plots/{wandb_logger.name}_{wandb_logger.experiment.name}.png", bbox_inches="tight")