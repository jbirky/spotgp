import numpy as np
import torch
import wandb
import random
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

import os
import sys
sys.path.append("../src")
from nnkernel import *

api_key = os.environ.get("WANDB_API_KEY")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--reload", dest="reload", action='store', type=bool, default=False)
# parser.add_argument("epochs", type=int, required=False, default=10)
# parser.add_argument("learning_rate", type=float, required=False, default=1e-3)
# args = parser.parse_args()
# print(args.reload)
reload = True

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
  "epochs": 50,
  "batch_size": 1024,
  "num_hidden": 3,
  "dim_hidden": 32,
  "activation": torch.nn.Tanh(),
  "dropout_rate": 0.0, 
  "loss_criteria": torch.nn.MSELoss(),
  "optimizer": torch.optim.Adam,
  "learning_rate": 0.01,
  "device": device,
  "ncpu": 32,
  "save_weights": "weights/model_state",
}

# Initialize wandb log
wandb.init(project="starspot", config=config, save_code=True)

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

model = Feedforward(config["num_inputs"],
                    config["num_outputs"],
                    num_hidden=config["num_hidden"], 
                    dim_hidden=config["dim_hidden"], 
                    act=config["activation"],
                    dropout=torch.nn.Dropout(config["dropout_rate"]))
model.to(device)

if (reload == True) and (os.path.isfile(config["save_weights"])):
  print(f"Reloading model: {config['save_weights']}")
  model.load_state_dict(torch.load(config["save_weights"]))

# ====================================================

# Initialize lightning learner module
learn = Learner(model, 
                optimizer=config["optimizer"], 
                lr=config["learning_rate"],
                loss_fn=config["loss_criteria"],
                trainloader=train_loader,
                valloader=val_loader)

# Train model 
trainer = Trainer(max_epochs=int(config["epochs"]),
                  accelerator=device.type,
                  devices=torch.cuda.device_count(),
                  strategy=DDPStrategy(find_unused_parameters=False),
                  logger=WandbLogger(wandb_run=wandb.run))
trainer.fit(learn)

# save model weights
torch.save(model.state_dict(), config["save_weights"])

# ====================================================

nplots = 5
plot_ids = random.sample(list(np.arange(0,config["ncovs"])), nplots)

fig, ax = plt.subplots(nplots, 1, figsize=(18, 5*nplots), sharex=True)
plt.subplots_adjust(hspace=0)

for ii in range(nplots):
  idx = np.array(np.arange(config["ntlags"]*plot_ids[ii], (config["ntlags"]+1)*plot_ids[ii]), dtype=int)
  xeval, yeval = dataset.__getitem__(idx)
  tplot, ytrue, nnsol = predict_autocovariance(model, xeval, yeval)

  p = ax[ii].plot(tplot, ytrue)
  ax[ii].plot(tplot, nnsol, linestyle='--', color=p[0].get_color(), linewidth=4, alpha=.4)
  ax[ii].set_xlim(min(tplot), max(tplot))
  ax[ii].set_xlabel("Time (scaled)", fontsize=25)

ax[-1].xaxis.set_minor_locator(AutoMinorLocator())

fig.savefig(f"plots/{wandb.run.name}.png", bbox_inches="tight")