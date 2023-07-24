import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import RichProgressBar

import os
import sys
sys.path.append("../src")
from nnkernel import *
from preprocess import PreprocessData
from callbacks import DemoPlotCallback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

reload = False

# ====================================================

# Load training dataset
scaler = PreprocessData(data_file="../training/train_data.npz")
xtrain, ytrain = scaler.xtrain, scaler.ytrain
dataset = CustomDataset(xtrain, ytrain)

# Load demo dataset for visualization
vis = PreprocessData(data_file="../training/vis_data.npz")
xvis, yvis = scaler.scale_data_tensor(vis.Xtrain, vis.Ytrain)
demo = CustomDataset(xvis, yvis)
tarr = vis.tarr

# Training configuration
config = {
  "num_inputs": xtrain.shape[1],
  "num_outputs": ytrain.shape[1],
  "ncovs": 1000,
  "ntlags": 1400,
  "train_fraction": 0.8,
  "epochs": 10,
  "batch_size": 512,
  "num_hidden": 4,
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

# ====================================================

# Initialize lightning learner module
learn = Learner(model, 
                optimizer=config["optimizer"], 
                lr=config["learning_rate"],
                loss_fn=config["loss_criteria"],
                trainloader=train_loader,
                valloader=val_loader)
 
# Initialize wandb logger
wandb.login()
wandb_logger = WandbLogger(project="starspot", config=config, log_model=True)
log_name = str(wandb_logger.experiment.name)

# Custom callbacks
demo_plots_callback = DemoPlotCallback(tarr=tarr, 
                                       nfeat=config["num_inputs"] - 1,
                                       scaler=scaler,
                                       demo=demo,
                                       freq=1,
                                       output_dir="gifs/",
                                       save_name=log_name,
                                       ylim=[-2e-5, 5e-5])

# Train model
trainer = Trainer(max_epochs=int(config["epochs"]),
                  accelerator=device.type,
                  devices=torch.cuda.device_count(),
                  strategy=DDPStrategy(find_unused_parameters=False),
                  logger=wandb_logger,
                  callbacks=[RichProgressBar(), demo_plots_callback])
trainer.fit(learn)

wandb_logger.experiment.finish()

# save model weights
torch.save(model.state_dict(), config["save_weights"])