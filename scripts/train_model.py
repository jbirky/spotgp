import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

import os
import sys
sys.path.append("../src")
from nnkernel import Feedforward, Learner, CustomDataset, plot_prediction

api_key = os.environ.get("WANDB_API_KEY")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# ====================================================

# Load training dataset
train_scaled = np.load("../training/test_scaled.npz")
dataset = CustomDataset(train_scaled["X"], train_scaled["Y"])

# Training configuration
config = {
  "num_inputs": train_scaled["X"].shape[1],
  "num_outputs": train_scaled["Y"].shape[1],
  "train_fraction": 0.8,
  "learning_rate": 0.001,
  "epochs": 1,
  "batch_size": 256,
  "num_hidden": 3,
  "dim_hidden": 32,
  "activation": torch.nn.Tanh(),
  "dropout_rate": 0.0, 
  "loss_criteria": torch.nn.MSELoss(),
  "optimizer": torch.optim.Adam,
  "device": device,
  "ncpu": 32,
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
                  logger=WandbLogger())
trainer.fit(learn)

# save model weights
torch.save(model.state_dict(), "weights/model_state")

xeval, yeval = dataset.__getitem__([0])
fig = plot_prediction(model, xeval, yeval, title=wandb.run.name)
fig.savefig(f"plots/{wandb.run.name}.png")