import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable
import wandb
import pytorch_lightning as pl

import sys
sys.path.append("../src")
from starspot import Feedforward, Learner, plot_prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ====================================================

train_scaled = np.load("../scripts/training/train_scaled.npz")
test_scaled = np.load("../scripts/training/test_scaled.npz")

xtrain = Variable(torch.from_numpy(train_scaled["X"]).float(), requires_grad=True).to(device)
ytrain = Variable(torch.from_numpy(train_scaled["Y"]).float(), requires_grad=True).to(device)

xtest = Variable(torch.from_numpy(test_scaled["X"]).float(), requires_grad=True).to(device)
ytest = Variable(torch.from_numpy(test_scaled["Y"]).float(), requires_grad=True).to(device)

# ====================================================

wandb.init(project="starspot")
wandb.config = {
  "num_inputs": xtrain.shape[1],
  "num_outputs": ytrain.shape[1],
  "learning_rate": 0.001,
  "epochs": int(1e3),
  "batch_size": 100,
  "num_hidden": 4,
  "dim_hidden": 64,
  "activation": nn.Tanh(),
  "dropout_rate": 0.0, 
  "loss_criteria": torch.nn.MSELoss(),
  "optimizer": torch.optim.Adam,
  "device": device
}

model = Feedforward(wandb.config["num_inputs"],
                    wandb.config["num_outputs"],
                    num_hidden=wandb.config["num_hidden"], 
                    dim_hidden=wandb.config["dim_hidden"], 
                    act=wandb.config["activation"],
                    dropout=torch.nn.Dropout(wandb.config["dropout_rate"]))
model.to(device)
losses = []

# ====================================================

plot_prediction(model, xtrain, ytrain, mset=[0,1], title="Initial training fit")

train = data.TensorDataset(xtrain, ytrain)
trainloader = data.DataLoader(train, batch_size=wandb.config["batch_size"], shuffle=True)

learn = Learner(model, 
                optimizer=wandb.config["optimizer"], 
                lr=wandb.config["learning_rate"],
                loss_fn=wandb.config["loss_criteria"],
                trainloader=trainloader)
trainer = pl.Trainer(min_epochs=wandb.config["epochs"], accelerator=device)
trainer.fit(learn)

# ====================================================

epoch = round(len(losses)/wandb.config["batch_size"])

print('training loss:', torch.nn.MSELoss()(model(xtrain), ytrain).detach().numpy())
plot_prediction(model, xtrain, ytrain, mset=[0,1], title=f"Training fit (Epoch={epoch})")

print('test loss:', torch.nn.MSELoss()(model(xtest), ytest).detach().numpy())
plot_prediction(model, xtest, ytest, mset=[0,1], title=f"Test fit (Epoch={epoch})")