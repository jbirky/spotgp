import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ["Feedforward",
           "Learner",
           "CustomDataset",
           "plot_prediction"]


class Feedforward(nn.Module):
    
    def __init__(self, 
                 num_inputs, 
                 num_outputs, 
                 num_hidden=5, 
                 dim_hidden=128, 
                 act=nn.Tanh(),
                 dropout=torch.nn.Dropout(.10)):
        
        super().__init__()

        self.layer_in = nn.Linear(num_inputs, dim_hidden, dtype=torch.float32)
        self.layer_out = nn.Linear(dim_hidden, num_outputs, dtype=torch.float32)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act
        self.dropout = dropout
        
    def forward(self, t):
        
        out = self.act(self.layer_in(t))
        for _, layer in enumerate(self.middle_layers):
            out = self.dropout(self.act(layer(out)))
            
        return self.layer_out(out)
    
    
class Learner(pl.LightningModule):
    
    def __init__(self, model:nn.Module,
                 optimizer=torch.optim.Adam,
                 lr=1e-2,
                 loss_fn=nn.MSELoss(),
                 trainloader=None,
                 valloader=None):
        
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.loss_fn = loss_fn
        self.trainloader = trainloader
        self.valloader = valloader
        self.losses = []
        self.val_losses = []
    
    def forward(self, x):

        return self.model(x)
    
    def training_step(self, batch, batch_idx):

        x, y = batch       
        y_hat = self.model(x.float())
        
        # Compute Loss
        loss = self.loss_fn(y_hat, y)
        self.losses.append(loss.item())
        self.log("train_loss", loss)

        return loss 
    
    def validation_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self.model(x.float())
        val_loss = self.loss_fn(y_hat, y)
        self.val_losses.append(val_loss.item())
        self.log("val_loss", val_loss, sync_dist=True)
    
    def configure_optimizers(self):

        return self.optimizer(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):

        return self.trainloader
    
    def val_dataloader(self):

        return self.valloader
    

class CustomDataset(data.Dataset):

    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Convert data and targets to torch tensors
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.targets[index], dtype=torch.float32)
        return x, y


def plot_prediction(model, x_train, y_train, mset=None, title=None):

    try:
        nnsol = model(x_train).detach().numpy().T
        tplot = x_train.t()[-1].detach().numpy()
        ytrue = y_train.detach().numpy().T
    except:
        nnsol = model(x_train).cpu().detach().numpy().T
        tplot = x_train.t()[-1].cpu().detach().numpy()
        ytrue = y_train.cpu().detach().numpy().T

    fig, ax = plt.subplots(y_train.shape[1], 1, figsize=(18, 5*y_train.shape[1]), sharex=True)
    plt.subplots_adjust(hspace=0)

    if mset is not None:
        nts = len(tplot)
        for ms in mset:
            p = ax.plot(tplot[ms*nts:(ms+1)*nts], ytrue[ms*nts:(ms+1)*nts].flatten())
            ax.plot(tplot[ms*nts:(ms+1)*nts], nnsol[ms*nts:(ms+1)*nts].flatten(), linestyle='--', color=p[0].get_color(), linewidth=4, alpha=.4)
    else:
        p = ax.plot(tplot, ytrue)
        ax.plot(tplot, nnsol, linestyle='--', color=p[0].get_color(), linewidth=4, alpha=.4)
    ax.set_xlabel("Time (scaled)", fontsize=25)

    if title is not None:
        ax.set_title(title, fontsize=25)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.close()

    return fig