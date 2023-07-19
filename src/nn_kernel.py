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
import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

__all__ = ["Feedforward",
           "Learner",
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

        self.layer_in = nn.Linear(num_inputs, dim_hidden)
        self.layer_out = nn.Linear(dim_hidden, num_outputs)

        num_middle = num_hidden - 1
        self.middle_layers = nn.ModuleList(
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(num_middle)]
        )
        self.act = act
        self.dropout = dropout
        
        self.apply(self._init_weights)
        
        
    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.5)
            if module.bias is not None:
                module.bias.data.zero_()

        
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
                 trainloader=None):
        
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.loss_fn = loss_fn
        self.trainloader = trainloader
        self.losses = []
    
    def forward(self, x):

        return self.model(x)
    
    def training_step(self, batch, batch_idx):

        x, y = batch       
        y_hat = self.model(x)
        
        # Compute Loss
        loss = self.loss_fn(y_hat, y)
        self.losses.append(loss)
    
        return {'loss': loss}   
    
    def configure_optimizers(self):

        return self.optimizer(self.model.parameters(), lr=self.lr)

    def train_dataloader(self):

        return self.trainloader
    
    
def plot_prediction(model, x_train, y_train, mset=None, title=None):

    nnsol = model(x_train).detach().numpy().T
    tplot = xtrain.t()[-1].detach().numpy()
    ytrue = y_train.detach().numpy().T

    fig, axs = plt.subplots(y_train.shape[1], 1, figsize=(18, 5*y_train.shape[1]), sharex=True)
    plt.subplots_adjust(hspace=0)
    for ii, ax in enumerate(axs):
        if mset is not None:
            nts = len(teval)
            for ms in mset:
                p = ax.plot(tplot[ms*nts:(ms+1)*nts], ytrue[ii][ms*nts:(ms+1)*nts])
                ax.plot(tplot[ms*nts:(ms+1)*nts], nnsol[ii][ms*nts:(ms+1)*nts], linestyle='--', color=p[0].get_color(), linewidth=4, alpha=.4)
        else:
            p = ax.plot(tplot, ytrue[ii])
            ax.plot(tplot, nnsol[ii], linestyle='--', color=p[0].get_color(), linewidth=4, alpha=.4)
        ax.set_ylabel(r"$S_{%s}(t)$"%(ii), fontsize=25)
    ax.set_xlabel("Time (scaled)", fontsize=25)
    if title is not None:
        axs[0].set_title(title, fontsize=25)
    axs[-1].xaxis.set_minor_locator(AutoMinorLocator())
    plt.close()

    return fig