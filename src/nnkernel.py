import torch
import torch.nn as nn
import torch.utils.data as data
import pytorch_lightning as pl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__all__ = ["Feedforward",
           "Learner",
           "CustomDataset",
           "predict_autocovariance",
           "plot_prediction"]


class Feedforward(nn.Module):
    """
    Feedforward neural network module.

    Args:
        num_inputs (int): Number of input features.
        num_outputs (int): Number of output features.
        num_hidden (int, optional): Number of hidden layers (excluding the input and output layers). Default is 5.
        dim_hidden (int, optional): Number of units in each hidden layer. Default is 128.
        act (torch.nn.Module, optional): Activation function to be used in the hidden layers. Default is `torch.nn.Tanh()`.
        dropout (torch.nn.Module, optional): Dropout layer to be applied after each hidden layer. Default is `torch.nn.Dropout(0.10)`.

    Attributes:
        act (torch.nn.Module): Activation function for the hidden layers.
        dropout (torch.nn.Module): Dropout layer applied after each hidden layer.
        layer_in (torch.nn.Linear): Input layer of the network.
        layer_out (torch.nn.Linear): Output layer of the network.
        hidden_layers (torch.nn.ModuleList): List containing hidden layers.

    Methods:
        initialize_weights: Initializes the weights of linear layers using Xavier initialization.

    Note:
        This class defines a feedforward neural network with a flexible number of hidden layers.

    """
    def __init__(self, 
                 num_inputs, 
                 num_outputs, 
                 num_hidden=5, 
                 dim_hidden=128, 
                 act=nn.Tanh(),
                 dropout=torch.nn.Dropout(.10)):
        
        super().__init__()

        self.act = act
        self.dropout = dropout

        self.layer_in = nn.Linear(num_inputs, dim_hidden, dtype=torch.float32)
        self.layer_out = nn.Linear(dim_hidden, num_outputs, dtype=torch.float32)

        # Create hidden layers dynamically based on num_hidden
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_hidden - 1):
            self.hidden_layers.append(nn.Linear(dim_hidden, dim_hidden, dtype=torch.float32))

        # Apply Xavier initialization to all layers
        self.apply(self.initialize_weights)

    def initialize_weights(self, layer):
        """
        Initializes the weights of linear layers using Xavier initialization.

        Args:
            layer (torch.nn.Module): Linear layer.

        """
        if isinstance(layer, nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                layer.bias.data.zero_()
        
    def forward(self, t):
        """
        Performs forward pass through the neural network.

        Args:
            t (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        out = self.act(self.layer_in(t))
        for _, layer in enumerate(self.hidden_layers):
            out = self.dropout(self.act(layer(out)))
            
        return self.layer_out(out)
    
    
class Learner(pl.LightningModule):
    """
    PyTorch Lightning module for training the Feedforward neural network.

    Args:
        model (torch.nn.Module): The feedforward neural network model.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        lr (float, optional): Learning rate for the optimizer. Default is 1e-3.
        loss_fn (torch.nn.Module, optional): Loss function for training. Default is `torch.nn.MSELoss()`.
        trainloader (torch.utils.data.DataLoader): DataLoader for training data.
        valloader (torch.utils.data.DataLoader, optional): DataLoader for validation data. Default is None.

    Attributes:
        model (torch.nn.Module): The feedforward neural network model.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        lr (float): Learning rate for the optimizer.
        loss_fn (torch.nn.Module): Loss function for training.
        trainloader (torch.utils.data.DataLoader): DataLoader for training data.
        valloader (torch.utils.data.DataLoader, optional): DataLoader for validation data.
        losses (list): List to store training losses.
        val_losses (list): List to store validation losses.

    Methods:
        training_step: Performs a single training step.
        validation_step: Performs a single validation step.
        configure_optimizers: Configures the optimizer for training.
        train_dataloader: Returns the DataLoader for training data.
        val_dataloader: Returns the DataLoader for validation data (if provided).

    Note:
        This class defines a PyTorch Lightning module for training the Feedforward neural network.

    """
    def __init__(self, model:nn.Module,
                 optimizer=torch.optim.Adam,
                 lr=1e-3,
                 scheduler=None,
                 loss_fn=nn.MSELoss(),
                 trainloader=None,
                 valloader=None):
        
        super().__init__()

        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.trainloader = trainloader
        self.valloader = valloader
        self.losses = []
        self.val_losses = []
    
    def forward(self, x):
        """
        Performs forward pass through the neural network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.

        """
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch (tuple): Batch of input data and target labels.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Training loss.

        """
        x, y = batch       
        y_hat = self.model(x.float())
        
        # Compute Loss
        loss = self.loss_fn(y_hat, y)
        self.losses.append(loss.item())
        self.log("train_loss", loss)

        # Step the scheduler after each training step
        if self.scheduler is not None:
            self.scheduler.step(loss)

        return loss 
    
    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.

        Args:
            batch (tuple): Batch of input data and target labels.
            batch_idx (int): Index of the current batch.

        """
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
    """
    Custom Dataset class for handling data and targets.

    Args:
        data (list or numpy.ndarray): List or numpy array of data samples.
        targets (list or numpy.ndarray): List or numpy array of target labels.

    Attributes:
        data (torch.Tensor): Tensor containing data samples.
        targets (torch.Tensor): Tensor containing target labels.

    Methods:
        __len__: Returns the number of data samples.
        __getitem__: Returns the data sample and corresponding target label at the given index.

    Note:
        This class defines a custom dataset for handling data and target labels.

    """
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the data sample and corresponding target label at the given index.

        Args:
            index (int): Index of the data sample.

        Returns:
            torch.Tensor: Data sample.
            torch.Tensor: Target label.

        """
        # Convert data and targets to torch tensors
        x = torch.tensor(self.data[index], dtype=torch.float32)
        y = torch.tensor(self.targets[index], dtype=torch.float32)
        return x, y


def predict_autocovariance(model, x_train, y_train):

    try:
        nnsol = model(x_train).detach().numpy().T.flatten()
        tplot = x_train.t()[-1].detach().numpy().flatten()
        ytrue = y_train.detach().numpy().T.flatten()
    except:
        nnsol = model(x_train).cpu().detach().numpy().T.flatten()
        tplot = x_train.t()[-1].cpu().detach().numpy().flatten()
        ytrue = y_train.cpu().detach().numpy().T.flatten()

    return tplot, ytrue, nnsol


def plot_prediction(tplot, ytrue, nnsol, title=None):

    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    rc('xtick', labelsize=20)
    rc('ytick', labelsize=20)

    fig, ax = plt.subplots(1, 1, figsize=(18, 5), sharex=True)
    plt.subplots_adjust(hspace=0)

    p = ax.plot(tplot, ytrue)
    ax.plot(tplot, nnsol, linestyle='--', color=p[0].get_color(), linewidth=4, alpha=.4)
    ax.set_xlim(min(tplot), max(tplot))
    ax.set_xlabel("Time (scaled)", fontsize=25)

    if title is not None:
        ax.set_title(title, fontsize=25)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    plt.close()

    return fig