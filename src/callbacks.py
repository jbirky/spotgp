import numpy as np
import os
from pytorch_lightning.callbacks import Callback

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
rc('xtick', labelsize=20)
rc('ytick', labelsize=20)

import sys
sys.path.append("../src")
from nnkernel import predict_autocovariance

__all__ = ["DemoPlotCallback"]


class DemoPlotCallback(Callback):
    def __init__(self, 
                 tarr=None,
                 nfeat=None,
                 scaler=None,
                 demo=None,
                 freq=10,
                 output_dir="gifs/",
                 save_name="training_demo.gif",
                 ylim=None
                 ):

        self.tarr = tarr
        self.nfeat = nfeat
        self.scaler = scaler
        self.demo = demo
        self.freq = freq
        self.output_dir = output_dir
        self.save_name = save_name
        self.epoch_counter = 0
        self.plot_epochs = []

        # Plot range limits
        if ylim is None:
            ymin = scaler.scaler_y.data_min_
            ymax = scaler.scaler_y.data_max_
            yborder = 0.1 * np.abs(ymax - ymin)
            self.ylim = [ymin - yborder, ymax + yborder]
        else:
            self.ylim = ylim

        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)

    def on_validation_epoch_end(self, trainer, pl_module):

        if self.epoch_counter % self.freq == 0:

            model = pl_module.model
            ntlags = len(self.tarr)

            # Create a plot and add it to the plots list
            fig, axs = plt.subplots(self.nfeat, 2, figsize=(18, 4*self.nfeat), sharex=True, sharey=True)
            plt.subplots_adjust(hspace=0, wspace=0)

            niter = 0
            for ii in range(self.nfeat):
                for jj in range(2):
                    idx = np.array(np.arange(ntlags*niter, ntlags*(niter+1)), dtype=int)
                    xeval, yeval = self.demo.__getitem__(idx)
                    tplot, ytrue, nnsol = predict_autocovariance(model, xeval, yeval)
                    Ytrue = self.scaler.scaler_y.inverse_transform(ytrue.reshape(-1, 1)).flatten()
                    Nnsol = self.scaler.scaler_y.inverse_transform(nnsol.reshape(-1, 1)).flatten()

                    axs[ii][jj].plot(self.tarr, Ytrue, linestyle='--', color="k")
                    axs[ii][jj].plot(self.tarr, Nnsol, color="r", linewidth=4, alpha=.4)
                    axs[ii][jj].set_xlim(min(self.tarr), max(self.tarr))
                    niter += 1

            axs[0][0].set_ylim(min(self.ylim), max(self.ylim))
            axs[-1][0].set_xlabel("Time lag", fontsize=25)
            axs[-1][1].set_xlabel("Time lag", fontsize=25)

            axs[-1][0].xaxis.set_minor_locator(AutoMinorLocator())
            axs[-1][1].xaxis.set_minor_locator(AutoMinorLocator())

            plt.suptitle(f"Epoch: {self.epoch_counter}", fontsize=25)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"epoch_{self.epoch_counter}.png"), bbox_inches="tight")
            plt.close()

            self.plot_epochs.append(self.epoch_counter)

        # Increment the epoch counter
        self.epoch_counter += 1

    def on_train_end(self, trainer, pl_module):

        # Save the current plots as a GIF
        self.save_plots_as_gif()

        self.clear_plots()

    def save_plots_as_gif(self):

        # Create a list of PIL images from the PNG files
        images = [Image.open(os.path.join(self.output_dir, f"epoch_{epoch}.png")) for epoch in self.plot_epochs]

        if len(images) > 1:
            # Save the PIL images as a GIF
            gif_name = os.path.join(self.output_dir, self.save_name+".gif")
            images[0].save(gif_name, format="gif", append_images=images[1:], save_all=True, duration=100, loop=0)

        # Close all Matplotlib figures
        plt.close('all')

    def clear_plots(self):

        # delete all plots except for final fit
        for epoch in self.plot_epochs[:len(self.plot_epochs)-1]:
            plot_file = os.path.join(self.output_dir, f"epoch_{epoch}.png")
            if os.path.exists(plot_file):
                os.remove(plot_file)

        # Rename the last file
        # plot_file = os.path.join(self.output_dir, f"epoch_{self.plot_epochs[-1]}.png")
        # new_file = os.path.join(self.output_dir, self.save_name+".png")
        # os.rename(plot_file, new_file)