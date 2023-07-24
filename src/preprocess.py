import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

__all__ = ["PreprocessData",
           "format_data"]


def format_data(tarr, xmat, covs, nfeat):

    npts = tarr.shape[0] * xmat.shape[0]
    Ytrain = np.zeros((npts,1))
    Xtrain = np.zeros((npts,nfeat+1))

    cnt = 0
    for xind in range(xmat.shape[0]):
        for tind in range(tarr.shape[0]):
            Xtrain[cnt, 0:nfeat] = xmat[xind]
            Xtrain[cnt, nfeat] = tarr[tind]
            Ytrain[cnt] = covs[xind,tind]
            cnt += 1

    return Xtrain, Ytrain


class PreprocessData(object):

    def __init__(self, 
                 data_file="../training/train_data.npz"):

        # load unscaled matrices
        file = np.load(data_file)
        self.Xtrain = file["Xtrain"]
        self.Ytrain = file["Ytrain"]
        self.tarr = file["tarr"]

        # scale input data
        self.scaler_x = MinMaxScaler()
        self.scaler_x.fit(self.Xtrain)
        xtrain = self.scaler_x.transform(self.Xtrain)
        xtrain[:,0] = np.ones(self.Ytrain.shape[0])

        self.scaler_y = MinMaxScaler()
        self.scaler_y.fit(self.Ytrain)
        ytrain = self.scaler_y.transform(self.Ytrain)

        # scaled matrices
        self.xtrain = xtrain
        self.ytrain = ytrain

    def scale_data(self, Xdata, Ydata):

        xdata = self.scaler_x.transform(Xdata)
        ydata = self.scaler_y.transform(Ydata)

        return xdata, ydata
    
    def unscale_data(self, xdata, ydata):

        Xdata = self.scaler_x.inverse_transform(xdata)
        Ydata = self.scaler_y.inverse_transform(ydata)

        return Xdata, Ydata      

    def scale_data_tensor(self, Xdata, Ydata, device="cuda"):
        
        xdata, ydata = self.scale_data(Xdata, Ydata)

        return torch.tensor(xdata).to(device), torch.tensor(ydata).to(device)