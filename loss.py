import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pypher.pypher import psf2otf
from tqdm import tqdm

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, output, target):
        return F.mse_loss(output, target)

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, output, *args):
        # check if GPU is available, otherwise use CPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        fx = torch.from_numpy(psf2otf(np.array([[-1., 1.]]), output.shape[-2:])).to(device)
        fy = torch.from_numpy(psf2otf(np.array([[-1.], [1.]]), output.shape[-2:])).to(device)
        fxy = torch.stack((fx, fy), axis=0)

        oper = lambda x: torch.stack([torch.fft.ifft2(torch.fft.fft2(x) * fxy[i, :, :]) for i in range(fxy.shape[0])], dim=0)

        run_loss = .0
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                run_loss += torch.sum(torch.norm(oper(output[i, j, ...]), dim=0))

        denom = float(output.shape[0] * output.shape[1] * output.shape[2] * output.shape[3])
        return run_loss / denom

class TCLoss(nn.Module):
    def __init__(self):
        super(TCLoss, self).__init__()

    def forward(self, output, *args):
        # -- check if GPU is available, otherwise use CPU
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        output = output.to(device)

        # -- Compute difference between consecutive frames
        diff = output[:, 1:, ...] - output[:, :-1, ...]  # -- shape: (B, T-1, H, W)

        # -- Compute L2 norm (or other norm) across channels and spatial dimensions
        # -- run_loss = torch.norm(diff, dim=2).pow(2).sum()  # L2 over channels
        run_loss = diff.pow(2).sum()

        denom = float((output.shape[0]) * (output.shape[1] - 1) * output.shape[2] * output.shape[3])
        return run_loss / denom

