import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn as nn

class VDMADE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim * 2),
        )

    def forward(self, x):
        output = self.net(x)
        return output[..., :self.input_dim], output[..., self.input_dim:]


class VDMAF(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.made = VDMADE(input_dim, hidden_dim)

    def forward(self, z, ldj, reverse=False):
        shift, log_scale = self.made(z)
        scale = torch.exp(log_scale)

        if not reverse:
            z = scale * z + shift
            ldj += log_scale.sum(dim=-1)
        else:
            z = (z - shift) / scale
            ldj -= log_scale.sum(dim=-1)

        return z, ldj

class VDMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


class VDShiftScaleFlow(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.scale_net = VDMLP(input_dim, hidden_dim, 1)
        self.shift_net = VDMLP(input_dim, hidden_dim, 1)

    def forward(self, z, ldj, reverse=False):
        scale = self.scale_net(z)
        shift = self.shift_net(z)

        if not reverse:
            z = z * torch.exp(scale) + shift
            ldj += torch.sum(scale, dim=1)
        else:
            z = (z - shift) * torch.exp(-scale)
            ldj -= torch.sum(scale, dim=1)

        return z, ldj

class Dequantizer(nn.Module):

    def __init__(self, alpha=1e-5, num_cat=256):
        """
        Inputs:
            alpha - small constant that is used to scale the original input.
            num_cat - Number of possible categories
        """
        super().__init__()
        self.alpha = alpha
        self.num_cat = num_cat

    def forward(self, z, ldj, reverse=False):
        if not reverse:
            z, ldj = self.dequant(z, ldj)
            z, ldj = self.sigmoid(z, ldj, reverse=True)
        else:
            z, ldj = self.sigmoid(z, ldj, reverse=False)
            z = z * self.num_cat
            ldj += np.log(self.num_cat) * np.prod(z.shape[1:])
            z = torch.floor(z).clamp(min=0, max=self.num_cat-1).to(torch.int32)
        return z, ldj

    def sigmoid(self, z, ldj, reverse=False):
        # Applies an invertible sigmoid transformation
        if not reverse:
            ldj += (-z-2*F.softplus(-z)).sum(dim=1)
            z = torch.sigmoid(z)
            # Reversing scaling for numerical stability
            ldj -= np.log(1 - self.alpha) * np.prod(z.shape[1:])
            z = (z - 0.5 * self.alpha) / (1 - self.alpha)
        else:
            z = z * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            ldj += np.log(1 - self.alpha) * np.prod(z.shape[1:])
            ldj += (-torch.log(z) - torch.log(1-z)).sum(dim=1)
            z = torch.log(z) - torch.log(1-z)
        return z, ldj

    def dequant(self, z, ldj):
        # Transform discrete values to continuous volumes
        z = z.to(torch.float32)
        z = z + torch.rand_like(z).detach()
        z = z / self.num_cat
        ldj -= np.log(self.num_cat) * np.prod(z.shape[1:])
        return z, ldj

class VariationalDequantizer(Dequantizer):

    def __init__(self, var_flows, alpha=1e-5, num_cat=256):
        """
        Inputs:
            var_flows - A list of flow transformations to use for modeling q(u|x)
            alpha - Small constant, see Dequantization for details
            num_cat - Number of possible categories
        """
        super().__init__(alpha=alpha, num_cat=num_cat)
        self.flows = nn.ModuleList(var_flows)

    def dequant(self, z, ldj):
        z = z.to(torch.float32)

        deq_noise = torch.rand_like(z).detach()
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=True)
        for flow in self.flows:
            deq_noise, ldj = flow(deq_noise, ldj, reverse=False)
        deq_noise, ldj = self.sigmoid(deq_noise, ldj, reverse=False)

        z = (z + deq_noise) / self.num_cat
        ldj -= np.log(self.num_cat) * np.prod(z.shape[1:])
        return z, ldj

