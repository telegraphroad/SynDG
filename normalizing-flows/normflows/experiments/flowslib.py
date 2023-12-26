import numpy as np
import torch
import normflows as nf
from matplotlib import pyplot as plt
from tqdm import trange

# Make planar flow
def planar(K=16,dim=2):
    flows = []
    for i in range(K):
        flows.append(nf.flows.Planar((dim,)))
    return flows

# Make planar flow
def radial(K=16,dim=2):
    flows = []
    for i in range(K):
        flows.append(nf.flows.Radial((dim,)))
    return flows

# Make NICE flow
def nice(K=16,dim=2,hidden_units=64, hidden_layers=2):
    b = torch.Tensor([1 if i % dim == 0 else 0 for i in range(dim)])
    flows = []
    for i in range(K):
        lay = [dim] + [hidden_units]*hidden_layers + [dim]
        net = nf.nets.MLP(lay, init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, net)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, net)]
        flows += [nf.flows.ActNorm(dim)]
    return flows

# Make RealNVP flow
def rnvp(K=16,dim=2,hidden_units=64, hidden_layers=2,lipschitzconstrained=False,min=-1.,max=1.,func='tanh',boundtranslate=True):
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(dim)])
    flows = []
    for i in range(K):
        lay = [dim] + [hidden_units]*hidden_layers + [dim]
        s = nf.nets.MLP(lay, init_zeros=True)
        t = nf.nets.MLP(lay, init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s, lipschitzconstrained=lipschitzconstrained,min=min,max=max,func=func,boundtranslate=boundtranslate)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s, lipschitzconstrained=lipschitzconstrained,min=min,max=max,func=func,boundtranslate=boundtranslate)]
        flows += [nf.flows.ActNorm(dim)]
    return flows

# Make Neural Spline flow
def nsp(K=16,dim=2, hidden_units=64, hidden_layers=2):
    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(dim, hidden_layers, hidden_units)]
        flows += [nf.flows.LULinearPermute(dim)]
    return flows

# Make IAF flow
def iaf(K=16,dim=2, hidden_features=16):
    flows = []
    for i in range(K):
        flows.append(nf.flows.MaskedAffineAutoregressive(features=dim, hidden_features=hidden_features))
        flows.append(nf.flows.Permute(dim))
    return flows

# Make residual flow
def residual(K=16,dim=2, hidden_units=64, hidden_layers=2):
    flows = []
    for i in range(K):
        net = nf.nets.LipschitzMLP([dim] + [hidden_units] * (hidden_layers - 1) + [dim],
                                   init_zeros=True, lipschitz_const=0.9)
        flows += [nf.flows.Residual(net, reduce_memory=True)]
        flows += [nf.flows.ActNorm(dim)]
    return flows

def glow(K=16,dim=2, hidden_units=64, hidden_layers=2):
    L = 0
    input_shape = (dim,)
    n_dims = np.prod(input_shape)
    channels = 1
    hidden_channels = 256
    split_mode = 'channel'
    scale = True
    flows_ = []
    for j in range(K):
        flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1), hidden_channels,
                                    split_mode=split_mode, scale=scale)]

    return flows_

def glow_cifar10(L=3,K=16,dim=2, hidden_units=256):

    input_shape = (3, 32, 32)
    n_dims = np.prod(input_shape)
    channels = 3
    hidden_channels = hidden_units
    split_mode = 'channel'
    scale = True
    num_classes = 10

    # Set up flows, distributions and merge operations
    q0 = []
    merges = []
    flows = []
    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                        split_mode=split_mode, scale=scale)]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                            input_shape[2] // 2 ** (L - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
                            input_shape[2] // 2 ** L)
        q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]


    # Construct flow model with the multiscale architecture
    model = nf.MultiscaleFlow(q0, flows, merges)
    return model