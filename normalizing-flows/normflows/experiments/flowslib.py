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
