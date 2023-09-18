# %%
import itertools
import torch
import numpy as np
from matplotlib import gridspec

import normflows as nf

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from tqdm import tqdm
from ray import air, tune
from ray.air import session
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.ax import AxSearch
import numpy as np
import torch
  # Whether we should be minimizing or maximizing the objective
def get_flows(num_layers=20,w=128,l=4,IB='same'):
    flows = []
    
    for i in range(num_layers):
        # Create MLP with variable width and depth 
        if IB == 'same':
            lay = [1] + [w]*l + [2]
            param_map = nf.nets.MLP(lay, init_zeros=True)
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            flows.append(nf.flows.Permute(2, mode='swap'))
        elif IB == 'dec':
            try:
                _l = l - ((i // (num_layers // l))) + 1
                if _l == 0:
                    _l = 1
            except:
                _l = 1
            lay = [1] + [w]*l + [2]
            param_map = nf.nets.MLP(lay, init_zeros=True)
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            flows.append(nf.flows.Permute(2, mode='swap'))
        elif IB == 'inc':
            try:
                _l = (i // (num_layers // l)) + 1
                if _l == 0:
                    _l = 1
            except:
                _l = 1
            lay = [1] + [w]*_l + [2]
            param_map = nf.nets.MLP(lay, init_zeros=True)
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            flows.append(nf.flows.Permute(2, mode='swap'))
            

            
        return flows
    
def compute_average_grad_norm(model):
    total_norm = 0.0
    num_parameters = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num_parameters += 1
    return (total_norm / num_parameters) ** 0.5
# %% [markdown]
# # Setting up a flow model with a 2D Gaussian base distribution

# %%
# Set up model
def train_flow(config):
    enable_cuda = True

    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    target = nf.distributions.RingMixture(n_rings=4)
    target = nf.distributions.NealsFunnel(v1shift=3.,v2shift=0.)

    #target = nf.distributions.StudentTDistribution(2,df=2.)
    _l = target.sample(10000).median().item()

    # Define 2D Gaussian base distribution
    loc = torch.zeros((2, 2))  # 2x2 tensor filled with 0s
    scale = torch.ones((2, 2))  # 2x2 tensor filled with 1s
    p = 2.0  # Shape parameter for Gaussian

    # Create the 2-dimensional instance
    base3 = nf.distributions.base_extended.GeneralizedGaussianDistribution(loc, scale, p)

    base2 = nf.distributions.base.DiagGaussian(2)
    base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=20, rand_p=False, dim=2,loc=_l,scale=1.,p=2.,device=device,trainable_loc=False,trainable_scale=False,trainable_p=False,trainable_weights=False)
    base = nf.distributions.base.DiagGaussian(2)
    base4 = nf.distributions.GaussianMixture(n_modes=10,dim=2)
    base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=10, rand_p=False, dim=2,loc=0,scale=1.,p=2.,device=device,trainable_loc=False,trainable_scale=False,trainable_p=False,trainable_weights=False)
    base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=10, rand_p=True, dim=2,loc=0,scale=1.,p=2.,device=device,trainable_loc=True,trainable_scale=True,trainable_p=True,trainable_weights=True)
    base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=100, rand_p=True, dim=2,loc=_l,scale=1.,p=2.,device=device,trainable_loc=True,trainable_scale=True,trainable_p=True,trainable_weights=True)
    base = nf.distributions.base.DiagGaussian(2)
    base = nf.distributions.GaussianMixture(n_modes=10,dim=2)
    base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=100, rand_p=True, dim=2,loc=_l,scale=1.,p=2.,noise_scale=0.2,device=device,trainable_loc=True,trainable_scale=True,trainable_p=True,trainable_weights=True)
    base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=10, rand_p=True, dim=2,loc=_l,scale=1.,p=2.,device=device,trainable_loc=True,trainable_scale=True,trainable_p=True,trainable_weights=True)
    #base = nf.distributions.base.DiagGaussian(2)
    # Define list of flows
    num_layers = 24
    flows = []
    for i in range(num_layers):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(2, mode='swap'))
        
    # Construct flow model
    flows = get_flows(num_layers=20,w=128,l=4,IB='same')
    model = nf.NormalizingFlow(base, flows)


    model = model.to(device)

    def check_model_params(model):
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f'Parameter {name} has NaNs or infs')

    max_iter = 8000
    num_samples = 2 ** 11
    show_iter = 250
    loss_hist = np.array([])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-7)
    max_norm = 1.0
    adjust_rate = 0.01
    model.sample(10**4)
    
    best_loss = float('inf')
    best_model_params = None
    if True:
        l = config["l"]
        w = config["w"]
        num_layers = config["num_layers"]
        IB = config["IB"]

        # flows = []
        # for _ in range(num_layers):
        #     # Create MLP with variable width and depth 
        #     lay = [1] + [w]*l + [2]
        #     param_map = nf.nets.MLP(lay, init_zeros=True)
        #     flows.append(nf.flows.AffineCouplingBlock(param_map))
        #     flows.append(nf.flows.Permute(2, mode='swap'))
        flows = get_flows(num_layers=num_layers,w=w,l=l,IB=IB)
        model = nf.NormalizingFlow(base, flows)
        model = model.to(device)

        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

        loss_hist = np.array([])

        for it in tqdm(range(max_iter)):
            optimizer.zero_grad()
            x = target.sample(num_samples).to(device)
            try:
                loss = model.forward_kld(x, robust=True)

                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    avg_norm = compute_average_grad_norm(model)
                    if avg_norm > max_norm:
                        max_norm += adjust_rate
                    else:
                        max_norm -= adjust_rate
                    with torch.no_grad():
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
                session.report({"loss": loss.to('cpu').data.numpy()})
                #torch.save(model.state_dict(), "./model.pth")
                if loss < best_loss:
                    best_loss = loss
                    best_model_params = (l, w, num_layers)

                    

            except Exception as e:
                print('error',e)

search_space = {
    "w": tune.choice([32,64,128,256,512]),
    "l": tune.choice([2,3,4,5,6]),
    "num_layers": tune.choice([4,8,12,16,20,24,28]),
    "IB": tune.choice(['dec', 'inc','same']),
}
scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=1000,
        grace_period=1,
        reduction_factor=2)
# search_space = {
#     "w": tune.choice([32]),
#     "l": tune.choice([2]),
#     "num_layers": tune.choice([4]),
#     "IB": tune.choice(['dec', 'inc','same']),
# }
#resources_per_trial = {"cpu": 32, "gpu": 2}
resources_per_trial = {"cpu": 16}
tuner = tune.Tuner(
    #tune.with_resources(train_flow, resources=resources_per_trial),
    train_flow,
    param_space=search_space,
    tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            num_samples=500,
            #scheduler=scheduler,
        ),
)
analysis = tuner.fit()
# analysis = tune.run(
#     train_flow, config=search_space, resources_per_trial={'gpu': 1})
#dfs = {result.log_dir: result.metrics_dataframe for result in results}
# dfs = {result.log_dir: result.metrics_dataframe for result in analysis}

# [d.mean_accuracy.plot() for d in dfs.values()]
#torch.save(results, "./results.pth")
torch.save(analysis, "./analysis.pth")
# torch.save(dfs, "./dfs.pth")
#dfs = analysis.fetch_trial_dataframes()
#[d.loss.plot() for d in dfs.values()]
#torch.save(dfs, "./dfs2.pth")


import os

#logdir = results.get_best_result("mean_accuracy", mode="max").log_dir
# logdir = analysis.get_best_result("loss", mode="max").log_dir
# state_dict = torch.load(os.path.join(logdir, "model.pth"))

# %%
# analysis = torch.load("./analysis_old.pth")
# analysis.get_best_config("loss", mode="max")
# dfs = torch.load("./dfs2_old.pth")
# import pandas as pd
# analysis.fetch_trial_dataframes()

# # %%
# analysis.get_best_config("loss", mode="min")