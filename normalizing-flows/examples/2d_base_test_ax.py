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
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.service.utils.report_utils import exp_to_df
from ax.utils.notebook.plotting import init_notebook_plotting, render
from ax.utils.tutorials.cnn_utils import evaluate, load_mnist, train

init_notebook_plotting()
dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ax_client = AxClient()

    # "w": tune.choice([32,64,128,256,512]),
    # "l": tune.choice([2,3,4,5,6]),
    # "num_layers": tune.choice([4,8,12,16,20,24,28]),
    # "IB": tune.choice(['dec', 'inc','same']),

ax_client.create_experiment(
    name="tune_flow_on_nealsfunnel",  # The name of the experiment.
    parameters=[
        {
            "name": "w",  # The name of the parameter.
            "type": "choice",  # The type of the parameter ("range", "choice" or "fixed").
            "values": [32,64,128,192,256,378,512],  # The bounds for range parameters. 
        },
        {
            "name": "l",  # The name of the parameter.
            "type": "choice",  # The type of the parameter ("range", "choice" or "fixed").
            "values": [2,3,4,5,6,7],  # The bounds for range parameters. 
        },
        {
            "name": "num_layers",  # The name of the parameter.
            "type": "choice",  # The type of the parameter ("range", "choice" or "fixed").
            "values": [4,8,12,16,20,24,28],  # The bounds for range parameters. 
        },
        {
            "name": "IB",  # The name of the parameter.
            "type": "choice",  # The type of the parameter ("range", "choice" or "fixed").
            "values": ['dec', 'inc','same'],  # The bounds for range parameters. 
        },
    ],
    objectives={"loss": ObjectiveProperties(minimize=True)},  # The objective name and minimization setting.
    # parameter_constraints: Optional, a list of strings of form "p1 >= p2" or "p1 + p2 <= some_bound".
    # outcome_constraints: Optional, a list of strings of form "constrained_metric <= some_bound".
)



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
            _l = l - ((i // (num_layers // l))) + 1
            if _l == 0:
                _l = 1
            lay = [1] + [w]*l + [2]
            param_map = nf.nets.MLP(lay, init_zeros=True)
            flows.append(nf.flows.AffineCouplingBlock(param_map))
            flows.append(nf.flows.Permute(2, mode='swap'))
        elif IB == 'inc':
            _l = (i // (num_layers // l)) + 1
            if _l == 0:
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
def train_flow(parameterization):
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

    max_iter = 10000
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
        l = parameterization["l"]
        w = parameterization["w"]
        num_layers = parameterization["num_layers"]
        IB = parameterization["IB"]

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
                #session.report({"loss": loss.to('cpu').data.numpy()})
                torch.save(model.state_dict(), "./model.pth")
                if loss < best_loss:
                    best_loss = loss
                    best_model_params = (l, w, num_layers)

                    

            except Exception as e:
                print('error',e)
    return loss.to('cpu').data.numpy()

search_space = {
    "w": tune.choice([32,64,128,256,512]),
    "l": tune.choice([2,3,4,5,6]),
    "num_layers": tune.choice([4,8,12,16,20,24,28]),
    "IB": tune.choice(['dec', 'inc','same']),
}
# search_space = {
#     "w": tune.choice([32]),
#     "l": tune.choice([2]),
#     "num_layers": tune.choice([4]),
#     "IB": tune.choice(['dec', 'inc','same']),
# }

# %%


# %%

for i in range(25):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(trial_index=trial_index, raw_data=train_flow(parameters))

dfs = ax_client.get_trials_data_frame()
ax_client.save_to_json_file()  # For custom filepath, pass `filepath` argument.
dfs.to_csv('ax_results.csv')

best_parameters, values = ax_client.get_best_parameters()
torch.save(best_parameters, "./best_parameters.pth")
torch.save(values, "./values.pth")

#logdir = results.get_best_result("mean_accuracy", mode="max").log_dir
# logdir = analysis.get_best_result("loss", mode="max").log_dir
# state_dict = torch.load(os.path.join(logdir, "model.pth"))

# %%
# analysis = torch.load("./analysis_old.pth")
# analysis.get_best_config("loss", mode="max")
# dfs = torch.load("./dfs2_old.pth")
# import pandas as pd
# analysis.fetch_trial_dataframes()
# %%
