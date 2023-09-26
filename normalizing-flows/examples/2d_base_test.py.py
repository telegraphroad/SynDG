# %% [markdown]
# # Changing the base distribution of a flow model
# 
# This example shows how one can easily change the base distribution with our API.
# First, let's look at how the normalizing flow can learn a two moons target distribution with a Gaussian distribution as the base.

# %%
# Import packages
import torch
import numpy as np
from matplotlib import gridspec

import normflows as nf

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy
from tqdm import tqdm

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
enable_cuda = True

device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
target = nf.distributions.TwoMoons()
target = nf.distributions.StudentTDistribution(2,df=2.)
target = nf.distributions.NealsFunnel(v1shift=0.,v2shift=0.)
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
trnbl = True
base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=10, rand_p=True, noise_scale=0.5, dim=2,loc=0,scale=1.,p=2.,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)
#base = nf.distributions.base.DiagGaussian(2)

#base = nf.distributions.GaussianMixture(n_modes=10,dim=2)
#base = nf.distributions.base.DiagGaussian(2)
# Define list of flows
num_layers = 18
#num_layers = 8
flows = []
latent_size = 2
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
for i in range(num_layers):
    s = nf.nets.MLP([latent_size, 128,128,128, latent_size], init_zeros=True)
    t = nf.nets.MLP([latent_size, 128,128,128, latent_size], init_zeros=True)
    if i % 2 == 0:
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
    flows += [nf.flows.ActNorm(latent_size)]


# Construct flow model
model = nf.NormalizingFlow(base, flows)


# %%
# Move model on GPU if available

model = model.to(device)

# %%
# Define target distribution
def check_model_params(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f'Parameter {name} has NaNs or infs')


# %%
# Plot target distribution
grid_size = 200
xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)

log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure(figsize=(15, 15))
plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
plt.gca().set_aspect('equal', 'box')
plt.show()

# %%
print(zz.shape)
print(base.log_prob(zz).shape,base2.log_prob(zz.cpu()).shape)
log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
# %%
# Plot initial flow distribution
model.eval()
log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
model.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

plt.figure(figsize=(15, 15))
plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
plt.gca().set_aspect('equal', 'box')
plt.show()

# %% [markdown]
# ## Training the model

# %%
# Train model
max_iter = 8000
num_samples = 2 ** 11
show_iter = 250


loss_hist = np.array([])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-7)
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
max_norm = 0.5
adjust_rate = 0.01
model.sample(10**4)
best_params = copy.deepcopy(model.state_dict())
bestloss = 1e10
for it in tqdm(range(max_iter)):
    # if it == 1000:
    #     optimizer.set_lr(1e-6)
    optimizer.zero_grad()
    
    # Get training samples
    x = target.sample(num_samples).to(device)
    
    # Compute loss
    try:
        loss = model.forward_kld(x, robust=False)    
        # l2_lambda = 0.001  # The strength of the regularization
        # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
        # loss = loss + l2_lambda * l2_norm
    # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            avg_grad = 0.0
            num_params = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    avg_grad += param.grad.data.abs().mean().item()
                    num_params += 1
            avg_grad /= num_params
            
            avg_norm = avg_grad
            if avg_norm > max_norm:
                max_norm += adjust_rate
                #print('++++++++++++++++++++++++++',max_norm)
            else:
                max_norm -= adjust_rate
                #print('++++++++++++++++++++++++++',max_norm)
            # with torch.no_grad():
            #     #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            # print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
            # print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())

            if (it + 1) % 100 == 0:
                print(f'+++++++++++++ avgnorm: {avg_norm},{avg_grad},{bestloss}')
                print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
                print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())

                max_grad = 0.0
                min_grad = 1e10
                avg_grad = 0.0
                num_params = 0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        max_grad = max(max_grad, param.grad.data.abs().max().item())
                        min_grad = min(min_grad, param.grad.data.abs().min().item())
                        avg_grad += param.grad.data.abs().mean().item()
                        num_params += 1
                avg_grad /= num_params
                print(f'Epoch {it+1}, Max Gradient: {max_grad:.6f}, Min Gradient: {min_grad:.6f}, Avg Gradient: {avg_grad:.6f}')


            optimizer.step()
            import copy
            with torch.no_grad():
                if loss.item()<bestloss:
                    bestloss = copy.deepcopy(loss.item())
                    best_params = copy.deepcopy(model.state_dict())
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        if (it + 1) % show_iter == 0:
            
            model.eval()
            log_prob = model.log_prob(zz).detach().cpu()
            model.train()
            prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
            prob[torch.isnan(prob)] = 0

            plt.figure(figsize=(15, 15))
            plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
            plt.gca().set_aspect('equal', 'box')
            plt.show()
            with torch.no_grad():
                model.eval()
                x = target.sample(100000).to(device).cpu().detach().numpy()
                y,_ = model.sample(100000)
                y = y.to(device).cpu().detach().numpy()
                model.train()
                plt.figure(figsize=(15, 15))
                #line plot the first marginals from x and y on one plot
                plt.hist(x[:,0],bins=500,alpha=0.5,label='target')
                plt.hist(y[:,0],bins=500,alpha=0.5,label='model')
                plt.legend()
                plt.show()
                plt.figure(figsize=(15, 15))
                plt.hist(x[:,1],bins=500,alpha=0.5,label='target')
                plt.hist(y[:,1],bins=500,alpha=0.5,label='model')
                plt.legend()
                plt.show()
                plt.figure(figsize=(10, 10))
                plt.plot(loss_hist, label='loss')
                plt.legend()
                plt.show()


            print(f'+++++++++++++ avgnorm: {avg_norm},{avg_grad},{bestloss}')
            print('=======means: ',model.q0.loc.mean().item(),model.q0.scale.mean().item(),model.q0.p.mean().item())
            print('=======medians: ',model.q0.loc.median().item(),model.q0.scale.median().item(),model.q0.p.median().item())
            print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
            print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())
                


    except Exception as e:
        if True:
            #print('error',e)
            with torch.no_grad():
                # b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
                # flows = []
                # for i in range(num_layers):
                #     s = nf.nets.MLP([latent_size, 16,16, latent_size], init_zeros=True)
                #     t = nf.nets.MLP([latent_size, 16,16, latent_size], init_zeros=True)
                #     if i % 2 == 0:
                #         flows += [nf.flows.MaskedAffineFlow(b, t, s)]
                #     else:
                #         flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
                #     flows += [nf.flows.ActNorm(latent_size)]


                # # Construct flow model
                # model = nf.NormalizingFlow(base, flows)

                model.load_state_dict(best_params)
                print(f'+++++++++++++ avgnorm: {avg_norm},{avg_grad},{bestloss}')
                print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
                print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())

                # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

                #model = model.to(device)
                #model.train()

        #print('error')

    # Log loss
    
    
    # Plot learned distribution

# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()

# %%

# # %% [markdown]
# # Now the modes are in better shape! And there is no bridge between the two modes!
# num_samples = 2 ** 12
# _s,_ = model.sample(num_samples)
# mix = model.q0.mixture
# component_distributions = mix.component_distribution.base_dist
# mixture_weights = mix.mixture_distribution.probs

# num_samples = _s.shape[0]
# num_components = mixture_weights.shape[-1]

# # Create a tensor to store the probabilities for each component on each sample
# probabilities = []
# i = 0

# probabilities = []
# for i in range(model.q0.n_modes):
#     loc_i = model.q0.loc[i]
#     scale_i = model.q0.scale[i]
#     p_i = model.q0.p[i]
    
#     # Create a GeneralizedGaussianDistribution for the i-th component
#     gg_i = torch.distributions.Independent(
#         nf.distributions.base_extended.GeneralizedGaussianDistribution(loc_i, scale_i, p_i), 1)
        
#     # Evaluate the probability of _s under this distribution
#     log_prob_i = gg_i.log_prob(_s)
    
#     # Convert to probability from log probability
#     prob_i = torch.exp(log_prob_i)
#     #prob_i = log_prob_i
#     probabilities.append(prob_i)

# probabilities = torch.stack(probabilities).T
# probs = probabilities.detach().cpu().numpy()
# import pandas as pd
# import seaborn as sns
# df = pd.DataFrame(probs)

# bins = np.linspace(probabilities.min().item(), probabilities.max().item(), 20) # 20 bins between 0 and 1

# df_binned = df.apply(lambda col: pd.cut(col, bins, labels=bins[:-1]))

# df_mean_probs = df_binned.apply(lambda col: df[col.name].groupby(df_binned[col.name]).median())

# df_long = df_mean_probs.reset_index().melt(id_vars='index', var_name='Distribution', value_name='Mean Probability')

# plt.figure(figsize=(10, 10))
# scatter = sns.scatterplot(data=df_long, x='index', y='Distribution', size='Mean Probability', hue='Mean Probability', palette='viridis_r', legend=False)
# plt.xlabel('Probability Bin')
# plt.ylabel('Distribution')

# plt.yticks(ticks=np.arange(model.q0.n_modes), labels=np.arange(1, model.q0.n_modes + 1))

# norm = plt.Normalize(df_long['Mean Probability'].min(), df_long['Mean Probability'].max())
# sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=norm)
# sm.set_array([])

# plt.colorbar(sm, label='Mean Probability')

# plt.show()

# %% [markdown]
import numpy as np
import matplotlib.pyplot as plt
num_samples = 2 ** 15
_s,_ = model.sample(num_samples)
mix = model.q0.mixture
component_distributions = mix.component_distribution.base_dist
mixture_weights = mix.mixture_distribution.probs

num_samples = _s.shape[0]
num_components = mixture_weights.shape[-1]

# Create a tensor to store the probabilities for each component on each sample
probabilities = []
i = 0
import copy
probabilities = []
for i in range(model.q0.n_modes):
    loc_i = model.q0.loc[i]
    scale_i = model.q0.scale[i]
    p_i = model.q0.p[i]
    
    # Create a GeneralizedGaussianDistribution for the i-th component
    gg_i = torch.distributions.Independent(
        nf.distributions.base_extended.GeneralizedGaussianDistribution(loc_i, scale_i, p_i), 1)
        
    # Evaluate the probability of _s under this distribution
    log_prob_i = gg_i.log_prob(_s)
    
    # Convert to probability from log probability
    prob_i = torch.exp(log_prob_i)
    #prob_i = log_prob_i
    probabilities.append(prob_i)

probabilities = torch.stack(probabilities).T
probs = probabilities.detach().cpu().numpy()

# Assuming oprobs and mprobs are numpy arrays
oprobs = target.log_prob(_s).exp().detach().cpu().numpy()
mprobs = probs
nbins =30
# Create 100 bins between min and max of oprobs
bins = np.linspace(np.min(oprobs), np.max(oprobs), nbins+1)

# Digitize oprobs into bins
indices = np.digitize(oprobs, bins)

# Prepare an array to store mean probabilities
mean_probs = np.full((nbins, num_components), np.nan)

# Calculate mean probabilities for each bin and each mixture component
for i in range(num_components):
    for j in range(nbins):
        if np.any(indices == j + 1):
            mean_probs[j, i] = np.mean(mprobs[indices == j + 1, i])

mp = copy.deepcopy(mean_probs)
max_indices = np.argmax(mean_probs, axis=1)
for i in range(mean_probs.shape[0]):
    for j in range(mean_probs.shape[1]):
        if j!=max_indices[i]:
            mean_probs[i,j]=np.nan
# import numpy as np

# Assuming mean_probs is your 50x10 array
# Create a new array filled with NaN values
# nan_array = np.empty_like(mean_probs)
# nan_array[:] = np.nan

# # Find the maximum value in each row
# max_values = np.nanmax(mean_probs, axis=1)

# # Create a boolean mask where only the maximum values are True
# mask = mean_probs == max_values[:, np.newaxis]

# # Use the mask to select the maximum values from mean_probs
# mean_probs = np.where(mask, mean_probs, nan_array)
# Flattening the data for scatter plot
x, y = np.meshgrid(range(nbins), range(num_components))
x, y, c = x.flatten(), y.flatten(), mean_probs.flatten()

# Create a 2D plot
fig, ax = plt.subplots(figsize=(30, 30))

# Use a scatter plot with circle color intensity reflecting the mean probability
scatter = ax.scatter(x, y, c=c, cmap='viridis', alpha=0.6)

ax.set_xlabel('Bins')
ax.set_ylabel('Mixture Components')

fig.colorbar(scatter, label='Mean Probability')
plt.show()
# %%


# Create x, y coordinates using meshgrid
x, y = np.meshgrid(range(nbins), range(num_components))
x, y = x.flatten(), y.flatten()

# Filter x, y, and c arrays to keep only the highest probability components
x_filtered = x[max_indices]
y_filtered = y[max_indices]
c_filtered = mean_probs[range(nbins), max_indices]

# Create a 2D plot
fig, ax = plt.subplots(figsize=(10, 10))

# Use a scatter plot with circle color intensity reflecting the mean probability
scatter = ax.scatter(x_filtered, y_filtered, c=c_filtered, cmap='viridis', alpha=0.6)

plt.show()

# %%
mean_probs.shape# %%
