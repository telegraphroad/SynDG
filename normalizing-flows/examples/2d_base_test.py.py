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
base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=100, rand_p=True, noise_scale=0.1, dim=2,loc=_l,scale=1.,p=2.,device=device,trainable_loc=False, trainable_scale=False,trainable_p=True,trainable_weights=False)
#base = nf.distributions.base.DiagGaussian(2)

#base = nf.distributions.GaussianMixture(n_modes=10,dim=2)
#base = nf.distributions.base.DiagGaussian(2)
# Define list of flows
num_layers = 64
#num_layers = 8
flows = []
latent_size = 2
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
for i in range(num_layers):
    s = nf.nets.MLP([latent_size, 4 * latent_size, latent_size], init_zeros=True)
    t = nf.nets.MLP([latent_size, 4 * latent_size, latent_size], init_zeros=True)
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
max_iter = 10000
num_samples = 2 ** 11
show_iter = 250


loss_hist = np.array([])

#optimizer = torch.optim.Adam(model.parameters(), lr=1e-6, weight_decay=1e-7)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-6)
max_norm = 1.0
adjust_rate = 0.01
model.sample(10**4)
best_params = copy.deepcopy(model.state_dict())
for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    # Get training samples
    x = target.sample(num_samples).to(device)
    
    # Compute loss
    try:
        loss = model.forward_kld(x, robust=True)    
    # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            avg_norm = compute_average_grad_norm(model)
            if avg_norm > max_norm:
                max_norm += adjust_rate
                #print('++++++++++++++++++++++++++',max_norm)
            else:
                max_norm -= adjust_rate
                #print('++++++++++++++++++++++++++',max_norm)
            with torch.no_grad():
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            if (it + 1) % 100 == 0:

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
            
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        if (it + 1) % show_iter == 0:
            best_params = copy.deepcopy(model.state_dict())
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


            print(f'+++++++++++++ maxnorm: {max_norm}')
            print('=======means: ',model.q0.component_distribution.base_dist.loc.mean().item(),model.q0.component_distribution.base_dist.scale.mean().item(),model.q0.component_distribution.base_dist.p.mean().item())
            print('=======medians: ',model.q0.component_distribution.base_dist.loc.median().item(),model.q0.component_distribution.base_dist.scale.median().item(),model.q0.component_distribution.base_dist.p.median().item())
            print('=======mins: ',model.q0.component_distribution.base_dist.loc.min().item(),model.q0.component_distribution.base_dist.scale.min().item(),model.q0.component_distribution.base_dist.p.min().item())
            print('=======maxs: ',model.q0.component_distribution.base_dist.loc.max().item(),model.q0.component_distribution.base_dist.scale.max().item(),model.q0.component_distribution.base_dist.p.max().item())
                


    except Exception as e:
        print('error',e)
        model.state_dict = copy.deepcopy(best_params)
        #print('error')
        break

    # Log loss
    
    
    # Plot learned distribution

# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()

# %% [markdown]
# ## Visualizing the learned distribution
model.eval()
x = target.sample(10**4).to(device).cpu().detach().numpy()
y = model.sample(10**4)
#model.train()
#plt.figure(figsize=(15, 15))
# #line plot the first marginals from x and y on one plot
# plt.hist(x[:,0],bins=100,alpha=0.5,label='target')
# plt.hist(y[:,0],bins=100,alpha=0.5,label='model')
# plt.legend()
# plt.show()
# plt.figure(figsize=(15, 15))
# plt.hist(x[:,1],bins=100,alpha=0.5,label='target')
# plt.hist(y[:,1],bins=100,alpha=0.5,label='model')
# plt.legend()
# plt.show()

# %%
# Plot target distribution
f, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 7))

log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

ax[0].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')

ax[0].set_aspect('equal', 'box')
ax[0].set_axis_off()
ax[0].set_title('Target', fontsize=24)

# Plot learned distribution
model.eval()
log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
model.train()
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0

ax[1].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')

ax[1].set_aspect('equal', 'box')
ax[1].set_axis_off()
ax[1].set_title('Real NVP', fontsize=24)

plt.subplots_adjust(wspace=0.1)

plt.show()

# %% [markdown]
# Notice there is a bridge between the two modes of the learned target.
# This is not a big deal usually since the bridge is really thin, and going to higher dimensional space will make it expoentially unlike to have samples within the bridge.
# However, we can see the shape of each mode is also a bit distorted.
# So it would be nice to get rid of the bridge. Now let's try to use a Gaussian mixture distribution as our base distribution, instead of a single Gaussian.

# %% [markdown]
# # Use a Gaussian mixture model as the base instead

# %%
# Set up model

# Define a mixture of Gaussians with 2 modes.
# base = nf.distributions.base.GaussianMixture(2,2, loc=[[-2,0],[2,0]],scale=[[0.3,0.3],[0.3,0.3]])
# base = nf.distributions.base_extended.MultivariateStudentT(2,df=5.)
# Define list of flows
num_layers = 32
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
model = nf.NormalizingFlow(base3, flows).cuda()

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
# ## Train the new model

# %%
# Train model
max_iter = 4000
num_samples = 2 ** 9
show_iter = 500


loss_hist = np.array([])

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    # Get training samples
    x = target.sample(num_samples).to(device)
    
    # Compute loss
    loss = model.forward_kld(x, robust=True)
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    # Log loss
    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
    
    # Plot learned distribution
    if (it + 1) % show_iter == 0:
        model.eval()
        log_prob = model.log_prob(zz)
        model.train()
        prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
        prob[torch.isnan(prob)] = 0

        plt.figure(figsize=(15, 15))
        plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
        plt.gca().set_aspect('equal', 'box')
        plt.show()

# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()

# %% [markdown]
# Now the modes are in better shape! And there is no bridge between the two modes!
print('test')

