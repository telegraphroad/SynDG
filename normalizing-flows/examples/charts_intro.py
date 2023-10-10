# %% 
from normflows.distributions.target import *
from torch.distributions import MultivariateNormal, Normal
import normflows as nf
import torch
import numpy as np
from matplotlib import gridspec
from torch.optim.lr_scheduler import ReduceLROnPlateau

import normflows as nf

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy
from tqdm import tqdm
class NealsFunnel(Target):
    """
    Bimodal two-dimensional distribution

    Parameters:
    prop_scale (float, optional): Scale for the distribution. Default is 20.
    prop_shift (float, optional): Shift for the distribution. Default is -10.
    v1shift (float, optional): Shift parameter for v1. Default is 0.
    v2shift (float, optional): Shift parameter for v2. Default is 0.
    """

    def __init__(self, prop_scale=torch.tensor(20.), prop_shift=torch.tensor(-10.), v1shift = 0., v2shift = 0.):
        super().__init__()
        self.n_dims = 2
        self.max_log_prob = 0.
        self.v1shift = v1shift
        self.v2shift = v2shift
        self.register_buffer("prop_scale", prop_scale)
        self.register_buffer("prop_shift", prop_shift)


    def log_prob(self, z):
        """
        Compute the log probability of the distribution for z

        Parameters:
        z (Tensor): Value or batch of latent variable

        Returns:
        Tensor: Log probability of the distribution for z
        """
        v = z[:,0].cpu()
        x = z[:,1].cpu()
        v_like = nf.distributions.base_extended.GeneralizedGaussianDistribution(torch.tensor([0.0]).cpu(), torch.tensor([1.0]).cpu() + self.v1shift,torch.tensor([4.0]).cpu()).log_prob(v).cpu()
        x_like = Normal(torch.tensor([0.0]).cpu(), torch.exp(0.5*v).cpu() + self.v2shift).log_prob(x).cpu()
        return v_like + x_like


models = []
target = NealsFunnel()
grid_size = 200
device='cuda'
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

target.sample(10000).median(axis=0).values.cpu().detach().numpy()
target.sample(10000).std(axis=0).cpu().detach().numpy()
# %%
log_prob = target.log_prob(zz.to(device))
# %%
# Import packages


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
trnbl = False

for _betab in [0.]:
    for _betat in [1.,2.,3.]:
        for _trnbl in [False]:
            target = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=False, noise_scale=0.2, dim=2,loc=0,scale=1.,p=_betat,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)
            # target = nf.distributions.NealsFunnel(v1shift=0.,v2shift=0.)

            # target = NealsFunnel()
            try:
                _l = target.sample(10000).median().item()
                _l = target.sample(50000).median(axis=0).values.cpu().detach().numpy()

            except:
                _l = target.sample(torch.rand(50000).size()).median(axis=0).values.cpu().detach().numpy()

            # Define 2D Gaussian base distribution
            loc = torch.zeros((2, 2))  # 2x2 tensor filled with 0s
            scale = torch.ones((2, 2))  # 2x2 tensor filled with 1s
            p = 2.0  # Shape parameter for Gaussian

            trnbl = _trnbl

            #base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=False, noise_scale=0.2, dim=2,loc=_l,scale=target.sample(50000).std(axis=0).detach().cpu().numpy(),p=_betab,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)
            #base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=False, noise_scale=0.2, dim=2,loc=0.,scale=1.,p=_betab,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)
            #base = nf.distributions.base_extended.MultivariateStudentT(shape=(2,),df=2.,trainable=False).cuda()
            base = nf.distributions.base_extended.MultivariateStudentTDist(degree_of_freedom=2.,dim=2,trainable=False,device='cuda').cuda()


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
            max_iter = 3000
            num_samples = 2 ** 10
            show_iter = 250


            loss_hist = np.array([])

            optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, weight_decay=5e-7)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=250, verbose=True)
            # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
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
                    model.log_prob(zz).to('cpu').view(*xx.shape)
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
                        else:
                            max_norm -= adjust_rate

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
                        with torch.no_grad():
                            if loss.item()<bestloss:
                                model.log_prob(zz).to('cpu').view(*xx.shape)
                                bestloss = copy.deepcopy(loss.item())
                                best_params = copy.deepcopy(model.state_dict())
                        scheduler.step(bestloss)
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




                except Exception as e:
                    if True:
                        #print('error',e)
                        with torch.no_grad():

                            model.load_state_dict(best_params)

                # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

                #model = model.to(device)
            model.load_state_dict(best_params)
            models.append([_betab,_betat,_trnbl,model,base,target])
            torch.save(models,'models.pt')



for _betab in [1.,2.,3.]:
    for _betat in [1.,2.,3.]:
        for _trnbl in [False]:
            target = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=False, noise_scale=0.2, dim=2,loc=0,scale=1.,p=_betat,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)
            # target = nf.distributions.NealsFunnel(v1shift=0.,v2shift=0.)

            # target = NealsFunnel()
            try:
                _l = target.sample(10000).median().item()
                _l = target.sample(50000).median(axis=0).values.cpu().detach().numpy()

            except:
                _l = target.sample(torch.rand(50000).size()).median(axis=0).values.cpu().detach().numpy()

            # Define 2D Gaussian base distribution
            loc = torch.zeros((2, 2))  # 2x2 tensor filled with 0s
            scale = torch.ones((2, 2))  # 2x2 tensor filled with 1s
            p = 2.0  # Shape parameter for Gaussian

            trnbl = _trnbl

            #base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=False, noise_scale=0.2, dim=2,loc=_l,scale=target.sample(50000).std(axis=0).detach().cpu().numpy(),p=_betab,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)
            base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=False, noise_scale=0.2, dim=2,loc=0.,scale=1.,p=_betab,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)

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
            max_iter = 3000
            num_samples = 2 ** 10
            show_iter = 250


            loss_hist = np.array([])

            optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, weight_decay=5e-7)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=250, verbose=True)
            # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
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
                    model.log_prob(zz).to('cpu').view(*xx.shape)
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
                        else:
                            max_norm -= adjust_rate

                        if (it + 1) % 100 == 0:
                            # print(f'+++++++++++++ avgnorm: {avg_norm},{avg_grad},{bestloss}')
                            # print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
                            # print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())

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
                                model.log_prob(zz).to('cpu').view(*xx.shape)
                                bestloss = copy.deepcopy(loss.item())
                                best_params = copy.deepcopy(model.state_dict())
                        scheduler.step(bestloss)
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


                        print(f'+++++++++++++ BEST LOSS: {bestloss}')
                            


                except Exception as e:
                    if True:
                        #print('error',e)
                        with torch.no_grad():

                            model.load_state_dict(best_params)

                # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

                #model = model.to(device)
            model.load_state_dict(best_params)
            models.append([_betab,_betat,_trnbl,model,base,target])
            torch.save(models,'models.pt')


# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()

# %%
models = torch.load('../../models.pt')
model = models[1][3]
base = models[3][4]
target = models[3][5]
grid_size = 200
device='cuda'
xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)

import seaborn as sns

sns.set_style("darkgrid")

fig, axs = plt.subplots(3, 5, figsize=(23, 12))

for _r in models:
    j = int(_r[0]) - 1 
    if j == -1:
        j = 3
    j += 1
    i = int(_r[1]) - 1
    k = _r[2]
    model = _r[3]
    base = _r[4]
    target = _r[5]
    if not k:
        try:
            log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
            prob = torch.exp(log_prob)
            prob[torch.isnan(prob)] = 0
            prob[torch.isinf(prob)] = 0
            axs[i, j].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
            axs[i, j].set_aspect('equal', 'box')
            axs[i, j].tick_params(axis='x', labelsize=14)  # Increase x-ticks font size
            axs[i, j].tick_params(axis='y', labelsize=14)  # Increase y-ticks font size
            # Remove y ticks from second and third columns
            
            if i == j-1:
                log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
                prob = torch.exp(log_prob)
                prob[torch.isnan(prob)] = 0
                axs[i, 0].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
                axs[i, 0].set_aspect('equal', 'box')
                axs[i, 0].tick_params(axis='x', labelsize=14)  # Increase x-ticks font size
                axs[i, 0].tick_params(axis='y', labelsize=14)  # Increase y-ticks font size
                
            if j in [1, 2,3,4]:
                axs[i, j].set_yticks([])
            
            # Remove x ticks from first and second rows
            if i in [0, 1]:
                axs[i, j].set_xticks([])
                axs[i, 0].set_xticks([])
        except Exception as e:
            print(torch.isnan(prob).any())
            print(torch.isinf(prob).any())
            print(e)
            pass

cmap = plt.get_cmap('coolwarm')  # use your colormap here
norm = plt.Normalize(0, 1)  # adjust your scale here
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

plt.subplots_adjust(hspace=-0.1, wspace=0.04)
from matplotlib.lines import Line2D

line_xpos = 0.199  # Adjust this as needed

# Add the line to the figure
line = Line2D([line_xpos, line_xpos], [0.038, 0.961], transform=fig.transFigure, color='red', linestyle='--',lw=3)
fig.lines.append(line)

titles = ["Target", "Heavy tailed GGD base", "Gaussian base", "Light tailed GGD base","Student's t base"]
for col in range(5):
    axs[0, col].set_title(titles[col],fontsize=23)

row_labels = ["Heavy tailed GGD target", "Gaussian target", "Light tailed GGD target"]
for row in range(3):
    axs[row, 0].set_ylabel(row_labels[row], rotation=90, labelpad=9, verticalalignment='center', fontsize=21)

plt.tight_layout()  # Automatically adjust subplot parameters to give specified padding
cbar = fig.colorbar(sm, ax=axs, orientation='vertical',pad=0.01)
cbar.ax.tick_params(labelsize=14)


# Set colorbar label and increase its font size
#cbar.set_label('Probability density', size=23)


plt.savefig('intro_misspecification.png', bbox_inches='tight', pad_inches=0)




# %%

# %% 
print(len(models))
# %%
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
len(models)
# %%
for _betab in [1.]:
    for _betat in [1.,2.,3.]:
        for _trnbl in [False]:
            target = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=False, noise_scale=0.2, dim=2,loc=0,scale=1.,p=_betat,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)
            # target = nf.distributions.NealsFunnel(v1shift=0.,v2shift=0.)

            # target = NealsFunnel()
            try:
                _l = target.sample(10000).median().item()
                _l = target.sample(50000).median(axis=0).values.cpu().detach().numpy()

            except:
                _l = target.sample(torch.rand(50000).size()).median(axis=0).values.cpu().detach().numpy()

            # Define 2D Gaussian base distribution
            loc = torch.zeros((2, 2))  # 2x2 tensor filled with 0s
            scale = torch.ones((2, 2))  # 2x2 tensor filled with 1s
            p = 2.0  # Shape parameter for Gaussian

            trnbl = _trnbl

            #base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=False, noise_scale=0.2, dim=2,loc=_l,scale=target.sample(50000).std(axis=0).detach().cpu().numpy(),p=_betab,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)
            base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=False, noise_scale=0.2, dim=2,loc=0.,scale=1.,p=_betab,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)
            base = nf.distributions.base_extended.MultivariateStudentT(2,trainable=False)


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
            max_iter = 3000
            num_samples = 2 ** 10
            show_iter = 250


            loss_hist = np.array([])

            optimizer = torch.optim.Adam(model.parameters(), lr=5e-6, weight_decay=5e-7)
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=250, verbose=True)
            # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
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
                    model.log_prob(zz).to('cpu').view(*xx.shape)
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
                        else:
                            max_norm -= adjust_rate

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
                                model.log_prob(zz).to('cpu').view(*xx.shape)
                                bestloss = copy.deepcopy(loss.item())
                                best_params = copy.deepcopy(model.state_dict())
                        scheduler.step(bestloss)
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
            model.load_state_dict(best_params)
            models.append([_betab,_betat,_trnbl,model,base,target])
            torch.save(models,'models.pt')
