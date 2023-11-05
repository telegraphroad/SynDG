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
trnbl = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
target1 = nf.distributions.TwoMoons().cuda()
target2 = nf.distributions.StudentTDistribution(2,df=1.).cuda()
target3 = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=False, noise_scale=0.2, dim=2,loc=0,scale=1.,p=0.5,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)
target4 = nf.distributions.NealsFunnel(v1shift=0)
target5 = nf.distributions.NealsFunnel(v1shift=3,v2shift=0.)
target6 = nf.distributions.NealsFunnel(v1shift=6,v2shift=0.)
target8 = nf.distributions.target.CircularGaussianMixture()
target9 = nf.distributions.target.RingMixture(n_rings=3).cuda()

# target = NealsFunnel()
try:
    _l = target.sample(10000).median().item()
    _l = target.sample(50000).median(axis=0).values.cpu().detach().numpy()
    _s = target.sample(100000).std(axis=0).cpu().detach().numpy()
except:
    _l = target.sample(torch.rand(50000).size()).median(axis=0).values.cpu().detach().numpy()
    _s = target.sample(torch.rand(50000).size()).std(axis=0).cpu().detach().numpy()

# Define 2D Gaussian base distribution
loc = torch.zeros((2, 2))  # 2x2 tensor filled with 0s
scale = torch.ones((2, 2))  # 2x2 tensor filled with 1s
p = 2.0  # Shape parameter for Gaussian

# Create the 2-dimensional instance
# base3 = nf.distributions.base_extended.GeneralizedGaussianDistribution(loc, scale, p)

# base2 = nf.distributions.base.DiagGaussian(2)
# base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=20, rand_p=False, dim=2,loc=_l,scale=1.,p=2.,device=device,trainable_loc=False,trainable_scale=False,trainable_p=False,trainable_weights=False)
# base = nf.distributions.base.DiagGaussian(2)
# base4 = nf.distributions.GaussianMixture(n_modes=10,dim=2)
# base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=10, rand_p=False, dim=2,loc=0,scale=1.,p=2.,device=device,trainable_loc=False,trainable_scale=False,trainable_p=False,trainable_weights=False)
# base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=10, rand_p=True, dim=2,loc=0,scale=1.,p=2.,device=device,trainable_loc=True,trainable_scale=True,trainable_p=True,trainable_weights=True)
# base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=100, rand_p=True, dim=2,loc=_l,scale=1.,p=2.,device=device,trainable_loc=True,trainable_scale=True,trainable_p=True,trainable_weights=True)
# base = nf.distributions.base.DiagGaussian(2)
# base = nf.distributions.GaussianMixture(n_modes=10,dim=2)
# base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=100, rand_p=True, dim=2,loc=_l,scale=1.,p=2.,noise_scale=0.2,device=device,trainable_loc=True,trainable_scale=True,trainable_p=True,trainable_weights=True)
trnbl = True
base1 = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=20, rand_p=True, noise_scale=0.2, dim=2,loc=_l,scale=_s,p=2.,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl).cuda()
base2 = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=False, noise_scale=0.2, dim=2,loc=0,scale=1,p=2.,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl).cuda()
base3 = nf.distributions.base.DiagGaussian(2,trainable=True)
base4 = nf.distributions.base_extended.MultivariateStudentTDist(degree_of_freedom=2.,dim=2,trainable=True)
base5 = nf.distributions.GaussianMixture(n_modes=20,dim=2)
#base = nf.distributions.base.DiagGaussian(2)
max_iter = 20000
base = nf.distributions.GaussianMixture(n_modes=10,dim=2)
base = nf.distributions.base.DiagGaussian(2)

# for base,base_name in zip([base2,base3,base4,base5],['Generalized Gaussian','Gaussian',"Multivariate Student's t",'Gaussian Mixture']):
#     for target,target_name in zip([target1,target2,target3,target8],['Two Moons',r"Student's t, $\nu$=1 (Cauchy)",'Generalized Gaussian','Gaussian Mixtures']):
        
#         num_layers = 10
#         #num_layers = 8
#         flows = []
#         latent_size = 2
#         b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
#         flows = []
#         for i in range(num_layers):
#             s = nf.nets.MLP([latent_size, 128,128,128, latent_size], init_zeros=True)
#             t = nf.nets.MLP([latent_size, 128,128,128, latent_size], init_zeros=True)
#             if i % 2 == 0:
#                 flows += [nf.flows.MaskedAffineFlow(b, t, s)]
#             else:
#                 flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
#             flows += [nf.flows.ActNorm(latent_size)]


#         # Construct flow model
#         base.to(device)
#         model = nf.NormalizingFlow(base, flows)


#         # %%
#         # Move model on GPU if available

#         base.to(device)
#         model = model.to(device)

#         # %%
#         # Define target distribution
#         def check_model_params(model):
#             for name, param in model.named_parameters():
#                 if torch.isnan(param).any() or torch.isinf(param).any():
#                     print(f'Parameter {name} has NaNs or infs')


#         # %%
#         # Plot target distribution
#         grid_size = 200
#         xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
#         zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
#         zz = zz.to(device)
#         print('============================================')
#         print(base_name,target_name)
#         print('============================================')

#         log_prob = target.log_prob(zz.cuda()).to('cpu').view(*xx.shape)
#         prob = torch.exp(log_prob)
#         prob[torch.isnan(prob)] = 0

#         plt.figure(figsize=(15, 15))
#         plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
#         plt.gca().set_aspect('equal', 'box')
#         plt.show()

#         # %%
#         print(zz.shape)
#         print(base.log_prob(zz).shape,base2.log_prob(zz).shape)
#         log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
#         # %%
#         # Plot initial flow distribution
#         model.eval()
#         log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
#         model.train()
#         prob = torch.exp(log_prob)
#         prob[torch.isnan(prob)] = 0

#         plt.figure(figsize=(15, 15))
#         plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
#         plt.gca().set_aspect('equal', 'box')
#         plt.show()

#         # %% [markdown]
#         # ## Training the model

#         # %%
#         # Train model
        
#         num_samples = 2 ** 12
#         show_iter = 25000


#         loss_hist = np.array([])

#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
#         scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=300, verbose=True)
#         # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
#         max_norm = 0.5
#         adjust_rate = 0.01
#         model.sample(10**4)
#         best_params = copy.deepcopy(model.state_dict())
#         bestloss = 1e10
#         for it in tqdm(range(max_iter)):
#             # if it == 1000:
#             #     optimizer.set_lr(1e-6)
#             optimizer.zero_grad()
            
#             # Get training samples
#             x = target.sample(num_samples).to(device)
            
#             # Compute loss
#             try:
#                 if base_name == 'Our method':
#                     rbst = True
#                 else:
#                     rbst = False
#                 loss = model.forward_kld(x, robust=rbst,rmethod='med')    
#                 # l2_lambda = 0.001  # The strength of the regularization
#                 # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
#                 # loss = loss + l2_lambda * l2_norm
#             # Do backprop and optimizer step
#                 if ~(torch.isnan(loss) | torch.isinf(loss)):
#                     loss.backward()
#                     avg_grad = 0.0
#                     num_params = 0
#                     for name, param in model.named_parameters():
#                         if param.grad is not None:
#                             avg_grad += param.grad.data.abs().mean().item()
#                             num_params += 1
#                     avg_grad /= num_params
                    
#                     avg_norm = avg_grad
#                     if avg_norm > max_norm:
#                         max_norm += adjust_rate
#                         #print('++++++++++++++++++++++++++',max_norm)
#                     else:
#                         max_norm -= adjust_rate
#                         #print('++++++++++++++++++++++++++',max_norm)
#                     # with torch.no_grad():
#                     #     #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#                     #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
#                     # print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
#                     # print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())

#                     # if (it + 1) % 100 == 0:
#                     #     # print(f'+++++++++++++ avgnorm: {avg_norm},{avg_grad},{bestloss}')
#                     #     # print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
#                     #     # print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())

#                     #     max_grad = 0.0
#                     #     min_grad = 1e10
#                     #     avg_grad = 0.0
#                     #     num_params = 0
#                     #     for name, param in model.named_parameters():
#                     #         if param.grad is not None:
#                     #             max_grad = max(max_grad, param.grad.data.abs().max().item())
#                     #             min_grad = min(min_grad, param.grad.data.abs().min().item())
#                     #             avg_grad += param.grad.data.abs().mean().item()
#                     #             num_params += 1
#                     #     avg_grad /= num_params
#                     #     print(f'Epoch {it+1}, Max Gradient: {max_grad:.6f}, Min Gradient: {min_grad:.6f}, Avg Gradient: {avg_grad:.6f}')


#                     optimizer.step()
#                     import copy
#                     with torch.no_grad():
#                         if loss.item()<bestloss:
#                             bestloss = copy.deepcopy(loss.item())
#                             best_params = copy.deepcopy(model.state_dict())
#                     scheduler.step(bestloss)
#                     if optimizer.param_groups[0]['lr'] < 1e-7:
#                         model.load_state_dict(best_params)    
#                         torch.save(model, f'./pmodel_{base_name}_{target_name}.pt')
#                         torch.save(base, f'./pbase_{base_name}_{target_name}.pt')
#                         torch.save(target, f'./ptarget_{base_name}_{target_name}.pt')
#                         break
#                 loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
#                 if (it + 1) % show_iter == 0:
                    
#                     model.eval()
#                     log_prob = model.log_prob(zz).detach().cpu()
#                     model.train()
#                     prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
#                     prob[torch.isnan(prob)] = 0

#                     plt.figure(figsize=(15, 15))
#                     plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
#                     plt.gca().set_aspect('equal', 'box')
#                     plt.show()
#                     with torch.no_grad():
#                         model.eval()
#                         x = target.sample(100000).to(device).cpu().detach().numpy()
#                         y,_ = model.sample(100000)
#                         y = y.to(device).cpu().detach().numpy()
#                         model.train()
#                         plt.figure(figsize=(15, 15))
#                         #line plot the first marginals from x and y on one plot
#                         plt.hist(x[:,0],bins=500,alpha=0.5,label='target')
#                         plt.hist(y[:,0],bins=500,alpha=0.5,label='model')
#                         plt.legend()
#                         plt.show()
#                         plt.figure(figsize=(15, 15))
#                         plt.hist(x[:,1],bins=500,alpha=0.5,label='target')
#                         plt.hist(y[:,1],bins=500,alpha=0.5,label='model')
#                         plt.legend()
#                         plt.show()
#                         plt.figure(figsize=(10, 10))
#                         plt.plot(loss_hist, label='loss')
#                         plt.legend()
#                         plt.show()


#                     # print(f'+++++++++++++ avgnorm: {avg_norm},{avg_grad},{bestloss}')
#                     # print('=======means: ',model.q0.loc.mean().item(),model.q0.scale.mean().item(),model.q0.p.mean().item())
#                     # print('=======medians: ',model.q0.loc.median().item(),model.q0.scale.median().item(),model.q0.p.median().item())
#                     # print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
#                     # print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())
                        


#             except Exception as e:
#                 if True:
#                     #print('error',e)
#                     with torch.no_grad():
#                         # b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
#                         # flows = []
#                         # for i in range(num_layers):
#                         #     s = nf.nets.MLP([latent_size, 16,16, latent_size], init_zeros=True)
#                         #     t = nf.nets.MLP([latent_size, 16,16, latent_size], init_zeros=True)
#                         #     if i % 2 == 0:
#                         #         flows += [nf.flows.MaskedAffineFlow(b, t, s)]
#                         #     else:
#                         #         flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
#                         #     flows += [nf.flows.ActNorm(latent_size)]


#                         # # Construct flow model
#                         # model = nf.NormalizingFlow(base, flows)

#                         model.load_state_dict(best_params)
#                         # print(f'+++++++++++++ avgnorm: {avg_norm},{avg_grad},{bestloss}')
#                         # print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
#                         # print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())

#                         # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

#                         #model = model.to(device)
#                         #model.train()

#                 #print('error')

#             # Log loss
#         with torch.no_grad():
#             model.load_state_dict(best_params)    
#         torch.save(model, f'./pmodel_{base_name}_{target_name}.pt')
#         torch.save(base, f'./pbase_{base_name}_{target_name}.pt')
#         torch.save(target, f'./ptarget_{base_name}_{target_name}.pt')

# %%
max_iter = 30000
for target,target_name in zip([target8,target2],['Gaussian Mixtures',r"Student's t, $\nu$=1 (Cauchy)"]):
    for nm in [50]:
        for usestd in [False]:
            for useloc in [False]:                    
                for noisecoef in np.flip([0.001,0.01]):
                    for initp in np.flip([1.0,2.0,3.0]):
                        try:
                            base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=nm, rand_p=True, noise_scale=noisecoef, dim=2,loc=_l if useloc else 0,scale=_s if usestd else 1,p=initp,device=device,trainable_loc=useloc, trainable_scale=usestd,trainable_p=True,trainable_weights=True).cuda()
                        except:
                            base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=nm, rand_p=True, noise_scale=noisecoef/10, dim=2,loc=_l if useloc else 0,scale=_s if usestd else 1,p=initp,device=device,trainable_loc=useloc, trainable_scale=usestd,trainable_p=True,trainable_weights=True).cuda()
                        for base,base_name in zip([base],['Our method']):
                        
                            
                            num_layers = 10
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
                            base.to(device)
                            model = nf.NormalizingFlow(base, flows)


                            # %%
                            # Move model on GPU if available

                            base.to(device)
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
                            print('============================================')
                            print(base_name,target_name)
                            print('============================================')

                            log_prob = target.log_prob(zz.cuda()).to('cpu').view(*xx.shape)
                            prob = torch.exp(log_prob)
                            prob[torch.isnan(prob)] = 0

                            plt.figure(figsize=(15, 15))
                            plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
                            plt.gca().set_aspect('equal', 'box')
                            plt.show()

                            # %%
                            print(zz.shape)
                            print(base.log_prob(zz).shape,base2.log_prob(zz).shape)
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
                            
                            num_samples = 2 ** 12
                            show_iter = 25000


                            loss_hist = np.array([])

                            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
                            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, verbose=True)
                            # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)
                            max_norm = 0.5
                            adjust_rate = 0.01
                            model.sample(10**4)
                            best_params = copy.deepcopy(model.state_dict())
                            best_base = copy.deepcopy(model.q0.state_dict())
                            bestloss = 1e10
                            for it in tqdm(range(max_iter)):
                                # if it == 1000:
                                #     optimizer.set_lr(1e-6)
                                optimizer.zero_grad()
                                
                                # Get training samples
                                x = target.sample(num_samples).to(device)
                                
                                # Compute loss
                                try:
                                    if base_name == 'Our method':
                                        rbst = True
                                    else:
                                        rbst = False
                                    loss = model.forward_kld(x, robust=rbst,rmethod='med')    
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

                                        # if (it + 1) % 100 == 0:
                                        #     # print(f'+++++++++++++ avgnorm: {avg_norm},{avg_grad},{bestloss}')
                                        #     # print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
                                        #     # print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())

                                        #     max_grad = 0.0
                                        #     min_grad = 1e10
                                        #     avg_grad = 0.0
                                        #     num_params = 0
                                        #     for name, param in model.named_parameters():
                                        #         if param.grad is not None:
                                        #             max_grad = max(max_grad, param.grad.data.abs().max().item())
                                        #             min_grad = min(min_grad, param.grad.data.abs().min().item())
                                        #             avg_grad += param.grad.data.abs().mean().item()
                                        #             num_params += 1
                                        #     avg_grad /= num_params
                                        #     print(f'Epoch {it+1}, Max Gradient: {max_grad:.6f}, Min Gradient: {min_grad:.6f}, Avg Gradient: {avg_grad:.6f}')


                                        optimizer.step()
                                        import copy
                                        with torch.no_grad():
                                            if loss.item()<bestloss:
                                                model.log_prob(zz)
                                                bestloss = copy.deepcopy(loss.item())
                                                best_params = copy.deepcopy(model.state_dict())
                                                best_base = copy.deepcopy(model.q0.state_dict())
                                        scheduler.step(bestloss)
                                        if optimizer.param_groups[0]['lr'] < 1e-7:
                                            model.load_state_dict(best_params)    
                                            model.q0.load_state_dict(best_base)    
                                            torch.save(model, f'./model_{base_name}_{target_name}.pt')
                                            torch.save(base, f'./base_{base_name}_{target_name}.pt')
                                            torch.save(target, f'./target_{base_name}_{target_name}.pt')
                                            break

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


                                        # print(f'+++++++++++++ avgnorm: {avg_norm},{avg_grad},{bestloss}')
                                        # print('=======means: ',model.q0.loc.mean().item(),model.q0.scale.mean().item(),model.q0.p.mean().item())
                                        # print('=======medians: ',model.q0.loc.median().item(),model.q0.scale.median().item(),model.q0.p.median().item())
                                        # print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
                                        # print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())
                                            


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
                                            model.q0.load_state_dict(best_base)    
                                            
                                            # print(f'+++++++++++++ avgnorm: {avg_norm},{avg_grad},{bestloss}')
                                            # print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
                                            # print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())

                                            # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-6)

                                            #model = model.to(device)
                                            #model.train()

                                    #print('error')

                                # Log loss
                            with torch.no_grad():
                                model.load_state_dict(best_params)   
                                model.q0.load_state_dict(best_base)     
                                
                            torch.save(model, f'./model_{base_name}_{target_name}_{nm}_{usestd}_{useloc}_{noisecoef}_{initp}.pt')
                            torch.save(base, f'./base_{base_name}_{target_name}_{nm}_{usestd}_{useloc}_{noisecoef}_{initp}.pt')
                            torch.save(target, f'./target_{base_name}_{target_name}_{nm}_{usestd}_{useloc}_{noisecoef}_{initp}.pt')
                            with torch.no_grad():
                                model.eval()
                                fig,axs = plt.subplots(1, 2, figsize=(10,10))
                                log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
                                prob = torch.exp(log_prob)
                                prob[torch.isnan(prob)] = 0
                                prob[torch.isinf(prob)] = 0
                                
                                axs[0].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
                                axs[0].set_aspect('equal', 'box')
                                log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
                                prob = torch.exp(log_prob)
                                prob[torch.isnan(prob)] = 0
                                prob[torch.isinf(prob)] = 0
                                axs[1].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
                                axs[1].set_aspect('equal', 'box')
                                axs[1].set_title(f'./target_{base_name}_{target_name}_{nm}_{usestd}_{useloc}_{noisecoef}_{initp}')
                                plt.savefig(f'./plt_{base_name}_{target_name}_{nm}_{usestd}_{useloc}_{noisecoef}_{initp}.png',dpi=300)


# Plot loss
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()

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

grid_size = 200
device='cuda'
xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)

#add a plot with subfigure so all combinations of base and target are shown in one plot
fig, axs = plt.subplots(3, 6, figsize=(18,10))
# for base,base_name in zip([base2,base3,base4,base5],):
#     for target,target_name in zip([target1,target2,target3,target8],[]):

for k,base_name in enumerate(['Generalized Gaussian','Gaussian',"Multivariate Student's t",'Gaussian Mixture','Our Method']):
    for i,target_name in enumerate(['Two Moons','Generalized Gaussian','Gaussian Mixtures']):
        try:
            if base_name != 'Our Method':
                if base_name == 'Multivariate Student\'s t' and target_name == 'Two Moons':
                    model = torch.load(f'../../model_Our method_Two Moons_50_False_False_0.01_2.0.pt')
                    base = torch.load(f'../../base_Our method_Two Moons_50_False_False_0.01_2.0.pt')
                    target = torch.load(f'../../target_Our method_Two Moons_50_False_False_0.01_2.0.pt')
                else:
                    model = torch.load(f'../../pmodel_{base_name}_{target_name}.pt')
                    base = torch.load(f'../../pbase_{base_name}_{target_name}.pt')
                    target = torch.load(f'../../ptarget_{base_name}_{target_name}.pt')

            else:
                if target_name == 'Two Moons':
                    
                    model = torch.load('../../model_Our method_Two Moons_50_False_True_0.001_2.0.pt')
                    base = torch.load(f'../../base_Our method_Two Moons_50_False_True_0.001_2.0.pt')
                    target = torch.load(f'../../target_Our method_Two Moons_50_False_True_0.001_2.0.pt')
                elif target_name == 'Generalized Gaussian':
                    model = torch.load('../../model_Our method_Generalized Gaussian_50_True_False_0.01_0.5.pt')
                    base = torch.load(f'../../base_Our method_Two Moons_50_False_True_0.001_2.0.pt')
                    target = torch.load(f'../../target_Our method_Two Moons_50_False_True_0.001_2.0.pt')

                    pass

            model.eval()
            
            log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
            prob = torch.exp(log_prob)
            prob[torch.isnan(prob)] = 0
            prob[torch.isinf(prob)] = 0
            j = k + 1
            axs[i, j].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
            axs[i, j].set_aspect('equal', 'box')
            axs[i, j].tick_params(axis='x', labelsize=14)  # Increase x-ticks font size
            axs[i, j].tick_params(axis='y', labelsize=14)  # Increase y-ticks font size
            # Remove y ticks from second and third columns
            
            if i == j-1:
                log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
                prob = torch.exp(log_prob)
                prob[torch.isnan(prob)] = 0
                prob[torch.isinf(prob)] = 0
                axs[i, 0].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
                axs[i, 0].set_aspect('equal', 'box')
                axs[i, 0].tick_params(axis='x', labelsize=14)  # Increase x-ticks font size
                axs[i, 0].tick_params(axis='y', labelsize=14)  # Increase y-ticks font size
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                
            if j in [1, 2,3,4]:
                axs[i, j].set_yticks([])
            
            # Remove x ticks from first and second rows
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
            if i in [0, 1]:
                axs[i, j].set_xticks([])
                axs[i, 0].set_xticks([])
            axs[i, 0].set_xticks([])
            axs[i, 0].set_yticks([])



        except:
            pass

titles = ["", 'Generalized Gaussian','Gaussian',"Multivariate Student's t",'Gaussian mixture','Our Method']
for col in range(6):
    axs[0, col].set_title(titles[col],fontsize=17,color='green')

fig.text(0.57, 1.01, "Base", ha='center', fontsize=35,color='green')
row_labels = ["Two moons", r"GGD $\beta$=0.5", "Gaussians on a ring"]
for row in range(3):
    axs[row, 0].set_ylabel(row_labels[row], rotation=90, labelpad=9, verticalalignment='center', fontsize=18,color='blue')

from matplotlib.lines import Line2D
line_xpos = 0.182
fig.text(-0.025, 0.5, "Target", va='center',rotation=90, fontsize=35,color='blue')
line = Line2D([line_xpos, line_xpos], [0.04, 0.95], transform=fig.transFigure, color='red', linestyle='--',lw=3)
fig.lines.append(line)
plt.tight_layout()
plt.savefig('all.png',dpi=300)
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



target = NealsFunnel()
grid_size = 200
device='cuda'
xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)

for nm in [6]:
    for usestd in [True,False]:
        for useloc in [True,False]:
            for noisecoef in [0.001,0.01,0.1,0.2,0.3,0.4,0.5]:
                for initp in [0.5,1.,1.5,2.,2.5,3.,3.5,4.,4.5,5.]:
                    base_name = 'Our method'
                    for target,target_name in zip([target1,target2,target3,target8,target9],['Two Moons',r"Student's t, $\nu$=1 (Cauchy)",'Generalized Gaussian','Gaussian Mixtures','Ring Mixture']):
                        try:
                            with torch.no_grad():
                                target = torch.load(torch.load(f'/home/samiri/SynDG/ptarget_{base_name}_{target_name}_{nm}_{usestd}_{useloc}_{noisecoef}_{initp}.pt'))                        
                                base = torch.load(f'/home/samiri/SynDG/pbase_{base_name}_{target_name}_{nm}_{usestd}_{useloc}_{noisecoef}_{initp}.pt')
                                model = torch.load(f'/home/samiri/SynDG/pmodel_{base_name}_{target_name}_{nm}_{usestd}_{useloc}_{noisecoef}_{initp}.pt')
                                model.eval()
                                fig,axs = plt.subplots(8, 7, figsize=(1,2))
                                log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
                                prob = torch.exp(log_prob)
                                prob[torch.isnan(prob)] = 0
                                prob[torch.isinf(prob)] = 0
                                
                                axs[0].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
                                axs[0].set_aspect('equal', 'box')
                                log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
                                prob = torch.exp(log_prob)
                                prob[torch.isnan(prob)] = 0
                                prob[torch.isinf(prob)] = 0
                                axs[1].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
                                axs[1].set_aspect('equal', 'box')
                                axs[1].set_title(f'./target_{base_name}_{target_name}_{nm}_{usestd}_{useloc}_{noisecoef}_{initp}')
                        except Exception as e:
                            print(e)




# %%
