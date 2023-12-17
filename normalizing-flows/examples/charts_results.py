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


# %%
enable_cuda = True

device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
trnbl = True


for _b in ['ggd','ggd mixture','student t','gaussian']:
    for _betab in [2.]:
        for _vsh in [6.]:
            target = nf.distributions.target_extended.NealsFunnel(v1shift=_vsh)
            # target = nf.distributions.NealsFunnel(v1shift=0.,v2shift=0.)

            # target = NealsFunnel()
            try:
                #_l = target.sample(10000).median().item()
                _l = target.sample(50000).median(axis=0).values.cpu().detach().numpy()
                _s = target.sample(50000).std(axis=0).cpu().detach().numpy()

            except:
                _l = target.sample(torch.rand(50000).size()).median(axis=0).values.cpu().detach().numpy()
                _s = target.sample(torch.rand(50000).size()).std(axis=0).cpu().detach().numpy()

            # Define 2D Gaussian base distribution
            loc = torch.zeros((2, 2))  # 2x2 tensor filled with 0s
            scale = torch.ones((2, 2))  # 2x2 tensor filled with 1s
            p = 2.0  # Shape parameter for Gaussian

            trnbl = True

            #base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=False, noise_scale=0.2, dim=2,loc=_l,scale=target.sample(50000).std(axis=0).detach().cpu().numpy(),p=_betab,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)
            #base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=False, noise_scale=0.2, dim=2,loc=0.,scale=1.,p=_betab,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)
            #base = nf.distributions.base_extended.MultivariateStudentT(shape=(2,),df=2.,trainable=False).cuda()
            if _b == 'student t':
                base = nf.distributions.base_extended.MultivariateStudentTDist(degree_of_freedom=2.,dim=2,trainable=True,device='cuda').cuda()
            elif _b == 'ggd mixture':
                trnbl = True
                base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=20, rand_p=True, noise_scale=0.5, dim=2,loc=_l,scale=_s,p=_betab,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)    
            elif _b == 'ggd':
                trnbl = True
                base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=1, rand_p=True, noise_scale=0.5, dim=2,loc=0,scale=1.,p=_betab,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)    
            elif _b == 'gaussian':
                base = nf.distributions.base.DiagGaussian(2,trainable=trnbl).cuda()

            # Define list of flows
            num_layers = 14
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



            model = model.to(device)

            # Define target distribution
            def check_model_params(model):
                for name, param in model.named_parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f'Parameter {name} has NaNs or infs')


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

            # Train model
            max_iter = 1000
            num_samples = 2 ** 12
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
            models.append([_b,_betab,_vsh,model,base,target])
            torch.save(models,'models_results.pt')

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
import seaborn as sns

grid_size = 200
device='cuda'

models = torch.load('../../models_results.pt')
sns.set_style("darkgrid")
fig, axs = plt.subplots(1, 5, figsize=(15, 4.2))
i,j = 0,0
target = models[0][5]
xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
zz = zz.to(device)
log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
prob = torch.exp(log_prob)
prob[torch.isnan(prob)] = 0
sns.set_style("darkgrid")

axs[j].pcolormesh(xx, yy, prob.data.numpy(),label='test', cmap='coolwarm')
#axs[i,j].gca().set_aspect('equal', 'box')
axs[j].set_title('Target',fontsize=23)
axs[j].set_aspect('equal', 'box')
axs[j].tick_params(axis='x', labelsize=14)  # Increase x-ticks font size
axs[j].tick_params(axis='y', labelsize=14)  # Increase y-ticks font size
axs[j].set_xticks([])
axs[j].set_yticks([])
sns.set_style("darkgrid")
# i = 1
# axs[i,j].pcolormesh(xx, yy, prob.data.numpy(),label=_b, cmap='coolwarm')
# #axs[i,j].gca().set_aspect('equal', 'box')
# axs[i, j].set_aspect('equal', 'box')
# axs[i, j].tick_params(axis='x', labelsize=14)  # Increase x-ticks font size
# axs[i, j].tick_params(axis='y', labelsize=14)  # Increase y-ticks font size
# axs[i, j].set_xticks([])
# axs[i, j].set_yticks([])

i = 0
j = 1
for m in models:
    _b = m[0]
    model = m[3]
    log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0
    if _b == 'student t':
        b = "Student's t base"
        j = 2
    elif _b == 'ggd mixture':
        b = 'Our method'
        j = 4
    elif _b == 'ggd':
        b = 'GGD base'
        j = 3
    elif _b == 'gaussian':
        b = 'Gaussian base'
        j = 1
    
    axs[j].pcolormesh(xx, yy, prob.data.numpy(),label=_b, cmap='coolwarm')
    #axs[i,j].gca().set_aspect('equal', 'box')
    axs[j].set_title(b,fontsize=25)
    axs[j].set_aspect('equal', 'box')
    axs[j].tick_params(axis='x', labelsize=14)  # Increase x-ticks font size
    axs[j].tick_params(axis='y', labelsize=14)  # Increase y-ticks font size
    axs[j].set_xticks([])
    axs[j].set_yticks([])

sns.set_style("darkgrid")    

# i,j = 1,1
# for m in models:
#     _b = m[0]
#     base = m[4]
#     log_prob = base.log_prob(zz).to('cpu').view(*xx.shape)
#     prob = torch.exp(log_prob)
#     prob[torch.isnan(prob)] = 0
#     if _b == 'student t':
#         b = "Student's t base"
#         j = 2
#     elif _b == 'ggd mixture':
#         b = 'Our method'
#         j = 4
#     elif _b == 'ggd':
#         b = 'GGD base'
#         j = 3
#     elif _b == 'gaussian':
#         b = 'Gaussian base'
#         j = 1

    
#     axs[i,j].pcolormesh(xx, yy, prob.data.numpy(),label=_b, cmap='coolwarm')
#     #axs[i,j].gca().set_aspect('equal', 'box')
#     axs[i, j].set_aspect('equal', 'box')
    
#     axs[i, j].set_xticks([])
#     axs[i, j].set_yticks([])

row_labels = ["Trained flow"]
for row in range(1):
    axs[4].yaxis.set_label_position("right")  # This line sets the label position to the right
    axs[4].set_ylabel(row_labels[row], rotation=270, labelpad=9, verticalalignment='center', fontsize=25)
    
line_xpos = 0.20  # Adjust this as needed

from matplotlib.lines import Line2D

# Add the line to the figure
line = Line2D([line_xpos, line_xpos], [0.175, 0.825], transform=fig.transFigure, color='red', linestyle='--',lw=3)
fig.lines.append(line)
plt.subplots_adjust(hspace=0.05, wspace=0.01)
plt.tight_layout()
sns.set_style("darkgrid")
plt.savefig('2d_results.png',bbox_inches='tight')

# %%
def compute_arealoglog(data_true, data_synth):
    """
    Computes the Area under the log-log plot.
    data_true: 2D-Array containing the true data
    data_synth: 2D-Array containing synthetic data
    """
    n = len(data_true)
    area = 0
    for j in range(n):
        i = j + 1
        area += np.abs( np.log(np.quantile(np.abs(data_true), 1 - i/n)) - np.log(np.quantile(np.abs(data_synth), 1 - i/n))) * np.log((i + 1)/i)
    return area

def compute_tvar(data_true, data_synth,dim,alpha = 0.95):
    

    tvar_differences = []
    for component in range(dim):
        sorted_abs_samps_flow = np.sort(np.abs(data_synth[:, component]))
        sorted_abs_data_test = np.sort(np.abs(data_true[:, component]))
        tvar_flow = 1/(1-alpha) * np.mean(sorted_abs_samps_flow[int(alpha*len(data_synth)): ])
        tvar_test = 1/(1-alpha) * np.mean(sorted_abs_data_test[int(alpha*len(data_true)): ])

        # print(f"Component {component + 1}: tVaR in test data: {np.round(tvar_test, 2)}, tVaR in flow samples: {np.round(tvar_flow, 2)}")
        tvar_differences.append(np.abs(tvar_test - tvar_flow))

    return tvar_differences

try:
    del xx,yy,zz,log_prob,prob,model
except:
    pass

for m in models:
    _b = m[0]
    model = m[3]
    target = m[5]
    tds = target.sample(50000).cpu().detach().numpy()
    fds = model.sample(50000)[0].cpu().detach().numpy()
    areas = []
    for i in range(2):

        a = compute_arealoglog(tds[:,i], fds[:,i])  
        areas.append(a)
    print(_b,np.mean(areas))
    tv = compute_tvar(tds,fds,2)
    print(tv)
# %%


    # self.flow.eval()
    # with torch.no_grad():
    #     if sampling:
    #         self.samps_flow = self.flow.to(device).sample(num_samples).detach().cpu().numpy()
    #         if self.model in ["mTAF", "mTAF(fix)", "res_gTAF"]:
    #             self.samps_flow = self.samps_flow[:, self.inv_perm]


    # # compute marginal areas
    # area = []
    # for j in range(self.D):
    #     marginal_area = compute_arealoglog(self.data_test[:num_samples, j], self.samps_flow[:, j])
    #     area.append(np.round(marginal_area, 5))

    # if self.data=="":
    #     Path(f"{self.setting}/area/").mkdir(parents=True, exist_ok=True)
    #     PATH = f"{self.setting}/area/{self.flowtype}{self.model}.txt"
    # else:
    #     PATH = f"results/{self.flowtype}/{self.model}/{self.data}/{self.num_layers}layers/area.txt"

    # if self.track_results:
    #     with open(PATH, "a") as f:
    #         f.write(" ".join(str(e) for e in area) + "\n")
    # else:
    #     print(f"({self.model}) Average area under curve: {np.round(np.mean(area), 2)}")

    # return area


i = 0
j = 1
for m in models:
    _b = m[0]
    model = m[3]
    log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob = log_prob
    prob[torch.isnan(prob)] = 0
    print(_b,-prob.sum().item())
# %%
