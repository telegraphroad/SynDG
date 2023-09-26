# %%
import numpy as np
import pandas as pd
from scipy.stats import t
import torch
import numpy as np
from matplotlib import gridspec
import copy

import normflows as nf

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import gc
from normflows.experiments.flowslib import planar, radial, nice, rnvp, nsp, iaf

class MyDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        # Convert row to tensor
        row = self.dataframe.iloc[index]
        row_tensor = torch.tensor(row.values, dtype=torch.float)
        
        return row_tensor

def compute_average_grad_norm(model):
    total_norm = 0.0
    num_parameters = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num_parameters += 1
    return (total_norm / num_parameters) ** 0.5

np.random.seed(42)

# Number of samples
n_samples = 50000

# Generate numerical features from a mixture of three Gaussians
mu = [-3, 0, 3]  # Means of the Gaussians
sigma = [1, 1, 1]  # Standard deviations of the Gaussians
weights = [0.3, 0.4, 0.3]  # Weights of the Gaussians

numerical_features = []
for _ in range(3):  # Repeat for each numerical feature
    # Create an array with the component choice for each data point
    component = np.random.choice(len(weights), size=n_samples, p=weights)
    # Generate the data for this feature
    feature = np.fromiter((np.random.normal(mu[c], sigma[c]) for c in component), dtype=np.float64, count=n_samples)
    numerical_features.append(feature)

numerical_features = np.array(numerical_features).T

# Generate categorical features from a Gaussian and a Student's t-distribution
cat1 = np.random.normal(0, 1, size=n_samples)
cat2 = t.rvs(20, size=n_samples)  # Student's t-distribution with 20 degrees of freedom

# Map continuous values to categories
cat1 = np.clip(np.digitize(cat1, np.linspace(-3, 3, 6)), 0, 6)
cat2 = np.clip(np.digitize(cat2, np.linspace(-3, 3, 3)), 0, 3)

# Combine numerical and categorical features into a single DataFrame
data = pd.DataFrame(
    np.concatenate([numerical_features], axis=1),
    columns=['num_feature1', 'num_feature2', 'num_feature3']
)

# Convert categorical features to integer type
#data[['cat_feature1', 'cat_feature2']] = data[['cat_feature1', 'cat_feature2']].astype(int)

print(data.head())

    
# Create the Dataset and DataLoader
my_dataset = MyDataset(data)


enable_cuda = True

device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

loc = torch.zeros((2, 2))  # 2x2 tensor filled with 0s
scale = torch.ones((2, 2))  # 2x2 tensor filled with 1s
p = 2.0  # Shape parameter for Gaussian

# Create the 2-dimensional instance
loss_arr = []
tests_arr = []
_closs = 1e10
_stalecounter = 0
import sys

_nl = int(sys.argv[1])
_w = int(sys.argv[2])
_ml = int(sys.argv[3])
lr = float(sys.argv[4])

# for nl in list(reversed([8,16,32,48,64,80,100,120,140,180,220,280,320])):
for nl in [_nl]:
    for w in [_w]:
        for ml in [_ml]:
    # for w in list(reversed([64,128,192,256,378,512,768,1024,2048,4096,8192])):    
            try:
                _closs = 1e10
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    del model, optimizer, flows, base
                except:
                    pass
                dataloader = DataLoader(my_dataset, batch_size=256, shuffle=True)
                print(nl,ml,w)
                num_layers = nl
                flows = []
                latent_size = 3
                b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
                flows = []
                for i in range(num_layers):
                    lay = [latent_size] + [w]*ml + [latent_size]
                    s = nf.nets.MLP(lay, init_zeros=True)

                    
                    t = nf.nets.MLP(lay, init_zeros=True)
                    if i % 2 == 0:
                        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
                    else:
                        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
                    flows += [nf.flows.ActNorm(latent_size)]
                base = nf.distributions.base.DiagGaussian(latent_size)
                trnbl = True
                base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=20, rand_p=True, noise_scale=0.05, dim=latent_size,loc=0,scale=1.,p=2.,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)

                categoricals = {}
                categoricals[3] = 7
                categoricals[4] = 4
                model = nf.NormalizingFlow(base, flows)
                model = model.to(device)



                max_iter = 35
                num_samples = 2 ** 12
                show_iter = 250


                loss_hist = np.array([])

                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr/10)
                #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
                max_norm = 0.5
                adjust_rate = 0.01
                
                best_params = copy.deepcopy(model.state_dict())
                bestloss = 1e10
                _stalecounter = 0
                for it in tqdm(range(max_iter)):
                    for i, features in enumerate(dataloader, 0):
                        optimizer.zero_grad()
                        x = features.to(device)
                        try:
                            loss = model.forward_kld(x, robust=False)    
                            # l2_lambda = 0.001  # The strength of the regularization
                            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                            # loss = loss + l2_lambda * l2_norm
                            if ~(torch.isnan(loss) | torch.isinf(loss)):
                                loss.backward()
                                # avg_grad = 0.0
                                # num_params = 0
                                # # for name, param in model.named_parameters():
                                # #     if param.grad is not None:
                                # #         avg_grad += param.grad.data.abs().mean().item()
                                # #         num_params += 1
                                # # avg_grad /= num_params
                                
                                # # avg_norm = avg_grad
                                # # if avg_norm > max_norm:
                                # #     max_norm += adjust_rate
                                # # else:
                                # #     max_norm -= adjust_rate
                                # # with torch.no_grad():
                                # #     #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                                # #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                                # # print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
                                # # print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())
                                # if (it + 1) % 100 == 0:
                                #     #print(f'+++++++++++++ avgnorm: {avg_norm},{avg_grad},{bestloss}')
                                #     #print('=======mins: ',model.q0.loc.min().item(),model.q0.scale.min().item(),model.q0.p.min().item())
                                #     #print('=======maxs: ',model.q0.loc.max().item(),model.q0.scale.max().item(),model.q0.p.max().item())

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
                                    #print(f'Epoch {it+1}, Max Gradient: {max_grad:.6f}, Min Gradient: {min_grad:.6f}, Avg Gradient: {avg_grad:.6f}')
                                optimizer.step()
                                with torch.no_grad():
                                    if loss.item()<bestloss:
                                        _s,_ = model.sample(1000)
                                        if (torch.isnan(_s).sum()==0 & torch.isinf(_s).sum()==0):
                                        
                                            bestloss = copy.deepcopy(loss.item())
                                            best_params = copy.deepcopy(model.state_dict())
                            loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
                            print(f'Epoch {it+1}, Iter {i+1}, Loss: {loss.item():.6f}, Best Loss: {bestloss.item():.6f}')
                        except Exception as e:
                            if True:
                                #print(e)
                                with torch.no_grad():
                                    model.load_state_dict(best_params)
                    
                    print(f'Epoch {it+1}, Iter {i+1}, Loss: {bestloss:.6f}')
                    if _closs > bestloss:
                        _closs = bestloss
                        _stalecounter = 0
                    else:
                        if bestloss > 1e9:
                            _stalecounter += 5
                        else:
                            _stalecounter += 1
                    if _stalecounter > 10:
                        print('STALLED')
                        _stalecounter = 0
                        break
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

                loss_arr.append([nl,w,bestloss])
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

                del dataloader
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

                del model, optimizer, flows, base
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

                b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
                flows = []
                for i in range(num_layers):
                    lay = [latent_size] + [w]*ml + [latent_size]
                    s = nf.nets.MLP(lay, init_zeros=True)

                    
                    t = nf.nets.MLP(lay, init_zeros=True)
                    if i % 2 == 0:
                        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
                    else:
                        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
                    flows += [nf.flows.ActNorm(latent_size)]
                base = nf.distributions.base.DiagGaussian(latent_size)
                trnbl = True
                base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=20, rand_p=True, noise_scale=0.05, dim=latent_size,loc=0,scale=1.,p=2.,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)

                categoricals = {}
                categoricals[3] = 7
                categoricals[4] = 4
                model = nf.NormalizingFlow(base, flows)
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

                model = model.to(device)
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

                model.load_state_dict(best_params)
                model.eval()
                from scipy.stats import ks_2samp, chi2_contingency
                ds_gn = model.sample(n_samples)[0].detach().cpu().numpy()
                del model
                ds_gn = pd.DataFrame(ds_gn, columns=data.columns)
                
                # List of numerical and categorical feature names
                num_features = ['num_feature1', 'num_feature2', 'num_feature3']
                
                import matplotlib.pyplot as plt

                # List of numerical and categorical feature names
                num_features = ['num_feature1', 'num_feature2', 'num_feature3']

                # Create a figure with 5 subplots (one for each feature)
                fig, axs = plt.subplots(3, figsize=(10, 20))

                # Plot the numerical features
                for i, feature in enumerate(num_features):
                    axs[i].hist(data[feature], bins=30, alpha=0.5, label='Original')
                    axs[i].hist(ds_gn[feature], bins=30, alpha=0.5, label='Generated')
                    axs[i].set_title(f'Histogram of {feature}')
                    axs[i].legend()

                # Plot the categorical features
                # Adjust the layout and show the plot
                plt.tight_layout()
                plt.show()    
                print('plotted!')
                plt.savefig(f'./hist_{nl}_{w}_{ml}_{lr}.png')
                del ds_gn
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(e)
                pass

# %%
# model.load_state_dict(best_params)
# model.eval()
# from scipy.stats import ks_2samp, chi2_contingency
# ds_gn = model.sample(n_samples)[0].detach().cpu().numpy()
# ds_gn = pd.DataFrame(ds_gn, columns=data.columns)
# # List of numerical and categorical feature names
# num_features = ['num_feature1', 'num_feature2', 'num_feature3']
# cat_features = ['cat_feature1', 'cat_feature2']

# # Perform KS test for each numerical feature
# for feature in num_features:
#     original = data[feature]
#     generated = ds_gn[feature]
    
#     ks_stat, p_value = ks_2samp(original, generated)

#     print(f'KS test for {feature}:')
#     print(f'Statistics={ks_stat}, p={p_value}\n')

# # # Perform Chi-Square test for each categorical feature
# # for feature in cat_features:
# #     # Create contingency table
# #     contingency_table = pd.crosstab(data[feature], ds_gn[feature])
    
# #     chi2, p_value, _, _ = chi2_contingency(contingency_table)

# #     print(f'Chi-Square test for {feature}:')
# #     print(f'Statistics={chi2}, p={p_value}\n')
# # # %%
# # model.sample(n_samples)[0].cpu().detach().numpy()
# # # %%

# # %%
# from scipy.stats import ks_2samp, chi2_contingency, mannwhitneyu, ttest_ind

# # List of numerical and categorical feature names
# num_features = ['num_feature1', 'num_feature2', 'num_feature3']
# cat_features = ['cat_feature1', 'cat_feature2']

# # Initialize a list to store the results
# results = []

# # Perform KS test, Mann-Whitney U test, and t-test for each numerical feature
# for feature in num_features:
#     original = data[feature]
#     generated = ds_gn[feature]
    
#     ks_stat, ks_p = ks_2samp(original, generated)
#     mw_stat, mw_p = mannwhitneyu(original, generated, alternative='two-sided')
#     t_stat, t_p = ttest_ind(original, generated)
    
#     # Interpret the results
#     ks_result = 'Close' if ks_p > 0.05 else 'Not close'
#     mw_result = 'Close' if mw_p > 0.05 else 'Not close'
#     t_result = 'Close' if t_p > 0.05 else 'Not close'
    
#     results.append([feature, 'KS', ks_stat, ks_p, ks_result])
#     results.append([feature, 'Mann-Whitney U', mw_stat, mw_p, mw_result])
#     results.append([feature, 'T-test', t_stat, t_p, t_result])

# # Perform Chi-Square test for each categorical feature

# # Convert the results to a DataFrame
# results_df = pd.DataFrame(results, columns=['Feature', 'Test', 'Statistic', 'p-value', 'Result'])

# print(results_df)
# # %%

# %%
import numpy as np
import pandas as pd
from scipy.stats import t
import torch
import numpy as np
from matplotlib import gridspec
import copy

import normflows as nf

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import copy
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import gc
from normflows.experiments.flowslib import planar, radial, nice, rnvp, nsp, iaf

# %%
