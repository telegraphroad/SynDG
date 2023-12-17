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
import sys

from normflows.experiments.flowslib import planar, radial, nice, rnvp, nsp, iaf, residual

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import numpy as np
from skbio import DistanceMatrix
from skbio.stats.distance import permanova
import gower
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import pandas as pd
import pandas as pd
import numpy as np
import gower
from skbio.stats.distance import permanova
from skbio import DistanceMatrix
from sklearn.ensemble import GradientBoostingClassifier
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
from fastDP import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from sklearn.model_selection import KFold
from sklearn import model_selection, mixture, preprocessing, preprocessing

import math

import warnings
warnings.filterwarnings("ignore")

# %%

def calculate_privacy_loss(iterations, epsilon, delta, batch_size, dataset_size):
    # Calculate the Poisson subsampling rate
    q = batch_size / dataset_size

    # Scale epsilon according to the Poisson subsampling rate
    epsilon = q * epsilon

    # Calculate the total privacy loss according to the advanced composition theorem
    epsilon_prime = math.sqrt(2 * iterations * math.log(1/delta)) * epsilon + iterations * epsilon * (math.e**epsilon - 1)
    delta_prime = iterations * delta

    return epsilon_prime, delta_prime


import math

def calculate_privacy_loss2(iterations, epsilon, delta, batch_size, dataset_size):
    # Calculate the Poisson subsampling rate
    q = batch_size / dataset_size

    # Calculate the epsilon per iteration adjusted for Poisson subsampling
    # This approach assumes that the privacy loss scales linearly with the subsampling rate, which is an approximation.
    epsilon_per_iteration = q * epsilon

    # Use the advanced composition theorem to calculate the total privacy loss
    # The theorem states that for (epsilon, delta)-DP mechanisms run k times,
    # the total privacy loss is bounded by (epsilon_prime, delta_prime), where:
    epsilon_prime = math.sqrt(2 * iterations * math.log(1/delta)) * epsilon_per_iteration \
                    + iterations * epsilon_per_iteration * (math.e - 1)  # This part is an approximation

    # The growth of delta under composition is not a simple multiplication in the advanced composition theorem,
    # but for simplicity, we can use this as an upper bound. For more precise calculations, a tighter bound should be used.
    delta_prime = iterations * delta  # This should be calculated using a tighter composition theorem or numerical methods.

    return epsilon_prime, delta_prime

import math

def calculate_privacy_loss3(iterations, epsilon, delta, batch_size, dataset_size):
    # Calculate the Poisson subsampling rate
    q = batch_size / dataset_size

    # Scale epsilon according to the Poisson subsampling rate
    epsilon_i = q * epsilon

    # Calculate the total privacy loss according to the advanced composition theorem
    epsilon_prime = math.sqrt(2 * math.log(1/delta) * iterations * epsilon_i ** 2) + iterations * epsilon_i
    delta_prime = iterations * delta

    return epsilon_prime, delta_prime


import math

def calculate_privacy_loss4(iterations, epsilon, delta, batch_size, dataset_size):
    # Calculate the Poisson subsampling rate
    q = batch_size / dataset_size
    
    # Amplify epsilon using Poisson subsampling
    # The amplification effect depends on the specific subsampling mechanism
    # For Poisson subsampling, the amplification can be more complex
    # and might require numerical methods for tight analysis.
    # Here we use a simple amplification bound that works for small epsilon.
    amplified_epsilon = epsilon * q
    
    # Calculate the total privacy loss according to the advanced composition theorem
    # Note: This assumes (epsilon, delta)-DP of each iteration, not (q*epsilon, q*delta)-DP
    epsilon_prime = math.sqrt(2 * iterations * math.log(1.0 / delta)) * amplified_epsilon + iterations * amplified_epsilon * (math.e ** amplified_epsilon - 1)
    delta_prime = iterations * delta


    return epsilon_prime, delta_prime

def calculate_privacy_loss5(iterations, epsilon, delta, batch_size, dataset_size):
    # Calculate the Poisson subsampling rate
    q = batch_size / dataset_size

    # Scale epsilon according to the Poisson subsampling rate
    epsilon_scaled = q * epsilon

    # Calculate the total privacy loss according to the advanced composition theorem
    epsilon_prime = math.sqrt(2 * iterations * math.log(1/delta)) * epsilon_scaled + iterations * epsilon_scaled * (math.e**epsilon_scaled - 1)
    delta_prime = iterations * delta * q

    return epsilon_prime, delta_prime

def calculate_privacy_loss6(iterations, epsilon, delta, batch_size, dataset_size):
    # Calculate the Poisson subsampling rate
    q = batch_size / dataset_size

    # Calculate the privacy loss per iteration
    rdp = lambda x: x * q

    # Calculate the privacy loss over all iterations using the moments accountant
    total_privacy_loss = 0.0
    for i in range(iterations):
        total_privacy_loss += rdp(1.0)

    # Calculate the privacy loss bound using the moments accountant
    epsilon_prime = total_privacy_loss + math.sqrt(2 * math.log(1.25 / delta)) * iterations

    return epsilon_prime, delta


def calculate_privacy_loss7(iterations, epsilon, delta, batch_size, dataset_size):
    # Calculate the Poisson subsampling rate
    q = batch_size / dataset_size
    
    # Adjust epsilon for Poisson subsampling (this is a simplified version and may not be accurate)
    # Note: The adjustment here is very rough; the real adjustment is more complex
    epsilon = 2 * q * epsilon
    
    # Calculate the total privacy loss according to the advanced composition theorem
    epsilon_prime = math.sqrt(2 * iterations * math.log(1 / delta)) * epsilon + iterations * epsilon * (math.exp(epsilon) - 1)
    
    # Calculate the new delta which is the same in this case (this is not strictly correct, but for simplicity)
    delta_prime = delta  # This is not correct, but the correct calculation requires more information
    
    return epsilon_prime, delta_prime

def calculate_privacy_loss8(iterations, epsilon, delta, batch_size, dataset_size):
    # Calculate the Poisson subsampling rate
    q = batch_size / dataset_size

    # Privacy amplification by subsampling: Adjust epsilon for each iteration
    # This adjustment can be problem-specific and may require a more complex function in practice
    adjusted_epsilon = epsilon * q

    # Advanced composition theorem for overall epsilon_prime and delta_prime
    # Note: The formula used here assumes delta is small and the adjusted epsilon per step is also small
    epsilon_prime = math.sqrt(2 * iterations * math.log(1/delta)) * adjusted_epsilon + iterations * adjusted_epsilon * (math.exp(adjusted_epsilon) - 1)
    delta_prime = delta + iterations * (math.exp(adjusted_epsilon) - 1) * delta / (math.exp(adjusted_epsilon))

    return epsilon_prime, delta_prime

def calculate_privacy_loss9(iterations, epsilon, delta, batch_size, dataset_size):
    # Calculate the Poisson subsampling rate
    q = batch_size / dataset_size

    # Calculate the total privacy loss according to the advanced composition theorem
    rho = batch_size / dataset_size
    max_lambda = iterations * rho * epsilon

    # Calculate the advanced composition
    epsilon_prime = math.sqrt(2 * math.log(1/delta) + 2 * max_lambda) + iterations * epsilon * (math.exp(epsilon) - 1)
    delta_prime = iterations * delta

    return epsilon_prime, delta_prime

class CSVDataset(Dataset):
    def __init__(self, file_path, categorical_column_names, transform=None):
        self.data = pd.read_csv(file_path,header=None)
        # scaler = preprocessing.StandardScaler()
        # self.data = pd.DataFrame(scaler.fit_transform(self.data))
        
        
        self.transform = transform
        self.label_encoders = {}
        self.categorical_column_names = categorical_column_names
        
        # Encode the categorical columns
        for column_name in self.categorical_column_names:
            self.label_encoders[column_name] = LabelEncoder()
            categorical_column = self.data[column_name].astype(float)
            encoded_categorical_column = self.label_encoders[column_name].fit_transform(categorical_column)
            self.data[column_name] = encoded_categorical_column
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # Retrieve item at the given index
        item = self.data.iloc[index]
        
        # Apply data transformation if provided
        if self.transform:
            item = self.transform(item)
        
        # Convert item to tensors if needed
        item = torch.tensor(item).float()
        
        return item
    
    def calculate_feature_means(self):
        category_means = []
        
        # Calculate means for each category
        for column_name in self.data.columns:
            category_means.append(self.data[[column_name]].mean().iloc[0])
            
        return category_means
    
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


#b = torch.Tensor([1 if i % 2 == 0 else 0 for i in [0]])

#X = X.drop(['Class'],1)
# from sklearn.preprocessing import MinMaxScaler
# from collections import Counter

# # Original dataset
# from normflows.utils import data_utils

# X = pd.read_csv('power.csv').drop(['Unnamed: 0'],axis=1)


# xcol = X.columns
# # for ii in range(len(categorical)):
# #     X[X.columns[categorical[ii]]] = X[X.columns[categorical[ii]]] * lcm / categorical_qlevels[ii]
# X=X.values
# dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
# num_samples = 2**9
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples,num_workers=4)
# train_iter = iter(train_loader)
latent_size = 10
categorical = []
categorical_qlevels = []
catlevels = []
lcm = 0
vlayers = []
#b = torch.Tensor([1 if i % 2 == 0 else 0 for i in [0]])

#X = X.drop(['Class'],1)

# Original dataset


categorical = []

# %%
my_dataset = CSVDataset('/home/samiri/SynDG/DOFLOWS/lifesci.csv',categorical)

categorical_qlevels = []
vdeq_categoricals = {int(k): int(v) for k, v in zip(categorical, categorical_qlevels)}
# Number of samples

enable_cuda = True

device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

n_sammple = my_dataset.__len__()
# Create the 2-dimensional instance
loss_arr = []
tests_arr = []
_closs = 1e10
_stalecounter = 0

nl = 10
w = 256
ml = 4
lr = 1e-5
fltyp = 'nice'

vlay = 0
nsamp = 2048
nmodes = 200
rndadd = 0.5
usestd = True
useloc = True
initp = 2.5
batch_size = 199


dp = 'NF'

    # for w in list(reversed([64,128,192,256,378,512,768,1024,2048,4096,8192])):    
_closs = 1e10
torch.cuda.empty_cache()
gc.collect()
try:
    del model, optimizer, flows, base
except:
    pass
dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
num_layers = nl
flows = []
latent_size = len(my_dataset.__getitem__(0))
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
if fltyp == 'nsp':
    flows = nsp(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
elif fltyp == 'rnvp':
    flows = rnvp(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
elif fltyp == 'residual':
    flows = residual(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
elif fltyp == 'nice':
    flows = nice(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
    

trnbl = True
# base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=nmodes, rand_p=True, noise_scale=rndadd, dim=latent_size,loc=list(my_dataset.data.median()) if useloc else 0.,scale=list(my_dataset.data.std()) if usestd else 1.,p=initp,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl,ds=my_dataset)

#model = nf.NormalizingFlow(base, flows)
loss_hists = np.array([])
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr/10)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
# target_epsilon = 1.0
# if dp == 'DPSGD':
#     sigma=get_noise_multiplier(
#                     target_epsilon = target_epsilon,
#                     target_delta = 1.52e-5,
#                     sample_rate = batch_size/(len(dataloader.dataset)*0.9),
#                     epochs = max_iter,
#                 )

#     privacy_engine = PrivacyEngine(
#                 model,
#                 batch_size=batch_size,
#                 sample_size=len(dataloader.dataset)*0.9,
#                 noise_multiplier=sigma,
#                 epochs=max_iter,
#                 clipping_mode='MixOpt',
#                 origin_params=None,
#             )
#     privacy_engine.attach(optimizer)





max_norm = 0.5
adjust_rate = 0.01
num_samples = nsamp
show_iter = 200
max_iter = 15
bestloss = 1e10
_stalecounter = 0
pbar = tqdm(range(max_iter))
# Prepare the data for KFold
data = list(dataloader.dataset)
kf = KFold(n_splits=10)

# To store average log likelihood for each epoch
avg_log_likelihoods = []
loss_hists = []
dctr = -1
rbst = False
falseflag = False
for train_index, test_index in kf.split(data):
    for trunc in ['tanh','thfixedscale']:
        flows = []
        latent_size = len(my_dataset.__getitem__(0))
        b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
        flows = []
        if fltyp == 'nsp':
            flows = nsp(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
        elif fltyp == 'rnvp':
            flows = rnvp(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml,lipschitzconstrained=lipschitzconstrained,min=min,max=max,func=func,boundtranslate=boundtranslate)
        elif fltyp == 'residual':
            flows = residual(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
        elif fltyp == 'nice':
            flows = nice(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
        
        try:
            del model,base,optimizer
            del best_params
            del bestloss
        except:
            pass

        dctr += 1
        # Reset the model and the best loss for each fold
        base = nf.distributions.base.DiagGaussian(latent_size)
        if dp == 'NF':
            cmin = -2.0
            cmax = 2.0   
            import scipy.stats as stats

            # Z-scores for 0.05% and 99.95% of the standard normal distribution
            Z_a = stats.norm.ppf(0.0005)
            Z_b = stats.norm.ppf(0.9995)

            # Since the mean is 0, the equations to solve are:
            # a = Z_a * sigma
            # b = Z_b * sigma

            # Solving these for sigma gives the same result, so we can just use one of them:
            sigma = cmin / Z_a
            sigma = 1.0

            print("New sigma: ", sigma)        
            base = nf.distributions.base_extended.TruncatedNormal(latent_size,0.,sigma, cmin, cmax)
            #base = nf.distributions.base_extended.TruncatedGaussian(latent_size,0.,sigma, cmin, cmax,trainable=False)
            if dctr < 1:
                # pl = []
                # for _i in range(50):
                #     s = base.sample(60000000)
                #     _pl = base.log_prob(s).exp().cpu().detach()
                #     pl.append(_pl)
                #     print(_i,':',np.log(torch.cat(pl).max().item()) - np.log(torch.cat(pl).min().item()))
                # del s
                # pl = torch.cat(pl)
                mn = base.log_prob(base.a.view(1,10).to(torch.float64)).exp()
                mx = base.log_prob(base.mean.view(1,10).to(torch.float64)).exp()
                eps = np.log(mx[0].item()) - np.log(mn[0].item())
                # del pl,_pl
                gc.collect()
                torch.cuda.empty_cache()
                et,dt = calculate_privacy_loss(max_iter, eps, 1e-5, batch_size, len(train_index))
                print(f'final consumed budget in interval {[cmin,cmax]} is {et} with delta {dt}')
                et,dt = calculate_privacy_loss2(max_iter, eps, 1e-5, batch_size, len(train_index))
                print(f'final consumed budget in interval {[cmin,cmax]} is {et} with delta {dt}')
                et,dt = calculate_privacy_loss3(max_iter, eps, 1e-5, batch_size, len(train_index))
                print(f'final consumed budget in interval {[cmin,cmax]} is {et} with delta {dt}')
                et,dt = calculate_privacy_loss4(max_iter, eps, 1e-5, batch_size, len(train_index))
                print(f'final consumed budget in interval {[cmin,cmax]} is {et} with delta {dt}')
                et,dt = calculate_privacy_loss5(max_iter, eps, 1e-5, batch_size, len(train_index))
                print(f'final consumed budget in interval {[cmin,cmax]} is {et} with delta {dt}')
                et,dt = calculate_privacy_loss6(max_iter, eps, 1e-5, batch_size, len(train_index))
                print(f'final consumed budget in interval {[cmin,cmax]} is {et} with delta {dt}')
                et,dt = calculate_privacy_loss7(max_iter, eps, 1e-5, batch_size, len(train_index))
                print(f'final consumed budget in interval {[cmin,cmax]} is {et} with delta {dt}')
                et,dt = calculate_privacy_loss8(max_iter, eps, 1e-5, batch_size, len(train_index))
                print(f'final consumed budget in interval {[cmin,cmax]} is {et} with delta {dt}')
                et,dt = calculate_privacy_loss9(max_iter, eps, 1e-5, batch_size, len(train_index))
                print(f'final consumed budget in interval {[cmin,cmax]} is {et} with delta {dt}')
                truncated = True

        model = nf.NormalizingFlow(base, flows,categoricals=vdeq_categoricals, vardeq_layers=vlay, vardeq_flow_type='shiftscale')
        model = model.to(device)





        loss_hist = np.array([])

        log_likelihoods = []

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr/10)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        target_epsilon = 1.0
        if dp == 'DPSGD':
            sigma=get_noise_multiplier(
                            target_epsilon = target_epsilon,
                            target_delta = 1.52e-5,
                            sample_rate = batch_size/(len(dataloader.dataset)*0.9),
                            epochs = max_iter,
                        )

            privacy_engine = PrivacyEngine(
                        model,
                        batch_size=batch_size,
                        sample_size=len(dataloader.dataset)*0.9,
                        noise_multiplier=sigma,
                        epochs=max_iter,
                        clipping_mode='MixOpt',
                        origin_params=None,
                    )
            privacy_engine.attach(optimizer)










        max_norm = 0.5
        adjust_rate = 0.01

        best_params = copy.deepcopy(model.state_dict())
        bestloss = 1e10
        _stalecounter = 0
        pbar = tqdm(range(max_iter))

        
        bestloss = float('inf')
        best_params = copy.deepcopy(model.state_dict())

        # Create dataloaders for this fold
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]
        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
        ctr = -1
        for it in pbar:
            ctr += 1
            for i, features in enumerate(train_dataloader, 0):
                optimizer.zero_grad()
                x = features.to(device)
                try:
                    loss = model.forward_kld(x, robust=rbst,rmethod='med',truncated=trunc)    
                    if ~(torch.isnan(loss) | torch.isinf(loss)):
                        loss.backward()
                        optimizer.step()
                        with torch.no_grad():
                            if loss.item()<bestloss:
                                _s,_ = model.sample(1000)
                                if (torch.isnan(_s).sum()==0 & torch.isinf(_s).sum()==0):
                                
                                    bestloss = copy.deepcopy(loss.item())
                                    best_params = copy.deepcopy(model.state_dict())
                                    pbar.set_description(f"Processing {it}, best loss={bestloss}")
                    loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
                    pbar.set_description(f"Processing {it}, best loss={bestloss}")
                except Exception as e:
                    if True:
                        #print(e)
                        with torch.no_grad():
                            model.load_state_dict(best_params)
            
            scheduler.step(bestloss)

            with torch.no_grad():
                model.eval()
                ds_gn = model.sample(len(my_dataset.data))[0].detach().cpu().numpy()
                ds_gn = pd.DataFrame(ds_gn, columns=my_dataset.data.columns)
                ds_gn.replace([np.inf, -np.inf], np.nan, inplace=True)
                ds_gn.dropna(inplace=True)
                ds_gn.dropna(axis=1, inplace=True)
                ds_gn.dropna(inplace=True)
                dict_dtype = my_dataset.data.dtypes.apply(lambda x: x.name).to_dict()
                ds_gn = ds_gn.astype(dict_dtype)

                #ds_gn.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_{dp}_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}_{nsamp}_{nmodes}_{rndadd}_{useloc}_{usestd}_{initp}.csv')
                my_dataset.data.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_gen.csv')
                nan_or_inf_df = ds_gn.isna() | np.isinf(ds_gn)

                # Assuming real_data and synthetic_data are your dataframes
                feature_names = my_dataset.data.columns

                # List of categorical features
                categorical_features = []

                fig, axs = plt.subplots(2, 5, figsize=(20, 5))

                for i, ax in enumerate(axs.flatten()):
                    if i < len(feature_names):
                        feature_name = feature_names[i]

                        # If the feature is categorical
                        if i in categorical_features:
                            real_counts = my_dataset.data[feature_name].value_counts()
                            synthetic_counts = ds_gn[feature_name].value_counts()
                            all_categories = list(set(real_counts.index) | set(synthetic_counts.index))
                            
                            ax.bar(all_categories, [real_counts.get(category, 0) for category in all_categories], color='blue', alpha=0.5, label='Real')
                            ax.bar(all_categories, [synthetic_counts.get(category, 0) for category in all_categories], color='red', alpha=0.5, label='Synthetic')
                        else:
                            sns.kdeplot(my_dataset.data[feature_name], ax=ax, color='blue', label='Real')
                            sns.kdeplot(ds_gn[feature_name], ax=ax, color='red', label='Synthetic')

                        ax.set_title(feature_name)
                        ax.legend()
                fig.suptitle(f'Fold {it}', fontsize=15)            
                plt.tight_layout()
                plt.savefig(f'/home/samiri/SynDG/DOFLOWS/images/lifesci_{trunc}_{ctr}_NICE.png')
                plt.close()
                model.train()  
                del ds_gn





            
            pbar.set_description(f"Processing {it}, best loss={bestloss}")
            with torch.no_grad():
                model.eval()
                total_log_likelihood = 0
                for i, features in enumerate(test_dataloader, 0):
                    x = features.to(device)
                    total_log_likelihood += -model.forward_kld(x, robust=rbst,rmethod='med',truncated=trunc).to(torch.float64) *len(x)    
                avg_log_likelihood = total_log_likelihood / len(test_data)
                log_likelihoods.append(avg_log_likelihood.item())        
                model.train()
        avg_log_likelihoods.append([trunc,log_likelihoods])     
        loss_hists.append([trunc,loss_hist])   
        torch.save(avg_log_likelihoods,f'/home/samiri/SynDG/DOFLOWS/images/NICE_loglikelihoods.pt')
        torch.save(loss_hists,f'/home/samiri/SynDG/DOFLOWS/images/NICE_losshists.pt')

        with torch.no_grad():
            model.eval()
            ds_gn = model.sample(len(my_dataset.data))[0].detach().cpu().numpy()
            ds_gn = pd.DataFrame(ds_gn, columns=my_dataset.data.columns)
            ds_gn.replace([np.inf, -np.inf], np.nan, inplace=True)
            ds_gn.dropna(inplace=True)
            ds_gn.dropna(axis=1, inplace=True)
            ds_gn.dropna(inplace=True)
            dict_dtype = my_dataset.data.dtypes.apply(lambda x: x.name).to_dict()
            ds_gn = ds_gn.astype(dict_dtype)

            #ds_gn.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_{dp}_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}_{nsamp}_{nmodes}_{rndadd}_{useloc}_{usestd}_{initp}.csv')
            #my_dataset.data.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_gen.csv')
            nan_or_inf_df = ds_gn.isna() | np.isinf(ds_gn)

            # Assuming real_data and synthetic_data are your dataframes
            feature_names = my_dataset.data.columns

            # List of categorical features
            categorical_features = []

            fig, axs = plt.subplots(2, 5, figsize=(20, 5))

            for i, ax in enumerate(axs.flatten()):
                if i < len(feature_names):
                    feature_name = feature_names[i]

                    # If the feature is categorical
                    if i in categorical_features:
                        real_counts = my_dataset.data[feature_name].value_counts()
                        synthetic_counts = ds_gn[feature_name].value_counts()
                        all_categories = list(set(real_counts.index) | set(synthetic_counts.index))
                        
                        ax.bar(all_categories, [real_counts.get(category, 0) for category in all_categories], color='blue', alpha=0.5, label='Real')
                        ax.bar(all_categories, [synthetic_counts.get(category, 0) for category in all_categories], color='red', alpha=0.5, label='Synthetic')
                    else:
                        sns.kdeplot(my_dataset.data[feature_name], ax=ax, color='blue', label='Real')
                        sns.kdeplot(ds_gn[feature_name], ax=ax, color='red', label='Synthetic')

                    ax.set_title(feature_name)
                    ax.legend()
            fig.suptitle(f'Fold {it}', fontsize=15)            
            plt.tight_layout()
            plt.show()
            model.train()  
            torch.save(avg_log_likelihoods,f'/home/samiri/SynDG/DOFLOWS/images/NICE_loglikelihoods.pt')
            torch.save(loss_hists,f'/home/samiri/SynDG/DOFLOWS/images/NICE_losshists.pt')
            del ds_gn

        torch.cuda.empty_cache()
        gc.collect()
        import imageio

    # Get the file names of the images
        img_files = [f'/home/samiri/SynDG/DOFLOWS/images/lifesci_{trunc}_{i}_NICE.png' for i in range(ctr)]

        # Read the images into memory
        imgs = [imageio.imread(img_file) for img_file in img_files]

        # Save the images as a GIF
        imageio.mimsave(f'/home/samiri/SynDG/DOFLOWS/images/lifesci_{trunc}_{dctr}_NICE.gif', imgs)
        plt.figure()
        plt.plot(loss_hist)
        plt.savefig(f'/home/samiri/SynDG/DOFLOWS/images/lifesci_{trunc}_{dctr}_NICE_loss.png')
        
del optimizer,scheduler,dataloader,flows    
torch.cuda.empty_cache()
gc.collect()

# %%


my_dataset = CSVDataset('/home/samiri/SynDG/DOFLOWS/lifesci.csv',categorical)

categorical_qlevels = []
vdeq_categoricals = {int(k): int(v) for k, v in zip(categorical, categorical_qlevels)}
# Number of samples

enable_cuda = True

device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

n_sammple = my_dataset.__len__()
# Create the 2-dimensional instance
loss_arr = []
tests_arr = []
_closs = 1e10
_stalecounter = 0

nl = 20
w = 256
ml = 4
lr = 1e-5
fltyp = 'rnvp'

vlay = 0
nsamp = 2048
nmodes = 200
rndadd = 0.5
usestd = True
useloc = True
initp = 2.5
batch_size = 199


dp = 'NF'

    # for w in list(reversed([64,128,192,256,378,512,768,1024,2048,4096,8192])):    
_closs = 1e10
torch.cuda.empty_cache()
gc.collect()
try:
    del model, optimizer, flows, base
except:
    pass
dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
num_layers = nl
flows = []
latent_size = len(my_dataset.__getitem__(0))
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
if fltyp == 'nsp':
    flows = nsp(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
elif fltyp == 'rnvp':
    flows = rnvp(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
elif fltyp == 'residual':
    flows = residual(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
elif fltyp == 'nice':
    flows = nice(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
    

trnbl = True
# base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=nmodes, rand_p=True, noise_scale=rndadd, dim=latent_size,loc=list(my_dataset.data.median()) if useloc else 0.,scale=list(my_dataset.data.std()) if usestd else 1.,p=initp,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl,ds=my_dataset)

#model = nf.NormalizingFlow(base, flows)
loss_hists = np.array([])
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr/10)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
# target_epsilon = 1.0
# if dp == 'DPSGD':
#     sigma=get_noise_multiplier(
#                     target_epsilon = target_epsilon,
#                     target_delta = 1.52e-5,
#                     sample_rate = batch_size/(len(dataloader.dataset)*0.9),
#                     epochs = max_iter,
#                 )

#     privacy_engine = PrivacyEngine(
#                 model,
#                 batch_size=batch_size,
#                 sample_size=len(dataloader.dataset)*0.9,
#                 noise_multiplier=sigma,
#                 epochs=max_iter,
#                 clipping_mode='MixOpt',
#                 origin_params=None,
#             )
#     privacy_engine.attach(optimizer)





max_norm = 0.5
adjust_rate = 0.01
num_samples = nsamp
show_iter = 200
max_iter = 15
bestloss = 1e10
_stalecounter = 0
pbar = tqdm(range(max_iter))
# Prepare the data for KFold
data = list(dataloader.dataset)
kf = KFold(n_splits=10)

# To store average log likelihood for each epoch
avg_log_likelihoods = []
loss_hists = []
dctr = -1
rbst = False
falseflag = False
for lipschitzconstrained in [True,False]:
    for min in [-1]:
        for max in [1]:
            for func in ['tanh','sigmoid']:
                for boundtranslate in [True,False]:
                    for train_index, test_index in kf.split(data):
                        for trunc in ['tanh']:
                            if lipschitzconstrained == False:
                                falseflag = True
                            if falseflag:
                                break
                            flows = []
                            latent_size = len(my_dataset.__getitem__(0))
                            b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
                            flows = []
                            if fltyp == 'nsp':
                                flows = nsp(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
                            elif fltyp == 'rnvp':
                                flows = rnvp(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml,lipschitzconstrained=lipschitzconstrained,min=min,max=max,func=func,boundtranslate=boundtranslate)
                            elif fltyp == 'residual':
                                flows = residual(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
                            elif fltyp == 'nice':
                                flows = nice(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
                            
                            try:
                                del model,base,optimizer
                                del best_params
                                del bestloss
                            except:
                                pass

                            dctr += 1
                            # Reset the model and the best loss for each fold
                            base = nf.distributions.base.DiagGaussian(latent_size)
                            if dp == 'NF':
                                cmin = -2.
                                cmax = 2.   
                                import scipy.stats as stats

                                # Z-scores for 0.05% and 99.95% of the standard normal distribution
                                Z_a = stats.norm.ppf(0.0005)
                                Z_b = stats.norm.ppf(0.9995)

                                # Since the mean is 0, the equations to solve are:
                                # a = Z_a * sigma
                                # b = Z_b * sigma

                                # Solving these for sigma gives the same result, so we can just use one of them:
                                sigma = cmin / Z_a
                                sigma = 1.0

                                print("New sigma: ", sigma)        
                                base = nf.distributions.base_extended.TruncatedNormal(latent_size,0.,sigma, cmin, cmax)
                                #base = nf.distributions.base_extended.TruncatedGaussian(latent_size,0.,sigma, cmin, cmax,trainable=False)
                                if dctr < 1:
                                    # pl = []
                                    # for _i in range(50):
                                    #     s = base.sample(60000000)
                                    #     _pl = base.log_prob(s).exp().cpu().detach()
                                    #     pl.append(_pl)
                                    #     print(_i,':',np.log(torch.cat(pl).max().item()) - np.log(torch.cat(pl).min().item()))
                                    # del s
                                    # pl = torch.cat(pl)
                                    mn = base.log_prob(base.a.view(1,10).to(torch.float64)).exp()
                                    mx = base.log_prob(base.mean.view(1,10).to(torch.float64)).exp()
                                    eps = np.log(mx[0].item()) - np.log(mn[0].item())
                                    # del pl,_pl
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    et,dt = calculate_privacy_loss(max_iter, eps, 1e-4, batch_size, len(train_index))
                                    print(f'final consumed budget in interval {[cmin,cmax]} is {et} with delta {dt}')
                                    truncated = True

                            model = nf.NormalizingFlow(base, flows,categoricals=vdeq_categoricals, vardeq_layers=vlay, vardeq_flow_type='shiftscale')
                            model = model.to(device)





                            loss_hist = np.array([])

                            log_likelihoods = []

                            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr/10)
                            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
                            target_epsilon = 1.0
                            if dp == 'DPSGD':
                                sigma=get_noise_multiplier(
                                                target_epsilon = target_epsilon,
                                                target_delta = 1.52e-5,
                                                sample_rate = batch_size/(len(dataloader.dataset)*0.9),
                                                epochs = max_iter,
                                            )

                                privacy_engine = PrivacyEngine(
                                            model,
                                            batch_size=batch_size,
                                            sample_size=len(dataloader.dataset)*0.9,
                                            noise_multiplier=sigma,
                                            epochs=max_iter,
                                            clipping_mode='MixOpt',
                                            origin_params=None,
                                        )
                                privacy_engine.attach(optimizer)










                            max_norm = 0.5
                            adjust_rate = 0.01

                            best_params = copy.deepcopy(model.state_dict())
                            bestloss = 1e10
                            _stalecounter = 0
                            pbar = tqdm(range(max_iter))

                            
                            bestloss = float('inf')
                            best_params = copy.deepcopy(model.state_dict())

                            # Create dataloaders for this fold
                            train_data = [data[i] for i in train_index]
                            test_data = [data[i] for i in test_index]
                            train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
                            test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
                            ctr = -1
                            for it in pbar:
                                ctr += 1
                                for i, features in enumerate(train_dataloader, 0):
                                    optimizer.zero_grad()
                                    x = features.to(device)
                                    try:
                                        loss = model.forward_kld(x, robust=rbst,rmethod='med',truncated=trunc)    
                                        if ~(torch.isnan(loss) | torch.isinf(loss)):
                                            loss.backward()
                                            optimizer.step()
                                            with torch.no_grad():
                                                if loss.item()<bestloss:
                                                    _s,_ = model.sample(1000)
                                                    if (torch.isnan(_s).sum()==0 & torch.isinf(_s).sum()==0):
                                                    
                                                        bestloss = copy.deepcopy(loss.item())
                                                        best_params = copy.deepcopy(model.state_dict())
                                                        pbar.set_description(f"Processing {it}, best loss={bestloss}")
                                        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
                                        pbar.set_description(f"Processing {it}, best loss={bestloss}")
                                    except Exception as e:
                                        if True:
                                            #print(e)
                                            with torch.no_grad():
                                                model.load_state_dict(best_params)
                                
                                scheduler.step(bestloss)

                                with torch.no_grad():
                                    model.eval()
                                    ds_gn = model.sample(len(my_dataset.data))[0].detach().cpu().numpy()
                                    ds_gn = pd.DataFrame(ds_gn, columns=my_dataset.data.columns)
                                    ds_gn.replace([np.inf, -np.inf], np.nan, inplace=True)
                                    ds_gn.dropna(inplace=True)
                                    ds_gn.dropna(axis=1, inplace=True)
                                    ds_gn.dropna(inplace=True)
                                    dict_dtype = my_dataset.data.dtypes.apply(lambda x: x.name).to_dict()
                                    ds_gn = ds_gn.astype(dict_dtype)

                                    #ds_gn.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_{dp}_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}_{nsamp}_{nmodes}_{rndadd}_{useloc}_{usestd}_{initp}.csv')
                                    my_dataset.data.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_gen.csv')
                                    nan_or_inf_df = ds_gn.isna() | np.isinf(ds_gn)

                                    # Assuming real_data and synthetic_data are your dataframes
                                    feature_names = my_dataset.data.columns

                                    # List of categorical features
                                    categorical_features = []

                                    fig, axs = plt.subplots(2, 5, figsize=(20, 5))

                                    for i, ax in enumerate(axs.flatten()):
                                        if i < len(feature_names):
                                            feature_name = feature_names[i]

                                            # If the feature is categorical
                                            if i in categorical_features:
                                                real_counts = my_dataset.data[feature_name].value_counts()
                                                synthetic_counts = ds_gn[feature_name].value_counts()
                                                all_categories = list(set(real_counts.index) | set(synthetic_counts.index))
                                                
                                                ax.bar(all_categories, [real_counts.get(category, 0) for category in all_categories], color='blue', alpha=0.5, label='Real')
                                                ax.bar(all_categories, [synthetic_counts.get(category, 0) for category in all_categories], color='red', alpha=0.5, label='Synthetic')
                                            else:
                                                sns.kdeplot(my_dataset.data[feature_name], ax=ax, color='blue', label='Real')
                                                sns.kdeplot(ds_gn[feature_name], ax=ax, color='red', label='Synthetic')

                                            ax.set_title(feature_name)
                                            ax.legend()
                                    fig.suptitle(f'Fold {it}', fontsize=15)            
                                    plt.tight_layout()
                                    plt.savefig(f'/home/samiri/SynDG/DOFLOWS/images/lifesci_{trunc}_{ctr}_{lipschitzconstrained}_{min}_{max}_{func}_{boundtranslate}.png')
                                    plt.close()
                                    model.train()  
                                    del ds_gn





                                
                                pbar.set_description(f"Processing {it}, best loss={bestloss}")
                                with torch.no_grad():
                                    model.eval()
                                    total_log_likelihood = 0
                                    for i, features in enumerate(test_dataloader, 0):
                                        x = features.to(device)
                                        total_log_likelihood += -model.forward_kld(x, robust=rbst,rmethod='med',truncated=trunc).to(torch.float64) *len(x)    
                                    avg_log_likelihood = total_log_likelihood / len(test_data)
                                    log_likelihoods.append(avg_log_likelihood.item())        
                                    model.train()
                            avg_log_likelihoods.append([trunc,lipschitzconstrained,min,max,func,boundtranslate,log_likelihoods])     
                            loss_hists.append([trunc,lipschitzconstrained,min,max,func,boundtranslate,loss_hist])   
                            torch.save(avg_log_likelihoods,f'/home/samiri/SynDG/DOFLOWS/images/loglikelihoods.pt')
                            torch.save(loss_hists,f'/home/samiri/SynDG/DOFLOWS/images/losshists.pt')

                            with torch.no_grad():
                                model.eval()
                                ds_gn = model.sample(len(my_dataset.data))[0].detach().cpu().numpy()
                                ds_gn = pd.DataFrame(ds_gn, columns=my_dataset.data.columns)
                                ds_gn.replace([np.inf, -np.inf], np.nan, inplace=True)
                                ds_gn.dropna(inplace=True)
                                ds_gn.dropna(axis=1, inplace=True)
                                ds_gn.dropna(inplace=True)
                                dict_dtype = my_dataset.data.dtypes.apply(lambda x: x.name).to_dict()
                                ds_gn = ds_gn.astype(dict_dtype)

                                #ds_gn.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_{dp}_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}_{nsamp}_{nmodes}_{rndadd}_{useloc}_{usestd}_{initp}.csv')
                                #my_dataset.data.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_gen.csv')
                                nan_or_inf_df = ds_gn.isna() | np.isinf(ds_gn)

                                # Assuming real_data and synthetic_data are your dataframes
                                feature_names = my_dataset.data.columns

                                # List of categorical features
                                categorical_features = []

                                fig, axs = plt.subplots(2, 5, figsize=(20, 5))

                                for i, ax in enumerate(axs.flatten()):
                                    if i < len(feature_names):
                                        feature_name = feature_names[i]

                                        # If the feature is categorical
                                        if i in categorical_features:
                                            real_counts = my_dataset.data[feature_name].value_counts()
                                            synthetic_counts = ds_gn[feature_name].value_counts()
                                            all_categories = list(set(real_counts.index) | set(synthetic_counts.index))
                                            
                                            ax.bar(all_categories, [real_counts.get(category, 0) for category in all_categories], color='blue', alpha=0.5, label='Real')
                                            ax.bar(all_categories, [synthetic_counts.get(category, 0) for category in all_categories], color='red', alpha=0.5, label='Synthetic')
                                        else:
                                            sns.kdeplot(my_dataset.data[feature_name], ax=ax, color='blue', label='Real')
                                            sns.kdeplot(ds_gn[feature_name], ax=ax, color='red', label='Synthetic')

                                        ax.set_title(feature_name)
                                        ax.legend()
                                fig.suptitle(f'Fold {it}', fontsize=15)            
                                plt.tight_layout()
                                plt.show()
                                model.train()  
                                torch.save(avg_log_likelihoods,f'/home/samiri/SynDG/DOFLOWS/images/loglikelihoods.pt')
                                torch.save(loss_hists,f'/home/samiri/SynDG/DOFLOWS/images/losshists.pt')
                                del ds_gn

                            torch.cuda.empty_cache()
                            gc.collect()
                            import imageio

                        # Get the file names of the images
                            img_files = [f'/home/samiri/SynDG/DOFLOWS/images/lifesci_{trunc}_{i}_{lipschitzconstrained}_{min}_{max}_{func}_{boundtranslate}.png' for i in range(ctr)]

                            # Read the images into memory
                            imgs = [imageio.imread(img_file) for img_file in img_files]

                            # Save the images as a GIF
                            imageio.mimsave(f'/home/samiri/SynDG/DOFLOWS/images/lifesci_{trunc}_{dctr}_{lipschitzconstrained}_{min}_{max}_{func}_{boundtranslate}.gif', imgs)
                            plt.figure()
                            plt.plot(loss_hist)
                            plt.savefig(f'/home/samiri/SynDG/DOFLOWS/images/lifesci_{trunc}_{dctr}_{lipschitzconstrained}_{min}_{max}_{func}_{boundtranslate}_loss.png')
        
del optimizer,scheduler,dataloader,flows    
torch.cuda.empty_cache()
gc.collect()
import sys
sys.exit()
loss_arr.append([nl,w,bestloss])
torch.cuda.empty_cache()
gc.collect()

# torch.save(model,f'/home/samiri/SynDG/DOFLOWS/lifesci_{dp}_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}_{nsamp}_{nmodes}_{rndadd}_{useloc}_{usestd}_{initp}.pt')
# with torch.no_grad():
#     model.eval()

#     ds_gn = model.sample(len(my_dataset.data))[0].detach().cpu().numpy()


# #    del model
#     ds_gn = pd.DataFrame(ds_gn, columns=my_dataset.data.columns)
#     ds_gn.replace([np.inf, -np.inf], np.nan, inplace=True)
#     ds_gn.dropna(inplace=True)
#     ds_gn.dropna(axis=1, inplace=True)
#     ds_gn.dropna(inplace=True)
#     dict_dtype = my_dataset.data.dtypes.apply(lambda x: x.name).to_dict()
#     ds_gn = ds_gn.astype(dict_dtype)

#     ds_gn.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_{dp}_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}_{nsamp}_{nmodes}_{rndadd}_{useloc}_{usestd}_{initp}.csv')
#     my_dataset.data.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_gen.csv')
#     nan_or_inf_df = ds_gn.isna() | np.isinf(ds_gn)

#     # Assuming real_data and synthetic_data are your dataframes
#     feature_names = my_dataset.data.columns

#     # List of categorical features
#     categorical_features = []

#     fig, axs = plt.subplots(2, 5, figsize=(10, 20))

#     for i, ax in enumerate(axs.flatten()):
#         if i < len(feature_names):
#             feature_name = feature_names[i]

#             # If the feature is categorical
#             if i in categorical_features:
#                 real_counts = my_dataset.data[feature_name].value_counts()
#                 synthetic_counts = ds_gn[feature_name].value_counts()
#                 all_categories = list(set(real_counts.index) | set(synthetic_counts.index))
                
#                 ax.bar(all_categories, [real_counts.get(category, 0) for category in all_categories], color='blue', alpha=0.5, label='Real')
#                 ax.bar(all_categories, [synthetic_counts.get(category, 0) for category in all_categories], color='red', alpha=0.5, label='Synthetic')
#             else:
#                 sns.kdeplot(my_dataset.data[feature_name], ax=ax, color='blue', label='Real')
#                 sns.kdeplot(ds_gn[feature_name], ax=ax, color='red', label='Synthetic')

#             ax.set_title(feature_name)
#             ax.legend()

#     plt.tight_layout()
#     plt.show()
#     model.train()

# plt.savefig(f'/home/samiri/SynDG/DOFLOWS/lifesci_{dp}_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}_{nsamp}_{nmodes}_{rndadd}_{useloc}_{usestd}_{initp}.png')
# del ds_gn
# torch.cuda.empty_cache()
# gc.collect()
# torch.cuda.empty_cache()
# gc.collect()
# torch.cuda.empty_cache()
# gc.collect()




# %%
import torch
import pandas as pd
import numpy as np
avg_log_likelihoods = torch.load(f'/home/samiri/SynDG/DOFLOWS/images/loglikelihoods.pt')
loss_hists = torch.load(f'/home/samiri/SynDG/DOFLOWS/images/losshists.pt')
# avg_log_likelihoods = [i[1] for i in avg_log_likelihoods if i[0] == 'tanh']
# loss_hists = [i[1] for i in loss_hists if i[0] == 'tanh']
avg_log_likelihoods = [i[6] for i in avg_log_likelihoods if i[0] == 'tanh']
loss_hists = [i[6] for i in loss_hists if i[0] == 'tanh']

avg_log_likelihoods = pd.DataFrame(avg_log_likelihoods)
loss_hists = pd.DataFrame(loss_hists)

# %%
#avg_log_likelihoods = torch.load(f'/home/samiri/SynDG/DOFLOWS/images/loglikelihoods.pt')
avg_log_likelihoods

# %%

df = pd.DataFrame(avg_log_likelihoods).T

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# assuming df is your DataFrame where rows represent measurements and columns represent experiments

# Calculate the mean and standard error for each measurement across all experiments
mean = df.mean(axis=1)
mean.index = np.arange(0,4.01,4/14)
std_error = df.sem(axis=1)

# Create a figure
plt.figure(figsize=(10, 6))

# Plot the mean line
plt.plot(mean.index, mean.values, color='blue', label='Mean')

# Add the shaded region representing the standard error
plt.fill_between(mean.index, mean.values - std_error, mean.values + std_error, color='blue', alpha=0.2)

# Set the title and labels
plt.title('Mean and Standard Error of Test Set Log Likelihood, RNVP')
plt.xlabel('Measurement')
plt.ylabel('Values')
plt.legend()

# Show the plot
plt.show()

# %%
import torch
import pandas as pd
import numpy as np
avg_log_likelihoods = torch.load(f'/home/samiri/SynDG/DOFLOWS/images/NICE_loglikelihoods.pt')
loss_hists = torch.load(f'/home/samiri/SynDG/DOFLOWS/images/losshists.pt')
avg_log_likelihoods = [i[1] for i in avg_log_likelihoods if i[0] == 'tanh']
loss_hists = [i[1] for i in loss_hists if i[0] == 'tanh']
# avg_log_likelihoods = [i[6] for i in avg_log_likelihoods if i[0] == 'tanh']
# loss_hists = [i[6] for i in loss_hists if i[0] == 'tanh']

avg_log_likelihoods = pd.DataFrame(avg_log_likelihoods)
loss_hists = pd.DataFrame(loss_hists)

# %%
#avg_log_likelihoods = torch.load(f'/home/samiri/SynDG/DOFLOWS/images/loglikelihoods.pt')
avg_log_likelihoods

# %%

df = pd.DataFrame(avg_log_likelihoods).T

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# assuming df is your DataFrame where rows represent measurements and columns represent experiments

# Calculate the mean and standard error for each measurement across all experiments
mean = df.mean(axis=1)

std_error = df.sem(axis=1)

# Create a figure
plt.figure(figsize=(10, 6))

# Plot the mean line
plt.plot(mean.index, mean.values, color='blue', label='Mean')

# Add the shaded region representing the standard error
plt.fill_between(mean.index, mean.values - std_error, mean.values + std_error, color='blue', alpha=0.2)

# Set the title and labels
plt.title('Mean and Standard Error of Test Set Log Likelihood, NICE')
plt.xlabel('Measurement')
plt.ylabel('Values')
plt.legend()
print(plt.xticks()[0])

# Show the plot
plt.show()

# %%

mean
# %%

my_dataset = CSVDataset('/home/samiri/SynDG/DOFLOWS/lifesci.csv',categorical)

categorical_qlevels = []
vdeq_categoricals = {int(k): int(v) for k, v in zip(categorical, categorical_qlevels)}
# Number of samples

enable_cuda = True

device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

n_sammple = my_dataset.__len__()
# Create the 2-dimensional instance
loss_arr = []
tests_arr = []
_closs = 1e10
_stalecounter = 0

nl = 40
w = 256
ml = 4
lr = 5e-4
fltyp = 'nice'
rbst = False
vlay = 0
nsamp = 2048
nmodes = 200
rndadd = 0.5
usestd = True
useloc = True
initp = 2.5
batch_size = 2**10


dp = 'NF'

    # for w in list(reversed([64,128,192,256,378,512,768,1024,2048,4096,8192])):    
_closs = 1e10
torch.cuda.empty_cache()
gc.collect()
try:
    del model, optimizer, flows, base
except:
    pass
dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True)
num_layers = nl
flows = []
latent_size = len(my_dataset.__getitem__(0))
b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
flows = []
if fltyp == 'nsp':
    flows = nsp(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
elif fltyp == 'rnvp':
    flows = rnvp(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
elif fltyp == 'residual':
    flows = residual(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
elif fltyp == 'nice':
    flows = nice(K=nl,dim=latent_size, hidden_units=w, hidden_layers=ml)
    

trnbl = True
# base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=nmodes, rand_p=True, noise_scale=rndadd, dim=latent_size,loc=list(my_dataset.data.median()) if useloc else 0.,scale=list(my_dataset.data.std()) if usestd else 1.,p=initp,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl,ds=my_dataset)

#model = nf.NormalizingFlow(base, flows)
loss_hists = np.array([])
# optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr/10)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
# target_epsilon = 1.0
# if dp == 'DPSGD':
#     sigma=get_noise_multiplier(
#                     target_epsilon = target_epsilon,
#                     target_delta = 1.52e-5,
#                     sample_rate = batch_size/(len(dataloader.dataset)*0.9),
#                     epochs = max_iter,
#                 )

#     privacy_engine = PrivacyEngine(
#                 model,
#                 batch_size=batch_size,
#                 sample_size=len(dataloader.dataset)*0.9,
#                 noise_multiplier=sigma,
#                 epochs=max_iter,
#                 clipping_mode='MixOpt',
#                 origin_params=None,
#             )
#     privacy_engine.attach(optimizer)










max_norm = 0.5
adjust_rate = 0.01
num_samples = nsamp
show_iter = 200
max_iter = 300
bestloss = 1e10
_stalecounter = 0
pbar = tqdm(range(max_iter))
# Prepare the data for KFold
data = list(dataloader.dataset)
kf = KFold(n_splits=10)

# To store average log likelihood for each epoch
avg_log_likelihoods = []
loss_hists = []

for train_index, test_index in kf.split(data):
    # Reset the model and the best loss for each fold
    base = nf.distributions.base.DiagGaussian(latent_size)
    if dp == 'NF':
        cmin = -2.01
        cmax = 2.01    
        
        base = nf.distributions.base_extended.TruncatedNormal(latent_size,0.,1., cmin, cmax)
        s = base.sample(10000)
        pl = base.log_prob(s).exp()
        print(f'final consumed budget in interval {[cmin,cmax]} is ', np.log(pl.max().item()/pl.min().item())*np.sqrt(max_iter))

    model = nf.NormalizingFlow(base, flows,categoricals=vdeq_categoricals, vardeq_layers=vlay, vardeq_flow_type='shiftscale')
    model = model.to(device)





    loss_hist = np.array([])

    log_likelihoods = []

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr/10)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    target_epsilon = 1.0
    if dp == 'DPSGD':
        sigma=get_noise_multiplier(
                        target_epsilon = target_epsilon,
                        target_delta = 1.52e-5,
                        sample_rate = batch_size/(len(dataloader.dataset)*0.9),
                        epochs = max_iter,
                    )

        privacy_engine = PrivacyEngine(
                    model,
                    batch_size=batch_size,
                    sample_size=len(dataloader.dataset)*0.9,
                    noise_multiplier=sigma,
                    epochs=max_iter,
                    clipping_mode='MixOpt',
                    origin_params=None,
                )
        privacy_engine.attach(optimizer)










    max_norm = 0.5
    adjust_rate = 0.01

    best_params = copy.deepcopy(model.state_dict())
    bestloss = 1e10
    _stalecounter = 0
    pbar = tqdm(range(max_iter))

    
    bestloss = float('inf')
    best_params = copy.deepcopy(model.state_dict())

    # Create dataloaders for this fold
    train_data = [data[i] for i in train_index]
    test_data = [data[i] for i in test_index]
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    for it in pbar:
        for i, features in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            x = features.to(device)
            try:
                loss = model.forward_kld(x, robust=rbst,rmethod='med')    
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()
                    with torch.no_grad():
                        if loss.item()<bestloss:
                            _s,_ = model.sample(1000)
                            if (torch.isnan(_s).sum()==0 & torch.isinf(_s).sum()==0):
                            
                                bestloss = copy.deepcopy(loss.item())
                                best_params = copy.deepcopy(model.state_dict())
                                pbar.set_description(f"Processing {it}, best loss={bestloss}")
                loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
                pbar.set_description(f"Processing {it}, best loss={bestloss}")
            except Exception as e:
                if True:
                    #print(e)
                    with torch.no_grad():
                        model.load_state_dict(best_params)
        

        # with torch.no_grad():
        #     model.eval()
        #     ds_gn = model.sample(len(my_dataset.data))[0].detach().cpu().numpy()
        #     ds_gn = pd.DataFrame(ds_gn, columns=my_dataset.data.columns)
        #     ds_gn.replace([np.inf, -np.inf], np.nan, inplace=True)
        #     ds_gn.dropna(inplace=True)
        #     ds_gn.dropna(axis=1, inplace=True)
        #     ds_gn.dropna(inplace=True)
        #     dict_dtype = my_dataset.data.dtypes.apply(lambda x: x.name).to_dict()
        #     ds_gn = ds_gn.astype(dict_dtype)

        #     ds_gn.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_{dp}_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}_{nsamp}_{nmodes}_{rndadd}_{useloc}_{usestd}_{initp}.csv')
        #     my_dataset.data.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_gen.csv')
        #     nan_or_inf_df = ds_gn.isna() | np.isinf(ds_gn)

        #     # Assuming real_data and synthetic_data are your dataframes
        #     feature_names = my_dataset.data.columns

        #     # List of categorical features
        #     categorical_features = []

        #     fig, axs = plt.subplots(2, 5, figsize=(20, 5))

        #     for i, ax in enumerate(axs.flatten()):
        #         if i < len(feature_names):
        #             feature_name = feature_names[i]

        #             # If the feature is categorical
        #             if i in categorical_features:
        #                 real_counts = my_dataset.data[feature_name].value_counts()
        #                 synthetic_counts = ds_gn[feature_name].value_counts()
        #                 all_categories = list(set(real_counts.index) | set(synthetic_counts.index))
                        
        #                 ax.bar(all_categories, [real_counts.get(category, 0) for category in all_categories], color='blue', alpha=0.5, label='Real')
        #                 ax.bar(all_categories, [synthetic_counts.get(category, 0) for category in all_categories], color='red', alpha=0.5, label='Synthetic')
        #             else:
        #                 sns.kdeplot(my_dataset.data[feature_name], ax=ax, color='blue', label='Real')
        #                 sns.kdeplot(ds_gn[feature_name], ax=ax, color='red', label='Synthetic')

        #             ax.set_title(feature_name)
        #             ax.legend()
        #     fig.suptitle(f'Fold {it}', fontsize=15)            
        #     plt.tight_layout()
        #     plt.show()
        #     model.train()  
        #     del ds_gn
        torch.cuda.empty_cache()
        gc.collect()

        scheduler.step(bestloss)
        pbar.set_description(f"Processing {it}, best loss={bestloss}")
    #     with torch.no_grad():
    #         model.eval()
    #         total_log_likelihood = 0
    #         for i, features in enumerate(test_dataloader, 0):
    #             x = features.to(device)
    #             total_log_likelihood += model.forward_kld(x, robust=rbst,rmethod='med')*len(x)    
    #         avg_log_likelihood = total_log_likelihood / len(test_data)
    #         log_likelihoods.append(avg_log_likelihood.item())        
    #         model.train()
    # avg_log_likelihoods.append(log_likelihoods)     
    loss_hists.append(loss_hist)   
        
del optimizer,scheduler,dataloader,flows    
torch.cuda.empty_cache()
gc.collect()

loss_arr.append([nl,w,bestloss])
torch.cuda.empty_cache()
gc.collect()

# torch.save(model,f'/home/samiri/SynDG/DOFLOWS/lifesci_{dp}_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}_{nsamp}_{nmodes}_{rndadd}_{useloc}_{usestd}_{initp}.pt')
# with torch.no_grad():
#     model.eval()

#     ds_gn = model.sample(len(my_dataset.data))[0].detach().cpu().numpy()


# #    del model
#     ds_gn = pd.DataFrame(ds_gn, columns=my_dataset.data.columns)
#     ds_gn.replace([np.inf, -np.inf], np.nan, inplace=True)
#     ds_gn.dropna(inplace=True)
#     ds_gn.dropna(axis=1, inplace=True)
#     ds_gn.dropna(inplace=True)
#     dict_dtype = my_dataset.data.dtypes.apply(lambda x: x.name).to_dict()
#     ds_gn = ds_gn.astype(dict_dtype)

#     ds_gn.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_{dp}_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}_{nsamp}_{nmodes}_{rndadd}_{useloc}_{usestd}_{initp}.csv')
#     my_dataset.data.to_csv(f'/home/samiri/SynDG/DOFLOWS/lifesci_gen.csv')
#     nan_or_inf_df = ds_gn.isna() | np.isinf(ds_gn)

#     # Assuming real_data and synthetic_data are your dataframes
#     feature_names = my_dataset.data.columns

#     # List of categorical features
#     categorical_features = []

#     fig, axs = plt.subplots(2, 5, figsize=(10, 20))

#     for i, ax in enumerate(axs.flatten()):
#         if i < len(feature_names):
#             feature_name = feature_names[i]

#             # If the feature is categorical
#             if i in categorical_features:
#                 real_counts = my_dataset.data[feature_name].value_counts()
#                 synthetic_counts = ds_gn[feature_name].value_counts()
#                 all_categories = list(set(real_counts.index) | set(synthetic_counts.index))
                
#                 ax.bar(all_categories, [real_counts.get(category, 0) for category in all_categories], color='blue', alpha=0.5, label='Real')
#                 ax.bar(all_categories, [synthetic_counts.get(category, 0) for category in all_categories], color='red', alpha=0.5, label='Synthetic')
#             else:
#                 sns.kdeplot(my_dataset.data[feature_name], ax=ax, color='blue', label='Real')
#                 sns.kdeplot(ds_gn[feature_name], ax=ax, color='red', label='Synthetic')

#             ax.set_title(feature_name)
#             ax.legend()

#     plt.tight_layout()
#     plt.show()
#     model.train()

# plt.savefig(f'/home/samiri/SynDG/DOFLOWS/lifesci_{dp}_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}_{nsamp}_{nmodes}_{rndadd}_{useloc}_{usestd}_{initp}.png')
# del ds_gn
# torch.cuda.empty_cache()
# gc.collect()
# torch.cuda.empty_cache()
# gc.collect()
# torch.cuda.empty_cache()
# gc.collect()




# %%
plt.plot(loss_hist)


