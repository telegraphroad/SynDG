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
    np.concatenate([numerical_features, cat1[:, None], cat2[:, None]], axis=1),
    columns=['num_feature1', 'num_feature2', 'num_feature3', 'cat_feature1', 'cat_feature2']
)

# Convert categorical features to integer type
data[['cat_feature1', 'cat_feature2']] = data[['cat_feature1', 'cat_feature2']].astype(int)

print(data.head())

    
# Create the Dataset and DataLoader
my_dataset = MyDataset(data)
dataloader = DataLoader(my_dataset, batch_size=128, shuffle=True)

enable_cuda = True

device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

loc = torch.zeros((2, 2))  # 2x2 tensor filled with 0s
scale = torch.ones((2, 2))  # 2x2 tensor filled with 1s
p = 2.0  # Shape parameter for Gaussian

# Create the 2-dimensional instance
base = nf.distributions.base.DiagGaussian(5)
# trnbl = True
# base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=25, rand_p=True, noise_scale=0.2, dim=2,loc=0,scale=1.,p=2.,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl)

num_layers = 20
flows = []
latent_size = 5
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

categoricals = {}
categoricals[3] = 7
categoricals[4] = 4
model = nf.NormalizingFlow(base, flows,categoricals=categoricals, vardeq_layers=4, vardeq_flow_type='shiftscale')
model = model.to(device)



max_iter = 50
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
                with torch.no_grad():
                    if loss.item()<bestloss:
                        bestloss = copy.deepcopy(loss.item())
                        best_params = copy.deepcopy(model.state_dict())
            loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
            print(f'Epoch {it+1}, Iter {i+1}, Loss: {loss.item():.6f}')
        except Exception as e:
            if True:
                print(e)
                with torch.no_grad():
                    model.load_state_dict(best_params)

