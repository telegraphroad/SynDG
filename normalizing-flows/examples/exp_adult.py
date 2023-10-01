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

import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, file_path, categorical_column_names, transform=None):
        self.data = pd.read_csv(file_path).drop(['Unnamed: 0'],axis=1)
        
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
categorical = ['1','3','4','5','6','7','8','9','13','14']
my_dataset = CSVDataset('./adult.csv',categorical)

categorical_qlevels = [9,16,16,7,15,6,5,2,42,2]
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
try:
    _nl = int(sys.argv[1])
    _w = int(sys.argv[2])
    _ml = int(sys.argv[3])
    lr = float(sys.argv[4])
    fltyp = str(sys.argv[5])
except:
    _nl = 4
    _w = 128
    _ml = 4
    lr = 1e-5
    fltyp = 'rnvp'
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
                dataloader = DataLoader(my_dataset, batch_size=2**12, shuffle=True)
                print(nl,ml,w)
                num_layers = nl
                flows = []
                latent_size = len(my_dataset.__getitem__(0))
                b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
                flows = []
                # for i in range(num_layers):
                #     lay = [latent_size] + [w]*ml + [latent_size]
                #     s = nf.nets.MLP(lay, init_zeros=True)

                    
                #     t = nf.nets.MLP(lay, init_zeros=True)
                #     if i % 2 == 0:
                #         flows += [nf.flows.MaskedAffineFlow(b, t, s)]
                #     else:
                #         flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
                #     flows += [nf.flows.ActNorm(latent_size)]
                if fltyp == 'nsp':
                    flows = nsp(K=_nl,dim=latent_size, hidden_units=_w, hidden_layers=_ml)
                elif fltyp == 'rnvp':
                    flows = rnvp(K=_nl,dim=latent_size, hidden_units=_w, hidden_layers=_ml)
                elif fltyp == 'residual':
                    flows = residual(K=_nl,dim=latent_size, hidden_units=_w, hidden_layers=_ml)
                base = nf.distributions.base.DiagGaussian(latent_size)
                trnbl = True
                base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=200, rand_p=True, noise_scale=0.1, dim=latent_size,loc=0,scale=1.,p=2.,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl,ds=my_dataset)

                #model = nf.NormalizingFlow(base, flows)
                model = nf.NormalizingFlow(base, flows,categoricals=vdeq_categoricals, vardeq_layers=4, vardeq_flow_type='shiftscale')
                model = model.to(device)



                max_iter = 100
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
                        #x = torch.stack(features).to(device)
                        x = features.to(device)
                        try:
                            loss = model.forward_kld(x, robust=False)    
                            # l2_lambda = 0.001  # The strength of the regularization
                            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                            # loss = loss + l2_lambda * l2_norm
                            if ~(torch.isnan(loss) | torch.isinf(loss)):
                                loss.backward()
                                optimizer.step()
                                with torch.no_grad():
                                    if loss.item()<bestloss:
                                        _s,_ = model.sample(1000)
                                        if (torch.isnan(_s).sum()==0 & torch.isinf(_s).sum()==0):
                                        
                                            bestloss = copy.deepcopy(loss.item())
                                            best_params = copy.deepcopy(model.state_dict())
                                print(f'Epoch {it+1}, Iter {i+1}, Loss: {loss.item():.6f}, Best Loss: {bestloss.item():.6f}')
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
                    if _stalecounter > 250:
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

                
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

                #del model, optimizer, flows, base
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()

                model.eval()
                ds_gn = model.sample(len(my_dataset.data))[0].detach().cpu().numpy()
                
                torch.save(model,f'./model_{nl}_{w}_{ml}_{lr}_{fltyp}.pt')
                del model
                ds_gn = pd.DataFrame(ds_gn, columns=my_dataset.data.columns)
                ds_gn.to_csv(f'./gen_{nl}_{w}_{ml}_{lr}_{fltyp}.csv')
                
                
                # List of numerical and categorical feature names
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

