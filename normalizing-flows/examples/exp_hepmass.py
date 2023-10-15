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


#b = torch.Tensor([1 if i % 2 == 0 else 0 for i in [0]])

#X = X.drop(['Class'],1)
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# Original dataset
from normflows.utils import data_utils

# X = pd.read_csv('power.csv').drop(['Unnamed: 0'],axis=1)


# xcol = X.columns
# # for ii in range(len(categorical)):
# #     X[X.columns[categorical[ii]]] = X[X.columns[categorical[ii]]] * lcm / categorical_qlevels[ii]
# X=X.values
# dataset = TensorDataset(torch.tensor(X, dtype=torch.float32))
# num_samples = 2**9
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=num_samples,num_workers=4)
# train_iter = iter(train_loader)
latent_size = 8
categorical = []
categorical_qlevels = []
catlevels = []
lcm = 0
vlayers = []
#b = torch.Tensor([1 if i % 2 == 0 else 0 for i in [0]])

#X = X.drop(['Class'],1)
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

# Original dataset

data_name = 'GAS'

data_mapping = {'BSDS300': data_utils.BSDS300,
                'GAS': data_utils.GAS,
                'MINIBOONE': data_utils.MINIBOONE,
                'POWER': data_utils.POWER,
                'HEPMASS': data_utils.HEPMASS}


categorical = []
my_dataset = CSVDataset('./hepmass.csv',categorical)

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
try:
    _nl = int(sys.argv[1])
    _w = int(sys.argv[2])
    _ml = int(sys.argv[3])
    lr = float(sys.argv[4])
    fltyp = str(sys.argv[5])
    rbst = bool(sys.argv[6])
    vlay = int(sys.argv[7])
except:
    _nl = 4
    _w = 128
    _ml = 4
    lr = 1e-1
    fltyp = 'rnvp'
    rbst = False
    vlay = 4
    print('Manual params!!!!')
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
                dataloader = DataLoader(my_dataset, batch_size=2**14, shuffle=True)
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
                base = nf.distributions.base_extended.GeneralizedGaussianMixture(n_modes=200, rand_p=True, noise_scale=0.5, dim=latent_size,loc=list(my_dataset.data.median()),scale=list(my_dataset.data.std()),p=2.5,device=device,trainable_loc=trnbl, trainable_scale=trnbl,trainable_p=trnbl,trainable_weights=trnbl,ds=my_dataset)

                #model = nf.NormalizingFlow(base, flows)
                model = nf.NormalizingFlow(base, flows,categoricals=vdeq_categoricals, vardeq_layers=vlay, vardeq_flow_type='shiftscale')
                model = model.to(device)



                max_iter = 100
                num_samples = 2 ** 12
                show_iter = 2500


                loss_hist = np.array([])

                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=lr/10)
                scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

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
                            loss = model.forward_kld(x, robust=rbst,rmethod='med')    
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
                    
                    scheduler.step(bestloss)
                    print(f"Epoch {it+1}, Iter {i+1}, Loss: {bestloss:.6f}, Learning Rate - {optimizer.param_groups[0]['lr']:.6f}")
                    

                    if _closs > bestloss:
                        _closs = bestloss
                        _stalecounter = 0
                    else:
                        if bestloss > 1e9:
                            _stalecounter += 5
                        else:
                            _stalecounter += 1
                    if _stalecounter > 101:
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
                
                torch.save(model,f'./miniboone_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}.pt')
                del model
                ds_gn = pd.DataFrame(ds_gn, columns=my_dataset.data.columns)
                ds_gn.replace([np.inf, -np.inf], np.nan, inplace=True)
                ds_gn.dropna(inplace=True)
                ds_gn.dropna(axis=1, inplace=True)
                ds_gn.dropna(inplace=True)
                dict_dtype = my_dataset.data.dtypes.apply(lambda x: x.name).to_dict()
                ds_gn = ds_gn.astype(dict_dtype)

                ds_gn.to_csv(f'./miniboone_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}.csv')
                my_dataset.data.to_csv(f'./miniboone_gen.csv')
                nan_or_inf_df = ds_gn.isna() | np.isinf(ds_gn)

                ################################################################################################## STATS
                # # Assuming my_dataset and ds_gn are your DataFrames
                # categorical = [1,3,4,5,6,7,8,9,13,14]
                # numerical = [col for col in range(len(my_dataset.data.columns)) if col not in categorical]

                # # Initialize an empty list to store the results
                # results = []

                # # Perform the tests for each categorical feature
                # for feature in categorical:
                #     # Create a contingency table
                #     crosstab = pd.crosstab(my_dataset.data.iloc[:, feature], ds_gn.iloc[:, feature])
                    
                #     # Perform Chi-square test
                #     try:
                #         chi2, p, dof, expected = chi2_contingency(crosstab)
                #         chi2_interpretation = 'Dependent' if p < 0.05 else 'Independent' #note:interpret "dependent" as "non-similar" and "independent" as "similar"
                #     except Exception as e:
                #         print(f"Chi-square test failed for feature {feature}: {str(e)}")

                #     # Append the results to the list
                #     results.append([feature, "Chi-square", chi2, p, chi2_interpretation])

                # # Perform the tests for each numerical feature
                # for feature in numerical:
                #     # Calculate the KS test
                #     try:
                #         ks_stat, p_ks = ks_2samp(my_dataset.data.iloc[:, feature].dropna(), ds_gn.iloc[:, feature].dropna())
                #         interpretation_ks = 'Similar' if p_ks > 0.05 else 'Not similar'
                #     except Exception as e:
                #         print(f"KS test failed for feature {feature}: {str(e)}")

                #     # Append the results to the list
                #     results.append([feature, "KS", ks_stat, p_ks, interpretation_ks])
                    
                #     # Calculate the Mann-Whitney U test
                #     try:
                #         u_stat, p_u = mannwhitneyu(my_dataset.data.iloc[:, feature].dropna(), ds_gn.iloc[:, feature].dropna(), alternative='two-sided')
                #         interpretation_u = 'Similar' if p_u > 0.05 else 'Not similar'
                #     except Exception as e:
                #         print(f"Mann-Whitney U test failed for feature {feature}: {str(e)}")

                #     # Append the results to the list
                #     results.append([feature, "Mann-Whitney", u_stat, p_u, interpretation_u])

                # # Convert the results into a DataFrame
                # results_df = pd.DataFrame(results, columns=['Feature', 'Test', 'Statistic', 'P-value', 'Interpretation'])

                # # Print the results
                # print(results_df)  
                # results_df.to_csv(f'./stattestresults_{nl}_{w}_{ml}_{lr}_{fltyp}.csv', index=False)
                
                ################################################################################################## DETECTION
                # # Assuming my_dataset and ds_gn are your DataFrames
                # # Concatenate your DataFrames
                # data = pd.concat([my_dataset.data, ds_gn])

                # # Compute Gower's distance
                # gower_dist = gower.gower_matrix(data)

                # # Create a list of labels: '0' for my_dataset and '1' for ds_gn
                # labels = ['0'] * len(my_dataset.data) + ['1'] * len(ds_gn)

                # # Perform PERMANOVA
                # dm = DistanceMatrix(gower_dist)
                # results = permanova(dm, grouping=labels, permutations=999)
                # #save results using pandas:
                # results_df = pd.DataFrame([results], columns=['PERMANOVA'])
                # results_df.to_csv(f'permanova_{nl}_{w}_{ml}_{lr}_{fltyp}.csv', index=False)
                # print(results)                
                # # List of numerical and categorical feature names

                # # Assuming real_data and synthetic_data are your dataframes
                # ds_gn['is_synthetic'] = 0
                # my_dataset.data['is_synthetic'] = 1

                # combined_data = pd.concat([ds_gn, my_dataset.data])
                # combined_data.replace([np.inf, -np.inf], np.nan, inplace=True)
                # combined_data.dropna(inplace=True)
                # combined_data.dropna(axis=1, inplace=True)
                
                # X = combined_data.drop('is_synthetic', axis=1)
                # y = combined_data['is_synthetic']

                # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # model = GradientBoostingClassifier()

                # model.fit(X_train, y_train)

                # y_pred = model.predict_proba(X_test)[:,1]

                # score = roc_auc_score(y_test, y_pred)
                # #save score using pandas:
                # score_df = pd.DataFrame([score], columns=['AUC-ROC'])
                # score_df.to_csv(f'detectionscore_{nl}_{w}_{ml}_{lr}_{fltyp}.csv', index=False)
                

                # print(f"AUC-ROC score: {score}")                              
                # import matplotlib.pyplot as plt
                # import seaborn as sns

                # Assuming real_data and synthetic_data are your dataframes
                feature_names = my_dataset.data.columns

                # List of categorical features
                categorical_features = [1,3,4,5,6,7,8,9,13,14]

                fig, axs = plt.subplots(5, 3, figsize=(15, 20))

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

                plt.tight_layout()
                plt.show()
                plt.savefig(f'./miniboone_{nl}_{w}_{ml}_{lr}_{fltyp}_{rbst}_{vlay}.png')
                del ds_gn
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(e)
                print(e.with_traceback())
                pass


