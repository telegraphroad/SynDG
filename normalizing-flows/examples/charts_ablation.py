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


# Set up model
enable_cuda = True

device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')


# %%
enable_cuda = True

device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
trnbl = True


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
import pandas as pd

grid_size = 200
device='cuda'

ablation_results = pd.read_csv('../../ablation_results.csv').drop(columns=['Unnamed: 0'])
ablation_results.columns = ['std','beta','base_distribution','trainable','stats','pval','interpretation','variable']
sns.set_style("darkgrid")


line_xpos = 0.217  # Adjust this as needed

from matplotlib.lines import Line2D

print(len(ablation_results))
# line = Line2D([line_xpos, line_xpos], [0.075, 0.910], transform=fig.transFigure, color='red', linestyle='--',lw=3)
# plt.subplots_adjust(hspace=0.05, wspace=0.01)
# plt.tight_layout()
# sns.set_style("darkgrid")
# plt.savefig('2d_results.png',bbox_inches='tight')

# %%
import seaborn as sns
import matplotlib.pyplot as plt
df = ablation_results
# Assuming df is your DataFrame
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

unique_values = df['base_distribution'].unique()
print(unique_values)
# Create a new column 'group' that contains the combination of 'base', 'trainable', and 'variable'

df['group'] = df['base_distribution'].astype(str) + ', ' + df['trainable'].astype(str)

df_v0 = df[df['variable'] == 'v0']
df_v1 = df[df['variable'] == 'v1']
n_colors = df['group'].nunique()

plt.figure(figsize=(10,6))
sns.lineplot(data=df_v0, x='beta', y='pval', hue='group', palette=sns.color_palette("viridis", n_colors))

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Pval vs Beta for different combinations of base, trainable, and variable')
plt.show()# %%

plt.figure(figsize=(10,6))
sns.lineplot(data=df_v1, x='beta', y='pval', hue='group', palette=sns.color_palette("viridis", n_colors))

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title('Pval vs Beta for different combinations of base, trainable, and variable')
plt.show()

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 2, figsize=(20,10
                                       ))


rowctr = 0
alpha = 0.9
for b in ablation_results['base_distribution'].unique():
    # Print unique values of the 'base_distribution' column
    
    if b == 'Gaussian':
        continue
    df = ablation_results[ablation_results['base_distribution'] == b]
    print(unique_values)

    # Create a new column 'group' that contains the combination of 'base', 'trainable', and 'variable'
    df['group'] = df['base_distribution'].astype(str) + ', ' + df['trainable'].astype(str)

    # Split the DataFrame based on the 'variable' column
    df_v0 = df[df['variable'] == 'v0']
    df_v1 = df[df['variable'] == 'v1']

    # Determine the number of unique groups
    n_colors = df['group'].nunique()

    # Create a figure with two subplots arranged in two columns

    # Plot the data for 'v0' in the first subplot
    sns.lineplot(data=df_v0, x='beta', y='pval', hue='group', ax=axs[rowctr,0],alpha=alpha)
    #axs[rowctr,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[rowctr,0].set_title('Pval vs Beta for different combinations of base, trainable (variable = v0)')

    # Plot the data for 'v1' in the second subplot
    sns.lineplot(data=df_v1, x='beta', y='pval', hue='group', ax=axs[rowctr,1],alpha=alpha)
    alpha -= 0.3
    #axs[rowctr,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[rowctr,1].set_title('Pval vs Beta for different combinations of base, trainable (variable = v1)')
    rowctr+=1
    # Display the figure
plt.tight_layout()
plt.show()

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(20,10))

rowctr = 0
alpha = 0.9
for b in ablation_results['base_distribution'].unique():
    # Print unique values of the 'base_distribution' column
    
    if b == 'Gaussian':
        continue
    df = ablation_results[ablation_results['base_distribution'] == b]

    # Create a new column 'group' that contains the combination of 'base', 'trainable', and 'variable'
    df['group'] = df['base_distribution'].astype(str) + ', ' + df['trainable'].astype(str)

    # Split the DataFrame based on the 'variable' column
    df_v0 = df[df['variable'] == 'v0']
    df_v1 = df[df['variable'] == 'v1']

    # Determine the number of unique groups
    n_colors = df['group'].nunique()

    # Create a figure with two subplots arranged in two columns

    # Plot the data for 'v0' in the first subplot
    sns.lineplot(data=df_v0, x='beta', y='pval', hue='group', ax=axs[rowctr,0],alpha=alpha)
    axs[rowctr,0].axhline(0.05, color='r', linestyle='--')
    axs[rowctr,0].annotate("Null\nHypothesis", xy=(-0.05, 0.07), xycoords='axes fraction', color='r')
    axs[rowctr,0].set_title('Pval vs Beta for different combinations of base, trainable (variable = v0)')
    axs[rowctr,0].set_ylim([0,1])  # set y limit here

    # Plot the data for 'v1' in the second subplot
    sns.lineplot(data=df_v1, x='beta', y='pval', hue='group', ax=axs[rowctr,1],alpha=alpha)
    axs[rowctr,1].axhline(0.05, color='r', linestyle='--')
    axs[rowctr,1].annotate("Null\nHypothesis", xy=(-0.05, 0.07), xycoords='axes fraction', color='r')
    axs[rowctr,1].set_title('Pval vs Beta for different combinations of base, trainable (variable = v1)')
    axs[rowctr,1].set_ylim([0,1])  # set y limit here
    
    alpha -= 0.3
    rowctr+=1
    # Display the figure
plt.tight_layout()
plt.show()

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

fig, axs = plt.subplots(2, 2, figsize=(20,10))

rowctr = 0
alpha = 0.9
for b in ablation_results['base_distribution'].unique():
    # Print unique values of the 'base_distribution' column
    sns.set_style("darkgrid")
    if b == 'Gaussian':
        continue
    df = ablation_results[ablation_results['base_distribution'] == b]

    # Create a new column 'group' that contains the combination of 'base', 'trainable', and 'variable'
    df['group'] = df['base_distribution'].astype(str) + ', ' + df['trainable'].astype(str)

    # Split the DataFrame based on the 'variable' column
    df_v0 = df[df['variable'] == 'v0']
    df_v1 = df[df['variable'] == 'v1']


    # Determine the number of unique groups
    n_colors = df['group'].nunique()
    
    # Define color palette
    palette = {group: 'silver' if group.endswith('False') else 'black' for group in df['group'].unique()}
    # Create a figure with two subplots arranged in two columns

    # Plot the data for 'v0' in the first subplot
    sns.set_style("darkgrid")
    sns.lineplot(data=df_v0, x='beta', y='pval', hue='group',palette=palette, ax=axs[rowctr,0],alpha=alpha)
    sns.set_style("darkgrid")
    axs[rowctr,0].axhline(0.05, color='r', linestyle='--')
    axs[rowctr,0].annotate("Null\nHypothesis", xy=(-0.05, 0.07), xycoords='axes fraction', color='darkorange')
    if rowctr==0:
        axs[rowctr,0].annotate(r"Student's t $\nu=1000$", xy=(0.79, 0.87), xycoords='axes fraction', color='darkorange')
        axs[rowctr,0].annotate(r"GGD $\beta=1$", xy=(0.79, 0.80), xycoords='axes fraction', color='darkorange')
        axs[rowctr,0].annotate(r"Student's t $\nu=1$", xy=(0.05, 0.87), xycoords='axes fraction', color='darkorange')
        axs[rowctr,0].annotate(r"GGD $\beta=4.33$", xy=(0.05, 0.80), xycoords='axes fraction', color='darkorange')
        axs[rowctr,0].set_title('p values for variable 1, shorter tailed', fontsize=16)
        axs[rowctr,0].set_xlabel('')
        
        axs[rowctr,0].set_xticklabels([])
        axs[rowctr,1].set_xticklabels([])
    axs[rowctr,0].set_ylim([0,1])  # set y limit hereta
    axs[rowctr,1].set_yticklabels([])
    

    # Plot the data for 'v1' in the second subplot
    sns.lineplot(data=df_v1, x='beta', y='pval', hue='group',palette=palette, ax=axs[rowctr,1],alpha=alpha)
    axs[rowctr,1].axhline(0.05, color='r', linestyle='--')
    axs[rowctr,1].annotate("Null\nHypothesis", xy=(-0.05, 0.07), xycoords='axes fraction', color='darkorange')
    if rowctr==0:
        axs[rowctr,1].annotate(r"Student's t $\nu=1000$", xy=(0.79, 0.87), xycoords='axes fraction', color='darkorange')
        axs[rowctr,1].annotate(r"GGD $\beta=4.33$", xy=(0.79, 0.80), xycoords='axes fraction', color='darkorange')
        axs[rowctr,1].annotate(r"Student's t $\nu=1$", xy=(0.057, 0.87), xycoords='axes fraction', color='darkorange')
        axs[rowctr,1].annotate(r"GGD $\beta=4.33$", xy=(0.057, 0.80), xycoords='axes fraction', color='darkorange')
        axs[rowctr,1].set_title('p values for variable 2, heavier tailed', fontsize=16)
        axs[rowctr,1].set_xlabel('')
    axs[rowctr,1].set_ylim([0,1])  # set y limit here
    axs[rowctr,0].grid(True)
    axs[rowctr,1].grid(True)
    alpha -= 0.3
    rowctr+=1
    # Display the figure

for ax in axs.flat:
    ax.get_legend().remove() 
x_start_rel = 0.055  # x relative position of vertical line (for x=1000)
line = Line2D([x_start_rel, x_start_rel], [0.06, 0.98], transform=fig.transFigure, color='darkorange', linestyle='--', linewidth=1.5)
fig.lines.append(line)

x_start_rel = 0.48  # x relative position of vertical line (for x=1000)
line = Line2D([x_start_rel, x_start_rel], [0.06, 0.98], transform=fig.transFigure, color='darkorange', linestyle='--', linewidth=1.5)
fig.lines.append(line)

x_start_rel = 0.55  # x relative position of vertical line (for x=1000)
line = Line2D([x_start_rel, x_start_rel], [0.06, 0.98], transform=fig.transFigure, color='darkorange', linestyle='--', linewidth=1.5)
fig.lines.append(line)

x_start_rel = 0.972  # x relative position of vertical line (for x=1000)
line = Line2D([x_start_rel, x_start_rel], [0.06, 0.98], transform=fig.transFigure, color='darkorange', linestyle='--', linewidth=1.5)
fig.lines.append(line)

row_labels = ["GGD base", "Mixture of GGD base"]
for row in range(2):
    axs[row, 0].set_ylabel(row_labels[row], rotation=90, labelpad=9, verticalalignment='center', fontsize=16)

axs[1,0].set_xlabel(r'$\gamma$')
axs[1,1].set_xlabel(r'$\gamma$')
plt.tight_layout()
plt.savefig('ablation_results.png',bbox_inches='tight',dpi=300)

# %%
import numpy as np
from scipy.stats import genpareto, t, norm, gamma, beta, gennorm

# Extend the figure to accommodate the new row
fig, axs = plt.subplots(1, 4, figsize=(20,5))

# Your previous code...

# Add the new plots
x = np.linspace(-5, 5, 1000)

# PDF of a gennorm with beta=4.33 and a student t with nu=1
axs[0].plot(x, gennorm.pdf(x, beta=4.33), label=r'GGD, $\beta$=4.33')
axs[0].plot(x, t.pdf(x, df=1), label=r"Student's t, $\nu$=1")
axs[0].legend()
axs[0].set_title(r'PDF of ablation random variables for $\gamma$=0',fontsize=16)



# PDFs of gennorm with beta =1 and student t with nu = 1001
axs[1].plot(x, gennorm.pdf(x, beta=1), label=r'GGD, $\beta$=1')
axs[1].plot(x, t.pdf(x, df=1001), label=r"Student's t, $\nu$=1001")
axs[1].legend()
axs[1].set_title(r'PDF of ablation random variables for $\gamma$=1000',fontsize=16)

# nu = 1 + gamma as gamma goes from 0 to 1000
gamma_values = np.linspace(0, 1000, 1000)
nu_values = 1 + gamma_values
axs[2].plot(gamma_values, nu_values)
axs[2].set_xlabel(r'$\gamma$')
axs[2].set_ylabel(r"Student's t $\nu$")
axs[2].set_title(r"Student's t $\nu$ as a function of $\gamma$",fontsize=16)
# beta according to the transform_beta function as gamma goes from 0 to 1000
def transform_beta(beta_val):
    beta_scaled = (1000 - beta_val) / 100
    gamma_cdf = gamma.cdf(beta_scaled, 9, scale=1)
    return 1 + gamma_cdf * 5

beta_values = transform_beta(gamma_values)
axs[3].plot(gamma_values, beta_values)
axs[3].set_xlabel(r'$\gamma$')
axs[3].set_ylabel(r'GGD $\beta$')
axs[3].set_title(r"GGD $\beta$ as a function of $\gamma$",fontsize=16)


plt.tight_layout()
plt.savefig('beta_transform.png',bbox_inches='tight',dpi=300)
# %%
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = ablation_results

# # Create a new column 'group' that contains the combination of 'base', 'trainable', and 'variable'
# df['group'] = df['base_distribution'].astype(str) + ', ' + df['trainable'].astype(str)

# # Determine the number of unique groups
# n_colors = df['group'].nunique()

# # Create a grid of subplots, with one row for each 'base_distribution' and one column for each 'variable'
# g = sns.FacetGrid(df, row="variable", col="base_distribution", height=5, aspect=2)

# # Map a line plot to each subplot
# g.map(sns.lineplot, 'beta', 'pval', 'group', legend='full')

# # Add a legend


# # Display the figure
# plt.show()
# # %%
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = ablation_results

# # Create a new column 'group' that contains the combination of 'base_distribution' and 'variable'
# df['group'] = df['base_distribution'].astype(str) + ', ' + df['trainable'].astype(str)

# # Filter the DataFrame based on the 'trainable' column
# df_false = df[df['trainable'] == False]
# df_true = df[df['trainable'] == True]

# # Create a grid of subplots, with one row for each 'variable' and one column for each 'trainable'
# g = sns.FacetGrid(df, row="variable", col="base_distribution", height=5, aspect=2)

# # Map a line plot to each subplot, using red for 'trainable' == False and green for 'trainable' == True
# g.map(sns.lineplot, 'beta', 'pval', 'group', color='red', data=df_false)
# g.map(sns.lineplot, 'beta', 'pval', 'group', color='green', data=df_true)

# # Iterate over the axes of the FacetGrid to add a legend to each one
# for ax in g.axes.flat:
#     box = ax.get_position()
#     ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
#     handles, labels = ax.get_legend_handles_labels()
#     ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1, 0.5))

# # Display the figure
# plt.show()