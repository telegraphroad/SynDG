# %% [markdown]
# # Normalizing flows

# %%
# Import required packages
import os
import torch
import torchvision as tv
import numpy as np
import normflows as nf
import torchvision
import torch.nn.functional as F
import logging
import random

from matplotlib import pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
from IPython.display import Image
from torchvision import datasets
from fastDP import PrivacyEngine
from opacus.validators import ModuleValidator


# %%
# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')

# %%
def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f'Set seed {seed}')

set_seed(42)  # Use any number you like here

# %%
img_size = 28
mnist_channels = 1

# Define flows
L = 2
K = 2
torch.manual_seed(0)

input_shape = (mnist_channels, img_size, img_size)
n_dims = np.prod(input_shape)
channels = mnist_channels
hidden_channels = 128
split_mode = 'channel'
scale = True
num_classes = 10

# Set up flows, distributions and merge operations
q0 = []
merges = []
batch_size = 128
num_samples = 32
n_flows = 6
n_bottleneck = n_dims
b = torch.tensor(n_bottleneck // 2 * [0, 1] + n_bottleneck % 2 * [0])
flows = []
for i in range(n_flows):
    s = nf.nets.MLP([n_bottleneck,n_bottleneck, n_bottleneck])
    t = nf.nets.MLP([n_bottleneck,n_bottleneck, n_bottleneck])
    if i % 2 == 0:
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
# Construct flow model with the multiscale architecture
#model = nf.MultiscaleFlow(q0, flows, merges)
q0 = torch.distributions.MultivariateNormal(torch.zeros(n_bottleneck, device=device),
                                               torch.eye(n_bottleneck, device=device))
q0 = nf.distributions.DiagGaussian([n_dims])
model = nf.NormalizingFlow(q0=q0, flows=flows)
model = model.to(device)

# %% [markdown]
# Let's load the MNIST dataset

# %%
img_size = 28
batch_size = 32

# Create a directory to store the MNIST dataset.
# 'exist_ok=True' makes sure the command does not fail if the directory already exists.
os.makedirs("../mnist/data", exist_ok=True)

# Create a DataLoader to handle batching of the MNIST dataset.
dataloader = torch.utils.data.DataLoader(

    # Load the MNIST dataset from PyTorch's built-in datasets.
    datasets.FashionMNIST(
        # Set the directory where the data is stored or where it will be downloaded.
        "../mnist/data",
        # Use the training portion of the dataset.
        train=True,
        # Download the dataset if it is not already in the specified directory.
        download=True,
        # Apply several transformations to the images in the dataset:
        transform=transforms.Compose(
            [
                # Resize images to the size specified by the 'img_size' variable.
                #transforms.Resize(img_size),
                # Convert images from PIL format to PyTorch tensors.
                transforms.ToTensor(),
                # Normalize tensors so that the pixel intensity values have a mean of 0.5 and a standard deviation of 0.5.
                transforms.Normalize([0.5], [0.5]),
                transforms.Lambda(lambda x: x.view(n_dims)),  
            ]
        ),
    ),

    # Set the number of samples per batch to load.
    batch_size=batch_size,
    # Enable shuffling to feed data points in a random manner.
    shuffle=True,
)

train_iter = iter(dataloader)

# %%
examples = enumerate(dataloader)

batch_idx, (example_data, example_targets) = next(examples)
example_data = example_data.view([32,1,28,28])
# There is some annoying warning regarding clipping because of the scaling of the data and this ignores is
logger = logging.getLogger()
old_level = logger.level
logger.setLevel(100)

fig = plt.figure(figsize=(20,20))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.tight_layout()
    plt.imshow((example_data[i].permute(1, 2, 0)), interpolation='none')
    plt.title("Ground Truth: {}".format(example_targets[i]), fontsize=16)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout();

logger.setLevel(old_level)

# %%
model.sample()

# %%
epochs = 100
num_sample = 32

max_iter = epochs
loss_hist = np.array([])
optimizer = torch.optim.Adamax(model.parameters(), lr=5e-5, weight_decay=5e-6)
dp = False

if dp:
    model = ModuleValidator.fix(model);
    target_epsilon = 0.6
    privacy_engine = PrivacyEngine(
        model,
        batch_size=batch_size,
        sample_size=len(dataloader.dataset),
        epochs=max_iter,
        target_epsilon=target_epsilon,
        clipping_fn='automatic',
        clipping_mode='MixOpt',
        origin_params=None,
        clipping_style='layer-wise'#'all-layer',
    )
    privacy_engine.attach(optimizer)

for i in tqdm(range(epochs)):
    loss = 0
    for x, y in dataloader:
        optimizer.zero_grad()
        ls = model.forward_kld(x.to(device))
        if ~(torch.isnan(ls) | torch.isinf(ls)):
            ls.backward()
            optimizer.step()
        loss += ls.item()
    fig = plt.figure(figsize=(20,20))
    for i in range(36):
        plt.subplot(6,6,i+1)
        plt.tight_layout()
        x, _ = model.sample()
        x = x.view([1,28,28])
        x_ = torch.clamp(x, 0, 1).cpu().detach()

        plt.imshow((x_.permute(1, 2, 0)), interpolation='none')
        #plt.title("Ground Truth: {}".format(example_targets[i]), fontsize=16)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout();
    plt.show()


        #print(model.sample())
    # y = torch.arange(num_classes).repeat(num_sample).to(device)
    # x, _ = model.sample()
    # x = x.view([32,1,28,28])
    # x_ = torch.clamp(x, 0, 1).cpu().detach()

    # fig = plt.figure(figsize=(20,20))
    # for i in range(36):
    #     plt.subplot(6,6,i+1)
    #     plt.tight_layout()
    #     plt.imshow((x_[i].permute(1, 2, 0)), interpolation='none')
    #     #plt.title("Ground Truth: {}".format(example_targets[i]), fontsize=16)
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.tight_layout();
    # plt.show()


    loss_hist = np.append(loss_hist, loss)

# %% [markdown]
# 

# %%

fig = plt.figure(figsize=(20,20))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.tight_layout()
    x, _ = model.sample()
    x = x.view([1,28,28])
    x_ = torch.clamp(x, 0, 1).cpu().detach()

    plt.imshow((x_.permute(1, 2, 0)), interpolation='none')
    #plt.title("Ground Truth: {}".format(example_targets[i]), fontsize=16)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout();
plt.show()


# %%
x

# %%
# Model samples
num_sample = 60

with torch.no_grad():
    y = torch.arange(num_classes).repeat(num_sample).to(device)
    x, _ = model.sample(y=y)
    x_ = torch.clamp(x, 0, 1)
    plt.figure(figsize=(20, 20))
    plt.imshow(np.transpose(tv.utils.make_grid(x_, nrow=num_classes).cpu().numpy(), (1, 2, 0)))
    plt.show()

y = torch.arange(num_classes).repeat(num_sample).to(device)
x, _ = model.sample(y=y)
x_ = torch.clamp(x, 0, 1).cpu().detach()

fig = plt.figure(figsize=(20,20))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.tight_layout()
    plt.imshow((x_[i].permute(1, 2, 0)), interpolation='none')
    #plt.title("Ground Truth: {}".format(example_targets[i]), fontsize=16)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout();

#

# %%
# Get bits per dim
n = 0
bpd_cum = 0
with torch.no_grad():
    for x, y in iter(dataloader):
        nll = model(x.to(device), y.to(device))
        nll_np = nll.cpu().numpy()
        bpd_cum += np.nansum(nll_np / np.log(2) / n_dims + 8)
        n += len(x) - np.sum(np.isnan(nll_np))

    print('Bits per dim: ', bpd_cum / n)

# %%
plt.plot(loss_hist)

# %%



