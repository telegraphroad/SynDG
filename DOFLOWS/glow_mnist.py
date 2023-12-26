# %% [markdown]
# <a href="https://colab.research.google.com/github/VincentStimper/normalizing-flows/blob/master/examples/glow_colab.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# # Image generation with Glow

# %% [markdown]
# Here, we show how a flow can be trained to generate images with the `normflows` package. The flow is a class-conditional [Glow](https://arxiv.org/abs/1807.03039) model, which is based on the [multi-scale architecture](https://arxiv.org/abs/1605.08803). This Glow model is applied to the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

# %% [markdown]
# ## Perparation

# %% [markdown]
# To get started, we have to install the `normflows` package.

# %%


# %%
# Import required packages
import torch
import torchvision as tv
import numpy as np
import normflows as nf

from matplotlib import pyplot as plt
from tqdm import tqdm

# %% [markdown]
# Now that we imported the necessary packages, we create a flow model. Glow consists of `nf.flows.GlowBlocks`, that are arranged in a `nf.MultiscaleFlow`, following the multi-scale architecture. The base distribution is a `nf.distributions.ClassCondDiagGaussian`, which is a diagonal Gaussian with mean and standard deviation dependent on the class label of the image.

# %%
# Set up model

# Define flows
L = 2
K = 32
torch.manual_seed(0)

input_shape = (1, 28, 28)
n_dims = np.prod(input_shape)
channels = 1
hidden_channels = 512
split_mode = 'channel'
scale = True
num_classes = 10

# Set up flows, distributions and merge operations
q0 = []
merges = []
flows = []
for i in range(L):
    flows_ = []
    for j in range(K):
        flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                     split_mode=split_mode, scale=scale)]
    flows_ += [nf.flows.Squeeze()]
    flows += [flows_]
    if i > 0:
        merges += [nf.flows.Merge()]
        latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i),
                        input_shape[2] // 2 ** (L - i))
    else:
        latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L,
                        input_shape[2] // 2 ** L)
    q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]


# Construct flow model with the multiscale architecture
model = nf.MultiscaleFlow(q0, flows, merges)

# %%
# Move model on GPU if available
enable_cuda = True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
model = model.to(device)

# %% [markdown]
# With `torchvision` we can download the CIFAR-10 dataset.

# %%
# Prepare training data
batch_size = 128

normalize = tv.transforms.Normalize((0.1307,), (0.3081,))

transform = tv.transforms.Compose([tv.transforms.Grayscale(num_output_channels=3),tv.transforms.ToTensor(), normalize])
transform = tv.transforms.Compose([tv.transforms.ToTensor(), normalize])
train_data = tv.datasets.MNIST('datasets/', train=True,
                                 download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                           drop_last=True)

test_data = tv.datasets.MNIST('datasets/', train=False,
                                download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

train_iter = iter(train_loader)

# %% [markdown]
# ## Training

# %% [markdown]
# Now, can train the model on the image data.

# %%
# Train model
max_iter = 20000

loss_hist = np.array([])

optimizer = torch.optim.Adamax(model.parameters(), lr=5e-4, weight_decay=5e-4)
pbar = tqdm(range(max_iter))
for i in pbar:
    try:
        x, y = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y = next(train_iter)
    optimizer.zero_grad()
    loss = model.forward_kld(x.to(device), y.to(device))

    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()

    loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
    pbar.set_description(f"Loss: {loss:.4f}")
    if i % 100 == 1:
        num_sample = 10

        # with torch.no_grad():
        #     x_ = torch.clamp(x, 0, 1)
        #     plt.figure(figsize=(10, 10))
        #     plt.imshow(np.transpose(tv.utils.make_grid(x_, nrow=num_classes).cpu().numpy(), (1, 2, 0)))
        #     plt.show()
        # num_sample = 10

        with torch.no_grad():
            y = torch.arange(num_classes).repeat(num_sample).to(device)
            x, _ = model.sample(y=y)
            x_ = torch.clamp(x, 0, 1)
            plt.figure(figsize=(10, 10))
            plt.imshow(np.transpose(tv.utils.make_grid(x_, nrow=num_classes).cpu().numpy(), (1, 2, 0)))
            plt.show()

# %%
plt.figure(figsize=(10, 10))
plt.plot(loss_hist, label='loss')
plt.legend()
plt.show()

# %% [markdown]
# ## Evaluation

# %% [markdown]
# To evaluate our model, we first draw samples from our model. When sampling, we can specify the classes, so we draw then samples from each class.

# %%
# Model samples
num_sample = 10

with torch.no_grad():
    y = torch.arange(num_classes).repeat(num_sample).to(device)
    x, _ = model.sample(y=y)
    x_ = torch.clamp(x, 0, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.transpose(tv.utils.make_grid(x_, nrow=num_classes).cpu().numpy(), (1, 2, 0)))
    plt.show()

# %% [markdown]
# For quantitative evaluation, we can compute the bits per dim of our model.

# %%
# Get bits per dim
n = 0
bpd_cum = 0
with torch.no_grad():
    for x, y in iter(test_loader):
        nll = model(x.to(device), y.to(device))
        nll_np = nll.cpu().numpy()
        bpd_cum += np.nansum(nll_np / np.log(2) / n_dims + 8)
        n += len(x) - np.sum(np.isnan(nll_np))

    print('Bits per dim: ', bpd_cum / n)

# %% [markdown]
# Note that to get competitive performance, a much larger model then specified in this notebook, which is trained over 100 thousand to 1 million iterations, is needed.


