# %%
# Application of FL task
from MLModel import *
from FLModel import *
from utils import *

from torchvision import datasets, transforms
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
device = 'cuda'

# %%
def load_cnn_mnist(num_users):
    train = datasets.FashionMNIST(root="~/data/", train=True, download=True, transform=transforms.ToTensor())
    train_data = train.data.float().unsqueeze(1)
    train_label = train.targets

    mean = train_data.mean()
    std = train_data.std()
    train_data = (train_data - mean) / std

    test = datasets.FashionMNIST(root="~/data/", train=False, download=True, transform=transforms.ToTensor())
    test_data = test.data.float().unsqueeze(1)
    test_label = test.targets
    test_data = (test_data - mean) / std

    # split MNIST (training set) into non-iid data sets
    non_iid = []
    user_dict = mnist_noniid(train_label, num_users)
    for i in range(num_users):
        idx = user_dict[i]
        d = train_data[idx]
        targets = train_label[idx].float()
        non_iid.append((d, targets))
    non_iid.append((test_data.float(), test_label.float()))
    return non_iid

# %%
"""
1. load_data
2. generate clients (step 3)
3. generate aggregator
4. training
"""
client_num = 1
d = load_cnn_mnist(client_num)

# %%
"""
FL model parameters.
"""
import warnings
warnings.filterwarnings("ignore")
for lr in [0.0001,0.0005,0.001,0.005,0.01,0.02,0.04,0.08,0.1,0.2,0.3,0.4,0.5,0.7,0.9]:
    fl_param = {
        'output_size': 10,
        'client_num': client_num,
        'model': VAE,
        'data': d,
        'lr': lr,
        'E': 100,
        'C': 1,
        'eps': 50000.0,
        'delta': 1e-5,
        'q': 0.01,
        'clip': 4.,
        'tot_T': 50,
        'batch_size': 2048,
        'device': 'cuda'
    }

    fl_entity = FLServer(fl_param).to(device)
    model = fl_entity.global_model
    model.eval()

    # Generate 36 images
    num_samples = 100
    samples = model.sample(num_samples).detach().view(num_samples, 1, 28, 28)

    # Make a grid from the images
    grid = vutils.make_grid(samples, nrow=10, padding=2, normalize=True).cpu().detach()

    # Convert grid to numpy for plotting
    grid_np = grid.numpy().transpose((1, 2, 0))

    # Plot the grid
    plt.figure(figsize=(10, 10))
    plt.imsave('sample_grid.png', grid_np, cmap='viridis')
    plt.figure(figsize=(20, 20), dpi=300)
    plt.imshow(grid_np, cmap='viridis')
    plt.axis('off')  # to hide the axis
    t = -1
    plt.savefig(f'/home/samiri/ttmm/Federated-Learning-with-Differential-Privacy/VAE_MNIST_{t+1}_{lr}.png', bbox_inches='tight', pad_inches=0)
    plt.close()
    model.train()
    fl_entity.global_model.train()
    # %%
    import time

    acc = []
    start_time = time.time()
    for t in range(fl_param['tot_T']):
        fl_entity.global_update()
        
        model = fl_entity.global_model
        with torch.no_grad():
            model.eval()

            # Generate 36 images
            num_samples = 100
            samples = model.sample(num_samples).detach().view(num_samples, 1, 28, 28)
            grid = vutils.make_grid(samples, nrow=10, padding=2, normalize=True).cpu().detach()

            # Convert grid to numpy for plotting
            grid_np = grid.numpy().transpose((1, 2, 0))        
            plt.figure(figsize=(20, 20), dpi=300)
            plt.imshow(grid_np, cmap='viridis')
            plt.axis('off')  # to hide the axis
            plt.savefig(f'/home/samiri/ttmm/Federated-Learning-with-Differential-Privacy/VAE_MNIST_{t+1}_{lr}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
            
        model.train()



# %%
# SGD (mnt=0.9)

