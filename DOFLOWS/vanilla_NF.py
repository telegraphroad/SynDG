# %%
# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
from torchvision import datasets, transforms
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import gaussian_noise
from rdp_analysis import calibrating_sampled_gaussian
from utils import mnist_noniid
from tqdm import tqdm
import argparse
import gc
import normflows as nf
import copy

parser = argparse.ArgumentParser(description='Train a NF model.')

# Add the arguments
parser.add_argument('--clip', type=float, default=1.0, required=False, help='The clip value.')
parser.add_argument('--lr', type=float, default = 0.0001, required=False, help='The learning rate.')

# bs = 100
# # MNIST Dataset
# train_dataset = 
# test_dataset = datasets.FashionMNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# # Data Loader (Input Pipeline)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
def load_cnn_mnist(num_users):
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
        )
    train = datasets.FashionMNIST(root="~/data/", train=True, download=True, transform=transform)
    train_data = train.data.float().unsqueeze(1)
    train_label = train.targets

    mean = train_data.mean()
    std = train_data.std()
    train_data = (train_data - mean) / std

    test = datasets.FashionMNIST(root="~/data/", train=False, download=True, transform=transform)
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


class DPNF(nn.Module):
    def __init__(self,model, E,q,batch_size,clip,eps,delta,lr,dp=False,iters=50,device='cuda',x_dim=784, h_dim1=512, h_dim2=256, z_dim=2,momentum=0.9):
        super(DPNF, self).__init__()
        self.E = E
        self.q = q
        self.BATCH_SIZE = batch_size
        self.device = device
        self.clip = clip
        self.eps = eps
        self.delta = delta
        self.iters = iters
        self.best_params = None
        transform=transforms.Compose(
            [
                # Resize images to the size specified by the 'img_size' variable.
                #transforms.Resize(img_size),
                # Convert images from PIL format to PyTorch tensors.
                transforms.ToTensor(),
                # Normalize tensors so that the pixel intensity values have a mean of 0.5 and a standard deviation of 0.5.
                #transforms.Normalize([0.5], [0.5]),
                #transforms.Lambda(lambda x: x.view(n_dims)),  
            ]
        )

        
        #self.train_dataset = datasets.FashionMNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
        self.train_dataset = datasets.FashionMNIST(root='./mnist_data/', train=True, transform=transform, download=True)
        #_ds = datasets.FashionMNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=True)
        _ds = datasets.FashionMNIST(root='./mnist_data/', train=False, transform=transform, download=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=_ds, batch_size=self.BATCH_SIZE, shuffle=True)
        
        self.data_size = len(self.train_dataset)
        self.model = model
        self.optimizer = optim.SGD(self.model.parameters(),lr=lr,momentum=momentum)
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr)
        if dp:
            self.sigma = calibrating_sampled_gaussian(self.q, self.eps, self.delta, self.E*self.iters, err=1e-3)
        else:
            self.sigma = 0.0

    def loss_function(self,recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD

    def loss_function_dp(self,recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='none')
        BCE = torch.mean(BCE, dim=1)
        KLD = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
        return BCE + KLD

    def trainmodel(self):
        if self.best_params is not None:
            self.model.load_state_dict(self.best_params)
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.model.train()
        train_loss = 0
        self.best_params = copy.deepcopy(model.state_dict())
        bestloss = 1e12
        for batch_idx, (data, _) in tqdm(enumerate(train_loader)):
        #for batch_idx, (data, _) in enumerate(train_loader):
            try:
                data = data.cuda()
                self.optimizer.zero_grad()
                _dsh = data.shape
                loss = self.model.forward_kld(data.view(_dsh[0],_dsh[-1]*_dsh[-1]))
                if ~(torch.isnan(loss) | torch.isinf(loss)):
                
                    loss.backward()
                    train_loss += loss.item()
                    self.optimizer.step()
                    with torch.no_grad():
                        if loss.item()<bestloss:
                            _s,_ = model.sample(1000)
                            if (torch.isnan(_s).sum()==0 & torch.isinf(_s).sum()==0):
                            
                                bestloss = copy.deepcopy(loss.item())
                                self.best_params = copy.deepcopy(self.model.state_dict())    
                                print('best loss: ',bestloss) 
                        # else:
                        #     self.model.load_state_dict(best_params)           

            except Exception as e:
                self.model.load_state_dict(self.best_params)
            
        #print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    def traindpmodel(self):
        self.model.train()
        print(r"$\sigma$: ", self.sigma)
        train_loss = 0
        
        self.best_params = copy.deepcopy(model.state_dict())
        bestloss = 1e12
        for e in range(self.E):
            try:
                idx = np.where(np.random.rand(len(self.train_dataset.data[:])) < self.q)[0]
                sampled_dataset = TensorDataset(self.train_dataset.data[idx], self.train_dataset.targets[idx])
                sample_data_loader = DataLoader(dataset=sampled_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

                self.optimizer.zero_grad()
                clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
                pbar = tqdm(sample_data_loader, desc=f"Epoch {e+1}/{self.E}, Best loss: {bestloss}")

                #for batch_x, _ in sample_data_loader:
                for batch_x, _ in pbar:
                    torch.cuda.empty_cache()
                    batch_x = batch_x.to(self.device)
                    _dsh = batch_x.shape
                    loss = self.model.forward_kld(batch_x.view(_dsh[0],_dsh[-1]*_dsh[-1]).float(),extended=True)
                    #loss = self.loss_function_dp(recon_batch, batch_x.view(-1,784).float(), mu, log_var)
                    self.model.sample(1000)
                    if not ((torch.isnan(loss).sum()>0).item() | (torch.isinf(loss).sum()>0).item()):
                        for i in range(loss.size()[0]):
                            loss[i].backward(retain_graph=True)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)

                            for name, param in self.model.named_parameters():
                                clipped_grads[name] += param.grad 

                            self.model.zero_grad()
                        pbar.set_description(f"Epoch {e+1}/{self.E}, Loss: {loss.mean().item()}")
                for name, param in self.model.named_parameters():
                    clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, self.clip, self.sigma, device=self.device)

                for name, param in self.model.named_parameters():
                    clipped_grads[name] /= (self.data_size*self.q)

                for name, param in self.model.named_parameters():
                    param.grad = clipped_grads[name]

                self.optimizer.step()        
                torch.cuda.empty_cache()
                with torch.no_grad():
                    if not ((torch.isnan(loss).sum()>0).item() | (torch.isinf(loss).sum()>0).item()):
                        if loss.sum().item()<bestloss:
                            
                            _s,_ = model.sample(1000)
                            if (torch.isnan(_s).sum()==0 & torch.isinf(_s).sum()==0):
                            
                                bestloss = copy.deepcopy(loss.item())
                                self.best_params = copy.deepcopy(self.model.state_dict())                
                                print('best loss: ',bestloss)            
            except Exception as e:
                print(e)
                print(e.with_traceback())
                self.model.load_state_dict(self.best_params)




# %%

E = 10
q = 0.5
batch_size = 1024
clip = 0.2
x_dim=784
h_dim1= 512
h_dim2=256
z_dim=2
lr=1e-1
momentum=0.9
C = 1
eps = 4000.0
delta = 1e-5
iters = 50
device='cuda'
args = parser.parse_args()

# Now you can use args.clip and args.lr in your script
clip = args.clip
lr = args.lr

client_num = 1

d = load_cnn_mnist(client_num)

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
batch_size = 512
num_samples = 32
n_flows = 12
n_bottleneck = n_dims
b = torch.tensor(n_bottleneck // 2 * [0, 1] + n_bottleneck % 2 * [0])
flows = []
for i in range(n_flows):
    s = nf.nets.MLP([n_bottleneck,n_bottleneck*3, n_bottleneck])
    t = nf.nets.MLP([n_bottleneck,n_bottleneck*3, n_bottleneck])
    if i % 2 == 0:
        flows += [nf.flows.MaskedAffineFlow(b, t, s)]
    else:
        flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
# Construct flow model with the multiscale architecture
#model = nf.MultiscaleFlow(q0, flows, merges)
q0 = torch.distributions.MultivariateNormal(torch.zeros(n_bottleneck, device=device),
                                               torch.eye(n_bottleneck, device=device))
q0 = nf.distributions.DiagGaussian([n_dims])
q0 = nf.distributions.base.GaussianMixture(10,n_dims)
model = nf.NormalizingFlow(q0=q0, flows=flows)
model = model.to(device)
dp = False
vae = DPNF(model,E,q,batch_size,clip,eps,delta,lr,dp,iters,device)
if torch.cuda.is_available():
    vae.cuda()

#optimizer = optim.SGD(vae.parameters(),lr=0.000001,momentum=0.9)
train_loader = torch.utils.data.DataLoader(dataset=vae.train_dataset, batch_size=vae.BATCH_SIZE, shuffle=True)
num_samples = 100
samples = []
for data in train_loader:
    inputs, _ = data
    samples.append(inputs)
    if len(samples) >= num_samples:
        break

samples = torch.cat(samples, 0)[:num_samples].detach().view(num_samples, 1, 28, 28)
grid = vutils.make_grid(samples, nrow=10, padding=2, normalize=True).cpu()
grid_np = grid.numpy().transpose((1, 2, 0))

plt.figure(figsize=(20, 20), dpi=300)
plt.imshow(grid_np, cmap='viridis')  # Change 'viridis' to 'gray'
plt.axis('off')  # to hide the axis
torch.cuda.empty_cache()
plt.savefig(f"images/NF.png")
del train_loader

print(model.sample(10)[0])
print(lr,clip,'+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(model.sample(10)[0])
for epoch in range(iters):
    if dp:
        vae.traindpmodel()
        gc.collect()
        torch.cuda.empty_cache()
    else:
        vae.trainmodel()
    torch.cuda.empty_cache()
    vae.model.eval()

    # Generate 36 images
    num_samples = 100
    samples,_ = vae.model.sample(num_samples)
    samples = samples.detach().view(num_samples, 1, 28, 28)
    grid = vutils.make_grid(samples, nrow=10, padding=2, normalize=True).cpu().detach()
    grid_np = grid.numpy().transpose((1, 2, 0))

    plt.figure(figsize=(20, 20), dpi=300)
    plt.imshow(grid_np, cmap='viridis')  # Change 'viridis' to 'gray'    
    plt.axis('off')  # to hide the axis
    torch.cuda.empty_cache()
    plt.savefig(f"images/NF_{dp}_clip_{clip}_lr_{lr}_epoch_{epoch}_sigma_{vae.sigma}.png")
    plt.close()
    vae.model.train()
    torch.cuda.empty_cache()
    gc.collect()

# %%
