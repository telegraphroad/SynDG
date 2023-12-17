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

parser = argparse.ArgumentParser(description='Train a VAE model.')

# Add the arguments
parser.add_argument('--clip', type=float, default=1.0, required=False, help='The clip value.')
parser.add_argument('--lr', type=float, default=1e-4, required=False, help='The learning rate.')

# bs = 100
# # MNIST Dataset
# train_dataset = 
# test_dataset = datasets.FashionMNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=False)

# # Data Loader (Input Pipeline)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True)
# test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)
def load_cnn_mnist(num_users):
    train = datasets.FashionMNIST(root="~/data/", train=True, download=True, transform=transforms.ToTensor())
    train_data = train.data.float().unsqueeze(1)
    train_label = train.targets

    # mean = train_data.mean()
    # std = train_data.std()
    # train_data = (train_data - mean) / std

    test = datasets.FashionMNIST(root="~/data/", train=False, download=True, transform=transforms.ToTensor())
    test_data = test.data.float().unsqueeze(1)
    test_label = test.targets
    # test_data = (test_data - mean) / std

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


class VAE(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim):
        super(VAE, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        return F.sigmoid(self.fc6(h)) 
    
    def forward(self, x):
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), mu, log_var

    def sample(self, num_samples):
        """
        Generate new data by sampling random variables and decoding them to the data space.
        
        Args:
            num_samples (int): the number of samples to generate.

        Returns:
            torch.Tensor: a tensor of generated data.
        """
        z = torch.randn(num_samples, self.fc31.out_features).to(self.fc1.weight.device)
        return self.decoder(z)

class DPVAE(nn.Module):
    def __init__(self, E,q,batch_size,clip,eps,delta,lr,iters=50,device='cuda',x_dim=784, h_dim1=512, h_dim2=256, z_dim=2,momentum=0.9):
        super(DPVAE, self).__init__()
        self.E = E
        self.q = q
        self.BATCH_SIZE = batch_size
        self.device = device
        self.clip = clip
        self.eps = eps
        self.delta = delta
        self.iters = iters
        
        
        #ds = dataset[0]   # test set
        # self.train_dataset = TensorDataset(ds[0].clone().detach().to(self.device),
        #                                    ds[1].clone().detach().to(self.device))
        self.train_dataset = datasets.FashionMNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
        _ds = datasets.FashionMNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(), download=True)
        self.test_loader = torch.utils.data.DataLoader(dataset=_ds, batch_size=self.BATCH_SIZE, shuffle=True)
        
        self.data_size = len(self.train_dataset)
        self.model = VAE(x_dim,h_dim1,h_dim2,z_dim)
        #self.optimizer = optim.SGD(self.model.parameters(),lr=lr,momentum=momentum)
        self.optimizer = optim.Adam(self.model.parameters(),lr=lr)
        self.sigma = calibrating_sampled_gaussian(self.q, self.eps, self.delta, self.E*self.iters, err=1e-3)

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
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.cuda()
            self.optimizer.zero_grad()
            
            recon_batch, mu, log_var = self.model(data)
            loss = self.loss_function(recon_batch, data, mu, log_var)
            
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            
            if batch_idx % 100 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item() / len(data)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

    def traindpmodel(self):
        self.model.train()
        print(r"$\sigma$: ", self.sigma)
        train_loss = 0
        

        for e in range(self.E):
            np.random.seed()
            idx = np.where(np.random.rand(len(self.train_dataset.data[:])) < self.q)[0]
            sampled_dataset = TensorDataset(self.train_dataset.data[idx], self.train_dataset.targets[idx])
            sample_data_loader = DataLoader(dataset=sampled_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

            self.optimizer.zero_grad()
            clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
            pbar = tqdm(sample_data_loader, desc=f"Epoch {e+1}/{self.E}")

            #for batch_x, _ in sample_data_loader:
            for batch_x, _ in pbar:
                torch.cuda.empty_cache()
                batch_x = batch_x.to(self.device)
                recon_batch, mu, log_var = self.model(batch_x.float())
                loss = self.loss_function_dp(recon_batch, batch_x.view(-1,784).float(), mu, log_var)

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

    def testmodel(self):
        self.model.eval()
        
        test_loss= 0
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.cuda()
                recon, mu, log_var = self.model(data)
                
                # sum up batch loss
                test_loss += self.loss_function(recon, data, mu, log_var).item()
            
        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def testdpmodel(self):
        self.model.eval()
        
        test_loss= 0
        with torch.no_grad():
            for data, _ in self.test_loader:
                data = data.cuda()
                recon, mu, log_var = self.model(data)
                
                # sum up batch loss
                test_loss += self.loss_function_dp(recon, data, mu, log_var).mean().item()
            
        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        gc.collect()
        torch.cuda.empty_cache()






# %%

E = 10
q = 0.1
batch_size = 4096
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
iters = 10
device='cuda'
args = parser.parse_args()

# Now you can use args.clip and args.lr in your script
clip = args.clip
lr = args.lr

client_num = 1

#d = datasets.FashionMNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)

vae = DPVAE(E,q,batch_size,clip,eps,delta,lr,iters,device)
if torch.cuda.is_available():
    vae.cuda()

#optimizer = optim.SGD(vae.parameters(),lr=0.000001,momentum=0.9)

dp = False
for epoch in range(iters):
    if dp:
        vae.traindpmodel()
        gc.collect()
        torch.cuda.empty_cache()
        vae.testdpmodel()
    else:
        vae.trainmodel()
        torch.cuda.empty_cache()
        vae.testmodel()
        torch.cuda.empty_cache()
    print()
    print("++++++++++++++++++++++++++++++++++++++++++++++Epoch: ", epoch)
    print()

vae.model.eval()

# Generate 36 images
num_samples = 100
samples = vae.model.sample(num_samples).detach().view(num_samples, 1, 28, 28)

# Make a grid from the images
grid = vutils.make_grid(samples, nrow=10, padding=2, normalize=True).cpu().detach()

# Convert grid to numpy for plotting
grid_np = grid.numpy().transpose((1, 2, 0))

# Plot the grid
plt.figure(figsize=(20, 20), dpi=300)
plt.imshow(grid_np, cmap='viridis')
plt.axis('off')  # to hide the axis
torch.cuda.empty_cache()
plt.savefig(f"images/{dp}_lr_{lr}_clip_{clip}_epoch_{epoch}_sigma_{vae.sigma}.png")
plt.close()
vae.model.train()
torch.cuda.empty_cache()
gc.collect()

# %%
