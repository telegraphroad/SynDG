# Federated Learning Model in PyTorch
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from utils import gaussian_noise
from rdp_analysis import calibrating_sampled_gaussian
import torchvision.utils as vutils
from MLModel import *

import numpy as np
import copy
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import normflows as nf

def plot_samples(data_loader, num_samples=36):
    data_iter = iter(data_loader)
    images = []

    while len(images) < num_samples:
        try:
            batch = next(data_iter)
        except StopIteration:
            # If the data_iter has no more batches, start over from the beginning.
            data_iter = iter(data_loader)
            batch = next(data_iter)

        batch_size = batch[0].size(0)
        indices = np.random.choice(batch_size, size=min(num_samples - len(images), batch_size), replace=False)

        for index in indices:
            images.append(batch[0][index])

    # Create a grid of images
    grid = vutils.make_grid(images, nrow=int(np.sqrt(num_samples)), padding=2, normalize=True)

    # Convert grid to numpy for plotting
    grid_np = grid.numpy().transpose((1, 2, 0))

    # Plot the grid of images
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_np)
    plt.axis('off')
    plt.savefig('sample_grid.png', bbox_inches='tight', pad_inches=0)

class FLClient(nn.Module):
    """ Client of Federated Learning framework.
        1. Receive global model from server
        2. Perform local training (compute gradients)
        3. Return local model (gradients) to server
    """
    def __init__(self, model, output_size, data, lr, E, batch_size, q, clip, sigma, device=None):
        """
        :param model: ML model's training process should be implemented
        :param data: (tuple) dataset, all data in client side is used as training data
        :param lr: learning rate
        :param E: epoch of local update
        """
        super(FLClient, self).__init__()
        self.device = device
        self.BATCH_SIZE = batch_size
        self.torch_dataset = TensorDataset(torch.tensor(data[0]),
                                           torch.tensor(data[1]))
        self.data_size = len(self.torch_dataset)
        self.data_loader = DataLoader(
            dataset=self.torch_dataset,
            batch_size=self.BATCH_SIZE,
            shuffle=True
        )
        self.sigma = sigma    # DP noise level
        self.lr = lr
        self.E = E
        self.clip = clip
        self.q = q
        if model == 'scatter':
            self.model = ScatterLinear(81, (7, 7), input_norm="GroupNorm", num_groups=27).to(self.device)
        elif model == VAE:
            self.model = model().to(self.device)
        elif model == nf.core.NormalizingFlow:
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
            q0 = torch.distributions.MultivariateNormal(torch.zeros(n_bottleneck, device=self.device),
                                                        torch.eye(n_bottleneck, device=self.device))
            q0 = nf.distributions.DiagGaussian([n_dims])
            self.model = nf.NormalizingFlow(q0=q0, flows=flows)

        else:
            self.model = model(data[0].shape[1], output_size).to(self.device)

    def recv(self, model_param):
        """receive global model from aggregator (server)"""
        self.model.load_state_dict(copy.deepcopy(model_param))

    def update(self):
        """local model update"""
        self.model.train()
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        # optimizer = torch.optim.Adam(self.model.parameters())
        
        for e in range(self.E):
            # randomly select q fraction samples from data
            # according to the privacy analysis of moments accountant
            # training "Lots" are sampled by poisson sampling
            idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]

            sampled_dataset = TensorDataset(self.torch_dataset[idx][0], self.torch_dataset[idx][1])
            sample_data_loader = DataLoader(
                dataset=sampled_dataset,
                batch_size=self.BATCH_SIZE,
                shuffle=True
            )
            
            optimizer.zero_grad()

            clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
            for batch_x, batch_y in sample_data_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                pred_y = self.model(batch_x.float())
                loss = criterion(pred_y, batch_y.long())
                
                # bound l2 sensitivity (gradient clipping)
                # clip each of the gradient in the "Lot"
                for i in range(loss.size()[0]):
                    loss[i].backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.clip)
                    for name, param in self.model.named_parameters():
                        clipped_grads[name] += param.grad 
                    self.model.zero_grad()
                    
            # add Gaussian noise
            for name, param in self.model.named_parameters():
                clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, self.clip, self.sigma, device=self.device)
                
            # scale back
            for name, param in self.model.named_parameters():
                clipped_grads[name] /= (self.data_size*self.q)
            
            for name, param in self.model.named_parameters():
                param.grad = clipped_grads[name]
            
            # update local model
            optimizer.step()

    def train_gan(self,generator, discriminator, data_loader, num_epochs=10, z_dim=100):
        criterion = nn.BCELoss()
        g_optimizer = torch.optim.SGD(generator.parameters(), lr=0.01, momentum=0.9)
        d_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.01, momentum=0.9)
        
        for epoch in range(num_epochs):
            for real_images, _ in data_loader:
                real_images = real_images.to(self.device)

                # Train discriminator
                d_optimizer.zero_grad()
                
                z = torch.randn(real_images.size(0), z_dim).to(self.device)
                fake_images = generator(z)
                
                real_outputs = discriminator(real_images)
                fake_outputs = discriminator(fake_images.detach())
                
                d_loss_real = criterion(real_outputs, torch.ones_like(real_outputs).to(self.device))
                d_loss_fake = criterion(fake_outputs, torch.zeros_like(fake_outputs).to(self.device))
                
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                d_optimizer.step()
                
                # Train generator
                g_optimizer.zero_grad()
                
                fake_outputs = discriminator(fake_images)
                g_loss = criterion(fake_outputs, torch.ones_like(fake_outputs).to(self.device))
                
                g_loss.backward()
                g_optimizer.step()
    def train_vae(self):
        # Training loop
        criterion = vae_loss
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr,momentum=0.9)

        for e in range(self.E):
            idx = np.where(np.random.rand(len(self.torch_dataset[:][0])) < self.q)[0]
            sampled_dataset = TensorDataset(self.torch_dataset[idx][0], self.torch_dataset[idx][1])
            sample_data_loader = DataLoader(dataset=sampled_dataset, batch_size=self.BATCH_SIZE, shuffle=True)

            optimizer.zero_grad()
            clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
            pbar = tqdm(sample_data_loader, desc=f"Epoch {e+1}/{self.E}")

            #for batch_x, _ in sample_data_loader:
            for batch_x, _ in pbar:
                batch_x = batch_x.to(self.device)
                recon_batch, mu, log_var = self.model(batch_x.float())
                loss = criterion(recon_batch, batch_x.view(-1,784).float(), mu, log_var)

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

            optimizer.step()        

class FLServer(nn.Module):
    """ Server of Federated Learning
        1. Receive model (or gradients) from clients
        2. Aggregate local models (or gradients)
        3. Compute global model, broadcast global model to clients
    """
    def __init__(self, fl_param):
        super(FLServer, self).__init__()
        self.device = fl_param['device']
        self.client_num = fl_param['client_num']
        self.C = fl_param['C']      # (float) C in [0, 1]
        self.clip = fl_param['clip']
        self.T = fl_param['tot_T']  # total number of global iterations (communication rounds)

        self.data = []
        self.target = []
        for sample in fl_param['data'][self.client_num:]:
            self.data += [torch.tensor(sample[0]).to(self.device)]    # test set
            self.target += [torch.tensor(sample[1]).to(self.device)]  # target label

        self.input_size = int(self.data[0].shape[1])
        self.lr = fl_param['lr']
        
        # compute noise using moments accountant
        # self.sigma = compute_noise(1, fl_param['q'], fl_param['eps'], fl_param['E']*fl_param['tot_T'], fl_param['delta'], 1e-5)
        
        # calibration with subsampeld Gaussian mechanism under composition 
        self.sigma = calibrating_sampled_gaussian(fl_param['q'], fl_param['eps'], fl_param['delta'], iters=fl_param['E']*fl_param['tot_T'], err=1e-3)
        print("noise scale =", self.sigma)
        
        self.clients = [FLClient(fl_param['model'],
                                 fl_param['output_size'],
                                 fl_param['data'][i],
                                 fl_param['lr'],
                                 fl_param['E'],
                                 fl_param['batch_size'],
                                 fl_param['q'],
                                 fl_param['clip'],
                                 self.sigma,
                                 self.device)
                        for i in range(self.client_num)]
        
        if fl_param['model'] == 'scatter':
            self.global_model = ScatterLinear(81, (7, 7), input_norm="GroupNorm", num_groups=27).to(self.device)
        elif fl_param['model'] == VAE:
            self.global_model = fl_param['model']().to(self.device)
        elif fl_param['model'] == nf.core.NormalizingFlow:
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
            q0 = torch.distributions.MultivariateNormal(torch.zeros(n_bottleneck, device=self.device),
                                                        torch.eye(n_bottleneck, device=self.device))
            q0 = nf.distributions.DiagGaussian([n_dims])
            self.global_model = nf.NormalizingFlow(q0=q0, flows=flows)

        else:
            self.global_model = fl_param['model'](self.input_size, fl_param['output_size']).to(self.device)
        
        self.weight = np.array([client.data_size * 1.0 for client in self.clients])
        self.broadcast(self.global_model.state_dict())

    def aggregated(self, idxs_users):
        """FedAvg"""
        model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
        new_par = copy.deepcopy(model_par[0])
        for name in new_par:
            new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
        for idx, par in enumerate(model_par):
            w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
            for name in new_par:
                # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
                new_par[name] += par[name] * (w / self.C)
        self.global_model.load_state_dict(copy.deepcopy(new_par))
        return self.global_model.state_dict().copy()

    def broadcast(self, new_par):
        """Send aggregated model to all clients"""
        for client in self.clients:
            client.recv(new_par.copy())

    def test_acc(self):
        self.global_model.eval()
        correct = 0
        tot_sample = 0
        for i in range(len(self.data)):
            t_pred_y = self.global_model(self.data[i])
            _, predicted = torch.max(t_pred_y, 1)
            correct += (predicted == self.target[i]).sum().item()
            tot_sample += self.target[i].size(0)
        acc = correct / tot_sample
        return acc

    def global_update(self):
        # idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
        idxs_users = np.sort(np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False))
        for idx in idxs_users:
            if type(self.global_model) == MNIST_VAE or type(self.global_model) == VAE:
                self.clients[idx].train_vae()
            else:
                self.clients[idx].update()
        self.broadcast(self.aggregated(idxs_users))
        #acc = self.test_acc()
        torch.cuda.empty_cache()
        #return acc

    def set_lr(self, lr):
        for c in self.clients:
            c.lr = lr

