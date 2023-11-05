# Machine learning models
import torch
from torch import nn
from kymatio.torch import Scattering2D
from torch.nn import functional as F
import normflows as nf

class MNIST_VAE(nn.Module):
    """
    VAE model for MNIST and Fashion-MNIST, with Tanh activations.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=128, latent_dim=10):
        super(MNIST_VAE, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, hidden_dim),
            nn.Tanh(),
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 32 * 7 * 7),
            nn.Tanh(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mu, log_var
    
    def sample(self, num_samples):
        """
        This function generates new images by sampling from the latent space.
        """
        z = torch.randn(num_samples, self.fc_mu.out_features).to(self.fc_mu.weight.device)
        samples = self.decoder(z)
        return samples


class VAE(nn.Module):
    def __init__(self, x_dim=784, h_dim1=512, h_dim2=256, z_dim=2):
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

def vae_loss(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='none')
    BCE = BCE.sum(1)  # Sum the BCE loss over the image dimensions
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    return BCE + KLD


class MNIST_GAN_Generator(nn.Module):
    """
    Generator model for MNIST and Fashion-MNIST, with Tanh activations. 
    """
    def __init__(self, input_dim, output_dim):
        super(MNIST_GAN_Generator, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 4 * 4 * 32),
            nn.Tanh()
        )
        
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=0),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, kernel_size=8, stride=2, padding=2),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 32, 4, 4)
        x = self.layer1(x)
        return x
    

class MNIST_GAN_Discriminator(nn.Module):
    """
    Discriminator model for MNIST and Fashion-MNIST, with Tanh activations.
    """
    def __init__(self):
        super(MNIST_GAN_Discriminator, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        
        self.fc = nn.Sequential(
            nn.Linear(4 * 4 * 32, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

class MNIST_CNN(nn.Module):
    """
    End-to-end CNN model for MNIST and Fashion-MNIST, with Tanh activations. 
    References:
    - Papernot, Nicolas, et al. Tempered Sigmoid Activations for Deep Learning with Differential Privacy. In AAAI 2021.
    - Tramer, Florian, and Dan Boneh. Differentially Private Learning Needs Better Features (or Much More Data). In ICLR 2021. 
    """
    def __init__(self, input_dim, output_dim):
        super(MNIST_CNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=2, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        
        self.fc = nn.Sequential(nn.Linear(4 * 4 * 32, 32),
                                        nn.Tanh(),
                                        nn.Linear(32, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
def get_scatter_transform():
    shape = (28, 28, 1)
    scattering = Scattering2D(J=2, shape=shape[:2])
    K = 81 * shape[2]
    (h, w) = shape[:2]
    return scattering, K, (h//4, w//4)


class ScatterLinear(nn.Module):
    """
    ScatterNet model used in the following paper
    - Tramer, Florian, and Dan Boneh. Differentially Private Learning Needs Better Features (or Much More Data). In ICLR 2021. 
    See https://github.com/ftramer/Handcrafted-DP/blob/main/models.py
    """
    def __init__(self, in_channels, hw_dims, input_norm=None, classes=10, clip_norm=None, **kwargs):
        super(ScatterLinear, self).__init__()
        self.K = in_channels
        self.h = hw_dims[0]
        self.w = hw_dims[1]
        self.fc = None
        self.norm = None
        self.clip = None
        self.build(input_norm, classes=classes, clip_norm=clip_norm, **kwargs)

    def build(self, input_norm=None, num_groups=None, bn_stats=None, clip_norm=None, classes=10):
        self.fc = nn.Linear(self.K * self.h * self.w, classes)

        if input_norm is None:
            self.norm = nn.Identity()
        elif input_norm == "GroupNorm":
            self.norm = nn.GroupNorm(num_groups, self.K, affine=False)
        else:
            self.norm = lambda x: standardize(x, bn_stats)

        if clip_norm is None:
            self.clip = nn.Identity()
        else:
            self.clip = ClipLayer(clip_norm)

    def forward(self, x):
        x = self.norm(x.view(-1, self.K, self.h, self.w))
        x = self.clip(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    
class LogisticRegression(nn.Module):
    """Logistic regression"""
    def __init__(self, num_feature, output_size):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(num_feature, output_size)

    def forward(self, x):
        return self.linear(x)

    
class MLP(nn.Module):
    """Neural Networks"""
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.Tanh(),

            nn.Linear(1000, output_dim))

    def forward(self, x):
        return self.model(x)
    
    
class three_layer_MLP(nn.Module):
    """Neural Networks"""
    def __init__(self, input_dim, output_dim):
        super(three_layer_MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 600),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(600, 300),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(300, 100),
            nn.Dropout(0.2),
            nn.ReLU(),
            
            nn.Linear(100, output_dim))

    def forward(self, x):
        return self.model(x)
    

class MnistCNN_(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MnistCNN_, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2))
        
        self.fc = nn.Linear(7 * 7 * 32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    
