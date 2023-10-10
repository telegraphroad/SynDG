from .base import *
import numpy as np
from typing import *
from numbers import Number
from scipy import stats

import torch
import torch.nn as nn
from torch.distributions import StudentT, MultivariateNormal, Categorical,MixtureSameFamily
from torch.distributions import constraints, Distribution
from torch.distributions.utils import broadcast_all

class GeneralizedGaussianDistribution(Distribution):
    """
    Generalized Gaussian Distribution

    Parameters:
    loc (Tensor): the location parameter of the distribution.
    scale (Tensor): the scale parameter of the distribution.
    p (Tensor): the shape parameter of the distribution.
    validate_args (bool, optional): checks if the arguments are valid. Default is None.

    """
    
    arg_constraints = {'loc': constraints.real, 'scale': constraints.positive, 'p': constraints.positive}
    support = constraints.real

    def __init__(self, loc, scale, p, validate_args=None):
        # Broadcast all inputs to ensure they have the same shape
        self.loc, self.scale = broadcast_all(loc, scale)
        (self.p,) = broadcast_all(p)
        
        # Convert tensor to numpy for scipy compatibility
        self.scipy_dist = stats.gennorm(loc=self.loc.cpu().detach().numpy(),
                            scale=self.scale.cpu().detach().numpy(),
                            beta=self.p.cpu().detach().numpy())
        
        # Determine batch shape based on input types
        if isinstance(loc, Number) and isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
            
        # Initialize the base class
        super(GeneralizedGaussianDistribution, self).__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self):
        # Mean of the distribution is its location parameter
        return self.loc

    @property
    def variance(self):
        # Variance is calculated using the scale and shape parameters
        return self.scale.pow(2) * (torch.lgamma(3/self.p) - torch.lgamma(1/self.p)).exp()

    @property
    def stddev(self):
        # Standard deviation is the square root of variance
        return self.variance**0.5

    def expand(self, batch_shape, _instance=None):
        # Create a new instance of the distribution with expanded batch shape
        new = self._get_checked_instance(GeneralizedGaussianDistribution, _instance)
        batch_shape = torch.Size(batch_shape)
        new.loc = self.loc.expand(batch_shape)
        new.scale = self.scale.expand(batch_shape)
        super(GeneralizedGaussianDistribution, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def sample(self, sample_shape=torch.Size()):
        # Generate a sample from the distribution
        sample_shape = sample_shape + self.loc.size()
        return torch.tensor(self.scipy_dist.rvs(
            list(sample_shape),
            random_state=torch.randint(2**32, ()).item()),  # Make deterministic if torch is seeded
                            dtype=self.loc.dtype, device=self.loc.device)

    def log_prob(self, value):
        # Compute the log probability of a given value
        if self._validate_args:
            self._validate_sample(value)
        return (-torch.log(2 * self.scale) - torch.lgamma(1/self.p) + torch.log(self.p)
                - torch.pow((torch.abs(value - self.loc) / self.scale), self.p))

    def cdf(self, value):
        # Compute the cumulative distribution function at a given value
        if isinstance(value, torch.Tensor):
            value = value.numpy()
        return torch.tensor(self.scipy_dist.cdf(value),
                            dtype=self.loc.dtype, device=self.loc.device)

    def icdf(self, value):
        # Inverse cumulative distribution function is not implemented
        raise NotImplementedError

    def entropy(self):
        # Compute the entropy of the distribution
        return (1/self.p) - torch.log(self.p) + torch.log(2*self.scale) + torch.lgamma(1/self.p)



class GeneralizedGaussianMixture(BaseDistribution):
    """
    Mixture of Generalized Gaussians

    Parameters:
    n_modes (int): Number of modes of the mixture model
    dim (int): Number of dimensions of each Gaussian
    loc (float, optional): Mean values. Default is 0.
    scale (float, optional): Diagonals of the covariance matrices. Default is 1.
    p (float, optional): Shape parameter for the Generalized Gaussian. Default is 2.
    rand_p (bool, optional): If True, shape parameter p is randomized. Default is True.
    noise_scale (float, optional): Scale of the noise added to shape parameter p. Default is 0.01.
    weights (list, optional): List of mode probabilities. Default is None.
    trainable_loc (bool, optional): If True, location parameters will be optimized during training. Default is False.
    trainable_scale (bool, optional): If True, scale parameters will be optimized during training. Default is True.
    trainable_p (bool, optional): If True, shape parameters will be optimized during training. Default is True.
    trainable_weights (bool, optional): If True, weights will be optimized during training. Default is True.
    device (str, optional): Device to which tensors will be moved. Default is 'cuda'.
    """

    def __init__(self, n_modes, dim, loc=0., scale=1., p=2., rand_p=True, noise_scale=0.01, weights=None, trainable_loc=True, trainable_scale=True, trainable_p=True, trainable_weights=True, ds=None, device='cuda'):
        super().__init__()
        with torch.no_grad():
            self.n_modes = n_modes
            self.dim = dim
            self.device = device

            # Initialize location, scale and shape parameters
            if ds is None:
                loc = np.zeros((self.n_modes, self.dim)) + loc
            else:
                loc = np.tile(ds.calculate_feature_means(),(n_modes,1)) + loc
            scale = np.zeros((self.n_modes, self.dim)) + scale
            p = np.zeros((self.n_modes, self.dim)) + p
            if rand_p:
                noise = np.random.normal(0, noise_scale, p.shape)
                p += noise
                loc += noise
                scale += np.abs(noise)
                
            # Initialize weights
            if weights is None:
                weights = np.ones(self.n_modes)
            weights /= np.sum(weights)

            # Create parameters or buffers depending on whether they are trainable or not
            if trainable_loc:
                self.loc = nn.Parameter(torch.tensor(1.0 * loc, device=self.device).float(),requires_grad=True)
            else:
                self.register_buffer("loc", torch.tensor(1.0 * loc, device=self.device).float())
            if trainable_scale:
                self.scale = nn.Parameter(torch.tensor(1.0 * scale, device=self.device).float(),requires_grad=True) 
            else:
                self.register_buffer("scale", torch.tensor(1.0 * scale, device=self.device).float())
            if trainable_p:
                self.p = nn.Parameter(torch.tensor(1.0 * p, device=self.device).float(),requires_grad=True)
            else:
                self.register_buffer("p", torch.tensor(1.0 * p, device=self.device).float())
            if trainable_weights:
                self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights), device=self.device).float(),requires_grad=True)
            else:
                self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights), device=self.device).float())
            # Initialize the underlying Generalized Gaussian and Categorical distributions
            self.gg = torch.distributions.Independent(GeneralizedGaussianDistribution(self.loc, self.scale, self.p),1)
            self.cat = Categorical(torch.softmax(self.weight_scores, 0))

            self.mixture = MixtureSameFamily(self.cat, self.gg)


    def forward(self, num_samples=1):
        # Sample mode indices
        z = self.mixture.sample((num_samples,))

        # Compute log probability
        log_p = self.mixture.log_prob(z)

        return z, log_p
    
    def log_prob(self, z):
        # Compute log probability
        log_p = self.mixture.log_prob(z)
        return log_p
    

# class MultivariateStudentT(BaseDistribution):
#     """
#     Multivariate Student's T-distribution with full covariance matrix

#     Parameters:
#     shape (tuple): Shape of data
#     df (float, optional): Degrees of freedom. Default is 2.0.
#     trainable (bool, optional): If True, parameters will be optimized during training. Default is True.
#     """

#     def __init__(self, shape, df=2.0, trainable=True,device='cuda'):
#         super().__init__()
#         if isinstance(shape, int):
#             shape = (shape,)
#         if isinstance(shape, list):
#             shape = tuple(shape)
#         self.shape = shape
#         self.n_dim = len(shape)
#         self.d = np.prod(shape)
#         self.device = device    
#         # Create parameters or buffers depending on whether they are trainable or not
#         if trainable:
#             self.loc = nn.Parameter(torch.zeros(1, *self.shape,device=self.device))
#             self.log_scale = nn.Parameter(torch.zeros(1, *self.shape, *self.shape,device=self.device))
#             self.df = nn.Parameter(torch.tensor(df,device=self.device))  # degrees of freedom
#         else:
#             self.register_buffer("loc", torch.zeros(1, *self.shape,device=self.device))
#             self.register_buffer("log_scale", torch.zeros(1, *self.shape, *self.shape,device=self.device))
#             self.register_buffer("df", torch.tensor(df,device=self.device))

#     def forward(self, num_samples=1):
#         # Draw samples from a multivariate normal
#         eps = MultivariateNormal(torch.zeros(self.d,device=self.device), torch.eye(self.d,device=self.device)).sample().view(*self.shape)
        
#         # Scale and shift by loc and scale, then apply Student's T transformation
#         print()
#         z = self.loc + torch.exp(self.log_scale @ eps.unsqueeze(-1)).squeeze(-1)
#         z = z / (StudentT(self.df).rsample() / ((self.df - 2) ** 0.5))

#         # Compute log probability
#         log_p = StudentT(self.df).log_prob(z.view(-1)).sum()

#         return z, log_p

#     def log_prob(self, z):
#         # Compute Student's T log probability
#         log_p = StudentT(self.df).log_prob((z - self.loc).view(-1)).sum()
#         return log_p
    


import torch
import torch.nn as nn
import pyro.distributions as dist

import torch
import torch.nn as nn
import pyro.distributions as dist

import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist

class MultivariateStudentTDist(nn.Module):
    def __init__(self, degree_of_freedom, dim, trainable=True, device='cpu'):
        super().__init__()
        self.dim = dim
        self.device = device

        if trainable:
            self.loc = nn.Parameter(torch.zeros((dim,), device=self.device))
            self.scale_tril = nn.Parameter(torch.eye(self.dim, device=self.device))
            self.df = nn.Parameter(torch.tensor(degree_of_freedom, device=self.device))
        else:
            self.register_buffer("loc", torch.zeros((dim,), device=self.device))
            self.register_buffer("scale_tril", torch.eye(self.dim, device=self.device))
            self.register_buffer("df", torch.tensor(degree_of_freedom, device=self.device))

    def forward(self, num_samples):
        mvt = dist.MultivariateStudentT(self.df,self.loc,self.scale_tril)
        samples = mvt.sample((num_samples,))
        log_prob = self.log_prob(samples)
        return samples, log_prob

    def log_prob(self, samples):
        
        return dist.MultivariateStudentT(self.df,self.loc,self.scale_tril).log_prob(samples)
    
class StudentTDistribution(BaseDistribution):
    """
    Multivariate single-variate Student's T-distribution

    Parameters:
    shape (tuple): Shape of data
    df (float, optional): Degrees of freedom. Default is 2.0.
    trainable (bool, optional): If True, parameters will be optimized during training. Default is True.
    """

    def __init__(self, shape, df=2.0, trainable=True, device='cuda'):
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.device = device

        # Create parameters or buffers depending on whether they are trainable or not
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self.shape, device=self.device))
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape, device=self.device))
            self.df = nn.Parameter(torch.tensor(df,device=self.device))  # degrees of freedom
        else:
            self.register_buffer("loc", torch.zeros(1, *self.shape))
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
            self.register_buffer("df", torch.tensor(df))

    def forward(self, num_samples=1):
        # Draw samples from a normal distribution
        # eps = Normal(torch.zeros_like(self.log_scale), torch.ones_like(self.log_scale)).sample(num_samples)

        # # Scale and shift by loc and scale, then apply Student's T transformation
        # z = self.loc + torch.exp(self.log_scale) * eps
        # z = z / (StudentT(self.df).rsample() / ((self.df - 2) ** 0.5))

        # # Compute log probability
        # log_p = StudentT(self.df, self.loc, torch.exp(self.log_scale)).log_prob(z).sum()
        z = StudentT(self.df, self.loc, torch.exp(self.log_scale)).sample((num_samples,))
        log_p = StudentT(self.df, self.loc, torch.exp(self.log_scale)).log_prob(z).sum(dim=1)
        return z.squeeze(), log_p

    def log_prob(self, z):
        # Compute Student's T log probability
        log_p = StudentT(self.df, self.loc, torch.exp(self.log_scale)).log_prob(z).sum(dim=1)
        return log_p