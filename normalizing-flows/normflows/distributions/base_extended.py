from torch import Tensor
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
from torch.distributions import Distribution
from botorch.utils.probability import TruncatedMultivariateNormal
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
    





import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from torch.distributions import Normal, MultivariateNormal, StudentT, Categorical
import torch
import torch.nn as nn
from torch.distributions import Normal
import numpy as np

# class TruncatedDiagGaussian2(Normal):
#     """
#     Truncated multivariate Gaussian distribution with diagonal covariance matrix
#     """

#     def __init__(self, shape, a, b, trainable=True):
#         """
#         Args:
#           shape: Tuple with shape of data, if int shape has one dimension
#           a: Lower bound of the truncation interval
#           b: Upper bound of the truncation interval
#           trainable: Flag whether to use trainable or fixed parameters
#         """
#         if isinstance(shape, int):
#             shape = (shape,)
#         if isinstance(shape, list):
#             shape = tuple(shape)
#         self.shape = shape
#         self.n_dim = len(shape)
#         self.d = np.prod(shape)
#         self.a = a
#         self.b = b
#         if trainable:
#             self.loc = nn.Parameter(torch.zeros(1, *self.shape))
#             self.scale = nn.Parameter(torch.ones(1, *self.shape))
#         else:
#             self.loc = torch.zeros(1, *self.shape)
#             self.scale =  torch.ones(1, *self.shape)
#         super().__init__(self.loc, self.scale)
#         self.temperature = None  # Temperature parameter for annealed sampling

#     def sample(self, num_samples=1, context=None):
#         samples = super().sample((num_samples,))
#         # Truncate samples
#         samples = samples.clamp(min=self.a, max=self.b)
#         return samples
    
#     def log_prob(self, value):
#         # Truncate value
#         value = value.clamp(min=self.a, max=self.b)
#         # Compute unnormalized log prob
#         log_p = super().log_prob(value)
#         # Compute normalization constant
#         cdf_a = super().cdf(torch.tensor(self.a).float())
#         cdf_b = super().cdf(torch.tensor(self.b).float())
#         normalization_constant = torch.log(cdf_b - cdf_a)
#         # Normalize log prob
#         log_p = log_p - normalization_constant
#         return log_p

#     def forward(self, num_samples=1, context=None):
#         z = self.sample(num_samples, context)
#         log_p = self.log_prob(z)
#         return z, log_p
        
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
        z = StudentT(self.df, self.loc, torch.exp(self.log_scale)).sample((num_samples,))
        log_p = StudentT(self.df, self.loc, torch.exp(self.log_scale)).log_prob(z).sum(dim=1)
        return z.squeeze(), log_p

    def log_prob(self, z):
        # Compute Student's T log probability
        log_p = StudentT(self.df, self.loc, torch.exp(self.log_scale)).log_prob(z).sum(dim=1)
        return log_p
    


import math
from numbers import Number

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        'a': constraints.real,
        'b': constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError('Truncation bounds types are different')
        if any((self.a >= self.b).view(-1,).tolist()):
            raise ValueError('Incorrect truncation range')
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * little_phi_coeff_b - self._little_phi_a * little_phi_coeff_a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    @property
    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x ** 2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value ** 2) * 0.5

    def sample(self, sample_shape=None):
        sample_shape = [sample_shape] if isinstance(sample_shape, Number) else sample_shape
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(p)

    def __call__(self, num_samples):
        z = self.sample(num_samples)
        lp = self.log_prob(z)
        return z, lp

class TruncatedNormal(TruncatedStandardNormal):
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self,dim, loc, scale, a, b,trainable = False,device='cuda', validate_args=None):

        self.a = a
        self.b = b        
        self.dim = dim
        self.device = device
        if trainable:
            self.loc = nn.Parameter(torch.zeros((dim,), device=self.device)) + loc
            self.scale = nn.Parameter(torch.ones(self.dim, device=self.device)) * scale
        else:
            self.loc = torch.zeros((dim,), device=self.device) + loc
            self.scale = torch.ones(self.dim, device=self.device) * scale
        a = torch.zeros((dim,), device=self.device) + a
        b = torch.zeros((dim,), device=self.device) + b

        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)

        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super(TruncatedNormal, self).__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale ** 2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super(TruncatedNormal, self).cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(super(TruncatedNormal, self).icdf(value))

    def log_prob(self, value):
        _lp = super(TruncatedNormal, self).log_prob(self._to_std_rv(value)) - self._log_scale
        return _lp.sum(1)



# class TruncatedGaussian(TruncatedMultivariateNormal):
#     """
#     Truncated Multivariate Gaussian Distribution

#     Parameters:
#     shape (tuple): Shape of data
#     bounds (Tensor): A `batch_shape x event_shape x 2` tensor of strictly increasing
#                      bounds for `x` so that `bounds[..., 0] < bounds[..., 1]` everywhere.
#     trainable (bool, optional): If True, parameters will be optimized during training. Default is True.
#     """

#     def __init__(self, ndim, loc,scale, a,b, trainable=True, device='cuda'):

#         self.n_dim = ndim
#         self.device = device
#         self.a = a  
#         self.b = b
#         self.bounds = torch.stack([torch.zeros(ndim)+a, torch.zeros(ndim) + b], dim=-1)

#         # Create loc and scale parameters or buffers
#         if trainable:
#             self.loc = nn.Parameter(torch.zeros(ndim),requires_grad=True) + loc
#             self.cov_mat = nn.Parameter(torch.eye(ndim),requires_grad=True) * scale
#         else:
#             self.loc = torch.zeros(ndim) +  loc
#             self.cov_mat = torch.eye(ndim) * scale
#         # with torch.no_grad():
#         #     self.loc = self.loc.cuda()
#         #     self.cov_mat = self.cov_mat.cuda()
#         #     self.bounds = self.bounds.cuda()

#         # Create the covariance matrix from the scale
        

#         # Initialize the parent class
#         super().__init__(self.loc, self.cov_mat, bounds=self.bounds)

#     def __call__(self, num_samples=1):
#         z = self.rsample(torch.Size([num_samples]))
#         log_p = self.log_prob(z)
#         return z.squeeze().to(self.device), log_p.to(self.device)
    
#     def sample(self, num_samples=1):
#         return self.rsample(torch.Size([num_samples])).to(self.device)
