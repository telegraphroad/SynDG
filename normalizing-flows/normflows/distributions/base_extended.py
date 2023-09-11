from .base import *

class GeneralizedGaussianMixture(BaseDistribution):
    """
    Mixture of generalized Gaussians with diagonal covariance matrix
    """

    def __init__(
        self, n_modes, dim, loc=None, scale=None, shape=None, 
        weights=None, trainable=True
    ):
        """Constructor

        Args:
          n_modes: Number of modes of the mixture model  
          dim: Number of dimensions of each Gaussian
          loc: List of mean values  
          scale: List of diagonals of the covariance matrices
          shape: List of shape parameters of the generalized Gaussians
          weights: List of mode probabilities
          trainable: Flag, if true parameters will be optimized during training
        """
        super().__init__()

        self.n_modes = n_modes
        self.dim = dim

        if loc is None:
            loc = np.random.randn(self.n_modes, self.dim)
        loc = np.array(loc)[None, ...]
        if scale is None:
            scale = np.ones((self.n_modes, self.dim)) 
        scale = np.array(scale)[None, ...]
        if shape is None:
            shape = np.ones((self.n_modes, self.dim))
        shape = np.array(shape)[None, ...]        
        if weights is None:
            weights = np.ones(self.n_modes)
        weights = np.array(weights)[None, ...]
        weights /= weights.sum(1)

        if trainable:
            self.loc = nn.Parameter(torch.tensor(1.0 * loc))
            self.log_scale = nn.Parameter(torch.tensor(np.log(1.0 * scale)))
            self.shape = nn.Parameter(torch.tensor(1.0 * shape)) 
            self.weight_scores = nn.Parameter(torch.tensor(np.log(1.0 * weights)))
        else:
            self.register_buffer("loc", torch.tensor(1.0 * loc))
            self.register_buffer("log_scale", torch.tensor(np.log(1.0 * scale)))
            self.register_buffer("shape", torch.tensor(1.0 * shape))
            self.register_buffer("weight_scores", torch.tensor(np.log(1.0 * weights)))

    def _generalized_gaussian(eps, shape, scale, loc):
        const = torch.exp(torch.lgamma(3/shape)/shape)
        coeff = (shape/scale)**shape
        z = const * eps * coeff + loc
        return z

    def _generalized_gaussian_logprob(x, shape, loc, scale):
        const = torch.lgamma(3/shape)/shape
        coeff = (shape/scale)**shape
        term1 = -const 
        term2 = -torch.log(scale)
        term3 = -(torch.abs(x - loc)/scale)**shape
        return term1 + term2 + term3

    def forward(self, num_samples=1):
        # Get weights
        weights = torch.softmax(self.weight_scores, 1)
        
        # Sample mode indices        
        mode = torch.multinomial(weights[0, :], num_samples, replacement=True)
        mode_1h = nn.functional.one_hot(mode, self.n_modes)
        mode_1h = mode_1h[..., None]
        
        # Get samples
        eps_ = torch.randn(num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device)
        scale_sample = torch.sum(torch.exp(self.log_scale) * mode_1h, 1)
        shape_sample = torch.sum(self.shape * mode_1h, 1)
        loc_sample = torch.sum(self.loc * mode_1h, 1)        
        z = self._generalized_gaussian(eps_, shape_sample, scale_sample, loc_sample)

        # Compute log probability
        log_p = self.log_prob(z)
        
        return z, log_p

    def log_prob(self, z):
        # Get weights  
        weights = torch.softmax(self.weight_scores, 1)
        
        # Compute log probability
        log_p = self._generalized_gaussian_logprob(z[:, None, :], self.shape, 
                        self.loc, torch.exp(self.log_scale))
        log_p += torch.log(weights)
        log_p = torch.logsumexp(log_p, 1)
        
        return log_p

    def sample(self, sample_shape=torch.Size()):
        """
        Sample from the generalized Gaussian mixture model

        Args:
        sample_shape: Shape of the samples  
        """
        with torch.no_grad():
            num_samples = sample_shape[0] if sample_shape else 1

        # Sample mode indices  
        weights = torch.softmax(self.weight_scores, 1)
        mode = torch.multinomial(weights[0, :], num_samples, replacement=True)
        mode_1h = nn.functional.one_hot(mode, self.n_modes)
        mode_1h = mode_1h[..., None]

        # Get parameters  
        scale_sample = torch.sum(torch.exp(self.log_scale) * mode_1h, 1)
        shape_sample = torch.sum(self.shape * mode_1h, 1)
        loc_sample = torch.sum(self.loc * mode_1h, 1)        

        # Sample from the distribution  
        eps = torch.randn(sample_shape, dtype=self.loc.dtype, device=self.loc.device)
        z = self._generalized_gaussian(eps, shape_sample, scale_sample, loc_sample)
        
        return z    

