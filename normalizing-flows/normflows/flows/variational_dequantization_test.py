import unittest
import torch
import numpy as np
from normflows.flows.variational_dequantization import ShiftScaleFlow, VariationalDequantization
from scipy.stats import entropy

class DequantizerTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.hidden_dim = 64
        cls.num_samples = 10000
        cls.num_cat = torch.tensor(13)
        cls.data_dim = 1

        # Gaussian parameters
        cls.mean = cls.num_cat.float() / 2.0  
        cls.std_dev = cls.num_cat.float() / 9.0  

        # Generate Gaussian distributed data
        cls.data = torch.normal(mean=cls.mean, std=cls.std_dev, size=(cls.num_samples, cls.data_dim))

        # Convert to integer values and clip to the range [0, num_cat)
        cls.data = torch.clamp(cls.data.round(), 0, cls.num_cat-1).long()

        flows = [ShiftScaleFlow(1,32), ShiftScaleFlow(1,32)]
        cls.dequant = VariationalDequantization(var_flows=flows, num_cat=cls.num_cat)
        
        cls.optimizer = torch.optim.Adam(cls.dequant.parameters(), lr=0.001)
        
        # Training
        for epoch in range(100):
            cls.optimizer.zero_grad()

            z, ldj = cls.dequant(cls.data, torch.zeros(cls.data.shape[0]), reverse=False)
            log_prob = -0.5 * z.pow(2)  # Prior is standard normal
            loss = -(log_prob + ldj).mean()

            loss.backward()
            cls.optimizer.step()

    def test_histogram_similarity(self):
        with torch.no_grad():
            z, _ = self.dequant(self.data, torch.zeros(self.data.shape[0]))
            z, ldj = self.dequant(self.data, torch.zeros(self.data.shape[0]), reverse=False)
            z, _ = self.dequant(z, ldj, reverse=True)
        
        # Compute histograms
        data_hist, _ = np.histogram(self.data.detach().numpy(), bins=100, density=True)
        z_hist, _ = np.histogram(z.detach().numpy(), bins=100, density=True)

        # Replace zeros to avoid division by zero
        data_hist += np.finfo(float).eps
        z_hist += np.finfo(float).eps

        # Compute KL divergence
        kl_divergence = entropy(data_hist, z_hist)
        # Test if KL divergence is below a threshold
        self.assertLess(kl_divergence, 1e-4)  # adjust the threshold as needed

if __name__ == '__main__':
    unittest.main()            
if __name__ == '__main__':
    unittest.main()
# %%
