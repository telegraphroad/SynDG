# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import numpy as np
import normflows as nf
import torch
import torch.nn as nn
from normflows.flows.variational_dequantization import MAF, VariationalDequantization, ShiftScaleFlow

hidden_dim = 64
num_samples = 10000
num_cat = torch.tensor(13)
data_dim = 1

# Gaussian parameters
mean = num_cat.float() / 2.0  
std_dev = num_cat.float() / 9.0  

# Generate Gaussian distributed data
data = torch.normal(mean=mean, std=std_dev, size=(num_samples, data_dim))

# Convert to integer values and clip to the range [0, num_cat)
data = torch.clamp(data.round(), 0, num_cat-1).long()

flows = []
flows.append(ShiftScaleFlow(1,32))
flows.append(ShiftScaleFlow(1,32))

dequant = VariationalDequantization(var_flows=flows, num_cat=num_cat)


epoch = 0
num_epochs = 10000
dequant.train()
optimizer = torch.optim.Adam(dequant.parameters(), lr=0.001)

for epoch in range(num_epochs):
    optimizer.zero_grad()

    z, ldj = dequant(data, torch.zeros(data.shape[0]), reverse=False)
    log_prob = -0.5 * z.pow(2)  # Prior is standard normal
    loss = -(log_prob + ldj).mean()

    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

        with torch.no_grad():
            z, _ = dequant(data, torch.zeros(data.shape[0]))
            import matplotlib.pyplot as plt
            z,ldj = dequant(data,torch.zeros(data.shape[0]), reverse=False)
            z,_ = dequant(z,ldj, reverse=True)
            print(data.unique(),z.unique())
            plt.hist(data.detach().numpy(), bins=100, density=True, color='blue')
            plt.title(f"Epoch {epoch}")
            plt.show()
            plt.hist(z.detach().numpy(), bins=100, density=True, color='red')
            #set plot title to epoch
            plt.title(f"Epoch {epoch}")
            plt.show()

# %%
z,ldj = dequant(data,torch.zeros(data.shape[0]), reverse=False)
z,_ = dequant(z,torch.zeros(data.shape[0]), reverse=True)
print(data.unique(),z.unique())
plt.hist(data.detach().numpy(), bins=100, density=True, color='blue')
plt.title(f"Epoch {epoch}")
plt.show()
plt.hist(z.detach().numpy(), bins=100, density=True, color='red')
#set plot title to epoch
plt.title(f"Epoch {epoch}")
plt.show()

# %%
