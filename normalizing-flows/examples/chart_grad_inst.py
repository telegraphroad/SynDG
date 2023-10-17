# Import required packages
import torch
import numpy as np
import normflows as nf

from matplotlib import pyplot as plt
from tqdm import tqdm
from geom_median.torch import compute_geometric_median   # PyTorch API
import pandas as pd

# Set up model

# Define flows
for indf,bddf in zip([10.0],[2.0]):
    K = 8
    torch.manual_seed(0)

    latent_size = 2
    b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
    flows = []
    for i in range(K):
        s = nf.nets.MLP([latent_size, 6 * latent_size, latent_size], init_zeros=True)
        t = nf.nets.MLP([latent_size, 6 * latent_size, latent_size], init_zeros=True)
        if i % 2 == 0:
            flows += [nf.flows.MaskedAffineFlow(b, t, s)]
        else:
            flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        flows += [nf.flows.ActNorm(latent_size)]

    # Set target and q0
    target = nf.distributions.TwoModes(2, 0.1)
    target = nf.distributions.MvStudentT(2,indf,0.,1.)
    q0 = nf.distributions.DiagGaussian(2)
    q0 = nf.distributions.MvStudentT(2,bddf,0.,1.)

    # Construct flow model
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)

    # Move model on GPU if available
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    nfm = nfm.to(device)
    nfm = nfm.double()

    # Initialize ActNorm
    z, _ = nfm.sample(num_samples=2 ** 7)
    z_np = z.to('cpu').data.numpy()
    plt.figure(figsize=(15, 15))
    plt.hist2d(z_np[:, 0].flatten(), z_np[:, 1].flatten(), (200, 200), range=[[-3, 3], [-3, 3]])
    plt.gca().set_aspect('equal', 'box')
    plt.show()

    # Plot target distribution
    grid_size = 200
    xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.double().to(device)
    log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
    prob_target = torch.exp(log_prob)

    # Plot initial posterior distribution
    log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    plt.figure(figsize=(15, 15))
    plt.pcolormesh(xx, yy, prob.data.numpy())
    plt.contour(xx, yy, prob_target.data.numpy(), cmap=plt.get_cmap('cool'), linewidths=2)
    plt.gca().set_aspect('equal', 'box')
    plt.show()


    # Train model
    max_iter =7000
    num_samples = 512
    anneal_iter = 10000
    annealing = True
    show_iter = 5


    loss_hist = np.array([])

    optimizer = torch.optim.Adam(nfm.parameters(), lr=1e-4, weight_decay=1e-6)
    exs = 0
    larr=[]
    marr=[]
    garr=[]
    garrmean=[]
    garrmed=[]
    garrgm=[]
    xmean = []
    xgm = []
    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()
        try:
            x = target.sample(num_samples).to(device)
            #loss = nfm.reverse_alpha_div(num_samples, dreg=True, alpha=1)
            if (it + 1) % show_iter == 0:
                loss,mloss,gloss2 = nfm.forward_kld(x,extended=True)
            else:
                loss = nfm.forward_kld(x,extended=False)
            
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()

                optimizer.step()

            
            loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
            
            # Plot learned posterior
            if (it + 1) % show_iter == 0:
                
                with torch.no_grad():

                    gradient_list = []

                    # Collect gradients
                    for param in nfm.parameters():
                        if param.grad is not None:
                            gradient_list.append(param.grad.view(-1))  # Flattening the gradient tensor

                    # Concatenate the flattened gradients into a single tensor
                    try:
                        flattened_gradients = torch.cat(gradient_list)

                    # Convert the tensor to a NumPy array and then to a flattened list
                        flattened_gradients_list = flattened_gradients.cpu().detach().numpy().tolist()
                        garrmean.append(np.mean(flattened_gradients_list))
                        #garrmed.append(np.median(flattened_gradients_list))
                        garrgm.append(compute_geometric_median(torch.tensor(flattened_gradients_list), weights=None).median.item())
                        larr.append(loss.item())
                    #marr.append(mloss.item())
                        garr.append(gloss2.item())
                        #print('================',torch.mean(x).mean())
                        xmean.append(torch.mean(x).mean().item())
                        _g = compute_geometric_median(x, weights=None).median
                        xgm.append(compute_geometric_median(_g, weights=None).median.item())
                        print('shapes',np.shape(larr),np.shape(garr),np.shape(garrmean),np.shape(garrgm))
                        df = pd.DataFrame({'loss':larr,'gloss':garr,'garrmean':garrmean,'garrgm':garrgm,'xmean':xmean,'xgm':xgm})
                        df.to_csv(f'losses_{indf}_{bddf}.csv')


                    except Exception as e:
                        print(e)


                log_prob = nfm.log_prob(zz).to('cpu').view(*xx.shape)
                prob = torch.exp(log_prob)
                prob[torch.isnan(prob)] = 0


        except Exception as e:
            exs+=1
        
    print(exs)
    print('shapes',np.shape(larr),np.shape(garr),np.shape(garrmean),np.shape(garrgm))
    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label='loss')
    plt.legend()
    plt.show()

    W=40
    import pandas as pd
    plt.figure(figsize=(10, 10))
    rolling_mean = pd.Series(larr).rolling(window=W).mean()
    plt.plot(rolling_mean, label='loss mean')
    rolling_mean = pd.Series(marr).rolling(window=W).mean()
    plt.plot(rolling_mean, label='loss median')
    rolling_mean = pd.Series(garr).rolling(window=W).mean()

    plt.plot(rolling_mean, label='loss gmed')
    plt.legend()
    plt.show()
    
    df = pd.DataFrame({'loss':larr,'gloss':garr,'garrmean':garrmean,'garrgm':garrgm,'xmean':xmean,'xgm':xgm})
    df.to_csv(f'losses_{indf}_{bddf}.csv')


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_style("darkgrid")


plt.figure(figsize=(10, 10))
W=40
a = pd.read_csv('losses_2.0_10.0.csv')
rolling_mean = pd.Series(np.clip(a['garrmean'],0.,0.01))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Mean,2.0->10.0',c='olivedrab',linestyle='dotted')
rolling_mean = pd.Series(np.clip(a['garrgm'],0.,0.01))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Geometric median,2.0->10.0',c='olivedrab',linestyle='solid')

a = pd.read_csv('losses_2.0_2.0.csv')
rolling_mean = pd.Series(np.clip(a['garrmean'],0.,0.01))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Mean,2.0->2.0',c='orangered',linestyle='dotted')
rolling_mean = pd.Series(np.clip(a['garrgm'],0.,0.01))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Geometric median,2.0->2.0',c='orangered',linestyle='solid')

a = pd.read_csv('losses_10.0_10.0.csv')
rolling_mean = pd.Series(np.clip(a['garrmean'],0.,0.01))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Mean,10.0->10.0',c='royalblue',linestyle='dotted')
rolling_mean = pd.Series(np.clip(a['garrgm'],0.,0.01))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Geometric median,10.0->10.0',c='royalblue',linestyle='solid')

a = pd.read_csv('losses_10.0_2.0.csv')
rolling_mean = pd.Series(np.clip(a['garrmean'],0.,0.01))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Mean,10.0->2.0',c='darkmagenta',linestyle='dotted')
rolling_mean = pd.Series(np.clip(a['garrgm'],0.,0.01))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Geometric median,10.0->2.0',c='darkmagenta',linestyle='solid')

plt.xlabel('Iteration')
plt.ylabel(r'$\nabla_\phi \mathcal{L}(\theta)$')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 10))
W=40
a = pd.read_csv('losses_2.0_10.0.csv')
rolling_mean = pd.Series(np.clip(a['loss'],0.,5.))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Mean,2.0->10.0',c='olivedrab',linestyle='dotted')
rolling_mean = pd.Series(np.clip(a['gloss'],0.,5.))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Geometric median,2.0->10.0',c='olivedrab',linestyle='solid')

a = pd.read_csv('losses_2.0_2.0.csv')
rolling_mean = pd.Series(np.clip(a['loss'],0.,5.))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Mean,2.0->2.0',c='orangered',linestyle='dotted')
rolling_mean = pd.Series(np.clip(a['gloss'],0.,5.))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Geometric median,2.0->2.0',c='orangered',linestyle='solid')

a = pd.read_csv('losses_10.0_10.0.csv')
rolling_mean = pd.Series(np.clip(a['loss'],0.,5.))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Mean,10.0->10.0',c='royalblue',linestyle='dotted')
rolling_mean = pd.Series(np.clip(a['gloss'],0.,5.))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Geometric median,10.0->10.0',c='royalblue',linestyle='solid')

a = pd.read_csv('losses_10.0_2.0.csv')
rolling_mean = pd.Series(np.clip(a['loss'],0.,5.))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Mean,10.0->2.0',c='darkmagenta',linestyle='dotted')
rolling_mean = pd.Series(np.clip(a['gloss'],0.,5.))[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Geometric median,10.0->2.0',c='darkmagenta',linestyle='solid')

plt.xlabel('Iteration')
plt.ylabel(r'$\mathcal{L}(\theta)$')
plt.legend()
plt.grid(True)

plt.figure(figsize=(10, 10))
W=30

a = pd.read_csv('losses_2.0_10.0.csv')
rolling_mean = pd.Series(a['xmean'])[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Mean,2.0',c='olivedrab',linestyle='dotted')
rolling_mean = pd.Series(a['xgm'])[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Geometric median,2.0',c='olivedrab',linestyle='solid')


plt.xlabel('Iteration')
plt.ylabel('Minibatch')
plt.legend()
plt.grid(True)
plt.figure(figsize=(10, 10))
W=30


a = pd.read_csv('losses_10.0_10.0.csv')
rolling_mean = pd.Series(a['xmean'])[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Mean,10.0',c='royalblue',linestyle='dotted')
rolling_mean = pd.Series(a['xgm'])[0:1000].rolling(window=W).mean()
plt.plot(rolling_mean,label='Geometric median,10.0',c='royalblue',linestyle='solid')

plt.xlabel('Iteration')
plt.ylabel('Minibatch')
plt.legend()
plt.grid(True)

# %%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")
data = [
    [10.0, 200, 0.9823336999774355], [5.0, 200, 0.9862078518532683], [3.0, 200, 0.986207851853263],
    [2.0, 200, 0.9837001817041581], [1.0, 200, 0.9590840769093517], [10.0, 100, 0.9873529913576006],
    [5.0, 100, 0.9823336999774324], [3.0, 100, 0.9903789982975604], [2.0, 100, 0.9912586462379048],
    [1.0, 100, 0.7557032682738257], [10.0, 50, 0.9862078518532683], [5.0, 50, 0.9947955776535747],
    [3.0, 50, 0.9953478364113307], [2.0, 50, 0.9873529913576006], [1.0, 50, 0.4837001817041567],
    [10.0, 20, 0.9808896339335229], [5.0, 20, 0.9884284374424696], [3.0, 20, 0.8926418974320388],
    [2.0, 20, 0.5669743984715625], [1.0, 20, 0.3793307092726419], [10.0, 5, 0.9849909213596568],
    [5.0, 5, 0.986207851853263], [3.0, 5, 0.14648045170579282], [2.0, 5, 5.899231022435559e-09],
    [1.0, 5, 4.2530427370037744e-16], [10.0, 1, 0.0009742027749385161], [5.0, 1, 8.447677432720082e-23],
    [3.0, 1, 5.566807019072004e-110], [2.0, 1, 2.110837432400461e-65], [1.0, 1, 8.086165958912869e-96]
]
# data = [[10.0, 1, 0.8187716068306793], [5.0, 1, 0.5631411685724304], [3.0, 1, 1.7861977249548916e-79], [2.0, 1, 8.895609316996438e-16], 
#         [1.5, 1, 0.00015584916621313885], [1.0, 1, 0.3421212529994949], [0.5, 1, 0.1848366064225823], [10.0, 5, 0.7303221961812356], 
#         [5.0, 5, 0.9396266735034042], [3.0, 5, 7.289862093461272e-90], [2.0, 5, 4.097008707631957e-31], [1.5, 5, 0.7036760184280271], 
#         [1.0, 5, 0.6145530883520515], [0.5, 5, 0.9689255148313698], [10.0, 20, 0.9332753100953362], [5.0, 20, 0.0], 
#         [3.0, 20, 8.595573840732425e-250], [2.0, 20, 1.995436377761373e-124], [1.5, 20, 1.4813344177499e-23], 
#         [1.0, 20, 0.32466985586117514], [0.5, 20, 0.2709285068779209], [10.0, 50, 0.9621336013208919], 
#         [5.0, 50, 0.34541123681520103], [3.0, 50, 0.9479028637738279], [2.0, 50, 0.7891449786526565], [1.5, 50, 0.19248276073662618], 
#         [1.0, 50, 0.22266645854715109], [0.5, 50, 0.4167874694657139], [10.0, 100, 0.9559923354667955], [5.0, 100, 2.408700444078801e-304], 
#         [3.0, 100, 1.1351946313384578e-237], [2.0, 100, 3.1521758756006435e-33], [1.5, 100, 6.4571358783442e-08], [1.0, 100, 0.4072564084888316], 
#         [0.5, 100, 0.31579957902813544], [10.0, 200, 0.8839787155024341], [5.0, 200, 0.05782326263057664], [3.0, 200, 0.1446504069544617], 
#         [2.0, 200, 0.722876008174016],
# [1.5, 200, 0.911465618744437], [1.0, 200, 0.42632826601994245], [0.5, 200, 0.4207708978380029]]
# Separate the data based on degrees_of_freedom
data_dict = {}
for row in data:
    dof = row[0]
    if dof not in data_dict:
        data_dict[dof] = {'num_components': [], 'ks_p_value': []}
    data_dict[dof]['num_components'].append(row[1])
    data_dict[dof]['ks_p_value'].append(1-row[2])

# data_dict.pop(1.0)
# data_dict.pop(5.0)
# data_dict.pop(3.0)
# data_dict.pop(1.5)
# Create the plot
plt.figure(figsize=(10, 6))
for dof, dof_data in data_dict.items():
    plt.plot(dof_data['num_components'], dof_data['ks_p_value'], label=r'$\nu=$'+str(dof))

plt.axhline(y=0.05, color='r', linestyle='--', label='Null hypothesis (0.05)')
plt.xlabel('Number of Mixture Components')
plt.ylabel('KS p-value')
plt.title('KS p-value vs. Number of Components for Different Degrees of Freedom')
plt.legend()
plt.grid(True)
plt.show()



# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_style("darkgrid")

# Create a single figure with two subplots sharing the x-axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

W = 40

# Plot the first set of data in the first subplot (ax1)
a = pd.read_csv('losses_2.0_10.0.csv')
rolling_mean = pd.Series(np.clip(a['garrmean'], 0., 0.01))[0:1000].rolling(window=W).mean()
ax1.plot(rolling_mean, label='Mean,2.0->10.0', c='orangered', linestyle='dashed')
rolling_mean = pd.Series(np.clip(a['garrgm'], 0., 0.01))[0:1000].rolling(window=W).mean()
ax1.plot(rolling_mean, label='Geometric median,2.0->10.0', c='orangered', linestyle='solid')

# a = pd.read_csv('losses_2.0_2.0.csv')
# rolling_mean = pd.Series(np.clip(a['garrmean'], 0., 0.01))[0:1000].rolling(window=W).mean()
# ax1.plot(rolling_mean, label='Mean,2.0->2.0', c='limegreen', linestyle='dashed')
# rolling_mean = pd.Series(np.clip(a['garrgm'], 0., 0.01))[0:1000].rolling(window=W).mean()
# ax1.plot(rolling_mean, label='Geometric median,2.0->2.0', c='limegreen', linestyle='solid')

a = pd.read_csv('losses_10.0_10.0.csv')
rolling_mean = pd.Series(np.clip(a['garrmean'], 0., 0.01))[0:1000].rolling(window=W).mean()
ax1.plot(rolling_mean, label='Mean,10.0->10.0', c='limegreen', linestyle='dotted')
rolling_mean = pd.Series(np.clip(a['garrgm'], 0., 0.01))[0:1000].rolling(window=W).mean()
ax1.plot(rolling_mean, label='Geometric median,10.0->10.0', c='limegreen', linestyle='solid')

a = pd.read_csv('losses_10.0_2.0.csv')
rolling_mean = pd.Series(np.clip(a['garrmean'], 0., 0.01))[0:1000].rolling(window=W).mean()
ax1.plot(rolling_mean, label='Mean,10.0->2.0', c='royalblue', linestyle='dashed')
rolling_mean = pd.Series(np.clip(a['garrgm'], 0., 0.01))[0:1000].rolling(window=W).mean()
ax1.plot(rolling_mean, label='Geometric median,10.0->2.0', c='royalblue', linestyle='solid')

ax1.set_ylabel(r'$\left |\nabla_\phi \mathcal{L}(\theta) \right |$',fontsize=24)
ax1.legend(fontsize=14.5)
ax1.grid(True)

# ax1.annotate('Geometric medians overlap', 
#             xy=(1, 0), 
#             xycoords='data',
#             xytext=(0.7, 0.15), 
#             textcoords='axes fraction',
#             arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
#             fontsize=18, 
#             ha='center')

ax1.annotate('Geometric medians overlap', 
             xy=(0.5, 0.06),  # Change x-coordinate to 0.5 for center of plot
             xycoords='axes fraction',  # Set to 'axes fraction' to use fraction of axes for coordinates
             xytext=(0.6, 0.15), 
             textcoords='axes fraction',
             arrowprops=dict(arrowstyle="->", color='black', lw=1.5),
             fontsize=18, 
             ha='center',
             annotation_clip=False)  # Allows the arrow to point outside the axes

# Plot the second set of data in the second subplot (ax2)
a = pd.read_csv('losses_2.0_10.0.csv')
rolling_mean = pd.Series(np.clip(a['loss'], 0., 5.))[0:1000].rolling(window=W).mean()
ax2.plot(rolling_mean, label='Mean,2.0->10.0', c='orangered', linestyle='dashed')
rolling_mean = pd.Series(np.clip(a['gloss'], 0., 5.))[0:1000].rolling(window=W).mean()
ax2.plot(rolling_mean, label='Geometric median,2.0->10.0', c='orangered', linestyle='solid')

# a = pd.read_csv('losses_2.0_2.0.csv')
# rolling_mean = pd.Series(np.clip(a['loss'], 0., 5.))[0:1000].rolling(window=W).mean()
# ax2.plot(rolling_mean, label='Mean,2.0->2.0', c='limegreen', linestyle='dashed')
# rolling_mean = pd.Series(np.clip(a['gloss'], 0., 5.))[0:1000].rolling(window=W).mean()
# ax2.plot(rolling_mean, label='Geometric median,2.0->2.0', c='limegreen', linestyle='solid')

a = pd.read_csv('losses_10.0_10.0.csv')
rolling_mean = pd.Series(np.clip(a['loss'], 0., 5.))[0:1000].rolling(window=W).mean()
ax2.plot(rolling_mean, label='Mean,10.0->10.0', c='limegreen', linestyle='dotted')
rolling_mean = pd.Series(np.clip(a['gloss'], 0., 5.))[0:1000].rolling(window=W).mean()
ax2.plot(rolling_mean, label='Geometric median,10.0->10.0', c='limegreen', linestyle='solid')

a = pd.read_csv('losses_10.0_2.0.csv')
rolling_mean = pd.Series(np.clip(a['loss'], 0., 5.))[0:1000].rolling(window=W).mean()
ax2.plot(rolling_mean, label='Mean,10.0->2.0', c='royalblue', linestyle='dashed')
rolling_mean = pd.Series(np.clip(a['gloss'], 0., 5.))[0:1000].rolling(window=W).mean()
ax2.plot(rolling_mean, label='Geometric median,10.0->2.0', c='royalblue', linestyle='solid')

ax2.set_xlabel('Iteration',fontsize=24)
ax2.set_ylabel(r'$\mathcal{L}(\theta)$',fontsize=24)
#ax2.legend()
ax2.grid(True)

# plt.annotate('overlap', xy=(0, 3), xytext=(3, 4),
#              arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=7))

plt.tight_layout()  # Adjust subplot layout for better spacing
plt.savefig('grad_instability.png')

# %%

# %%
