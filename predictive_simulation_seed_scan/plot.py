import numpy as np
import matplotlib.pyplot as plt
from train import *

ns = 5
tmin, tmax = 0, 1
xmin, xmax = 0, 1
n_x, n_t = 51, 51

x = np.linspace(xmin, xmax, n_x)
t = np.linspace(tmin, tmax, n_t)
X, T = np.meshgrid(x, t)
Ys = []
xt = np.zeros([n_x * n_t, 2])
for i in range(n_t):
    xt[i * n_x: (i+1) * n_x, 0] = x
    xt[i * n_x: (i+1) * n_x, 1] = t[i]
    
for seed in range(ns):
    agent = TransportPinn(seed)
    agent.load_weights("./save_weights/")
    y = agent.nn.predict(xt)
    Ys.append(y.reshape(n_t, n_x, 2))

# Load FDM results
n_th = int(2e5) + 1
n_xh = 101
th = np.linspace(tmin, tmax, n_th)
xh = np.linspace(xmin, xmax, n_xh)
u_h = np.load('../u_euler_2e5_101.npy')

# Load loss histories
iters = []
losses = []
for seed in range(ns):
    with open(f'save_weights/loss{seed}.txt', 'r') as f:
        a = f.readlines()
        iters.append([float(line.split()[0]) for line in a])
        losses.append([float(line.split()[1]) for line in a])
    iters[seed][0] = 1


# Plot figure
fig, axs = plt.subplots(1, 3, figsize=(7, 2))
cs = ['b', 'r', 'g', 'm', 'orange']

for seed in range(ns):
    axs[0].plot(iters[seed], losses[seed], c=cs[seed], label=f'seed={seed}')
    axs[1].plot(x, Ys[seed][-1, :, 0], c=cs[seed])
    axs[1].plot(x, Ys[seed][-1, :, 1], c=cs[seed], ls='--')
    #axs[2].plot(t, Ys[seed][:, 0, 0], c=cs[seed])
    #axs[2].plot(t, Ys[seed][:, 0, 1], c=cs[seed], ls='--')
    axs[2].plot(t, np.mean(Ys[seed][:, :, 0], axis=-1), c=cs[seed])
    axs[2].plot(t, np.mean(Ys[seed][:, :, 1], axis=-1), c=cs[seed], ls='--')
    
axs[0].set_yscale("log")
axs[0].set_xscale("log")
axs[0].set_xlabel('Iterations')
axs[0].set_ylabel('Weighted sum loss')
#axs[0].set_ylim([None, 100])
#axs[0].legend(fontsize=7, ncol=2, loc='upper right')

axs[1].plot(xh, u_h[0, -1, :], 'k')
axs[1].plot(xh, u_h[1, -1, :], 'k--')
#axs[1].set_ylim([0.8, 3.5])
axs[1].set_ylabel(' ')

#axs[2].plot(th, u_h[0, :, 0], 'k')
#axs[2].plot(th, u_h[1, :, 0], 'k--')
#axs[2].set_ylim([0.8, 3.5])

axs[2].plot(th, np.mean(u_h[0, :, :], axis=-1), 'k')
axs[2].plot(th, np.mean(u_h[1, :, :], axis=-1), 'k--')
#axs[2].set_ylim([0.8, 3.5])
axs[2].set_ylabel(' ')

for i in range(3):
    axs[i].spines['right'].set_visible(False)
    axs[i].spines['top'].set_visible(False)
    
plt.tight_layout()
#plt.savefig('seed_scan.svg')
plt.show()
