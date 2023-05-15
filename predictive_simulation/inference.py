import numpy as np
import matplotlib.pyplot as plt

from train import *
from ic_bc_data import ic_func

agent = TransportPinn()
#agent.load_weights("./saved_weights/")
agent.load_weights("./best_saved/")

tmin, tmax = 0, 1
xmin, xmax = 0, 1
n_x, n_t = 51, 51

x = np.linspace(xmin, xmax, n_x)
t = np.linspace(tmin, tmax, n_t)
X, T = np.meshgrid(x, t)
xt = np.zeros([n_x * n_t, 2])
for i in range(n_t):
    xt[i * n_x: (i+1) * n_x, 0] = x
    xt[i * n_x: (i+1) * n_x, 1] = t[i]

y = agent.nn.predict(xt)
Y = y.reshape(n_t, n_x, 2)



# Load FDM results
n_th = int(2e5) + 1
n_xh = 101
th = np.linspace(tmin, tmax, n_th)
xh = np.linspace(xmin, xmax, n_xh)
u_h = np.load('u_euler_2e5_101.npy')
n_tl = int(5e2) + 1
n_xl = 11
tl = np.linspace(tmin, tmax, n_tl)
xl = np.linspace(xmin, xmax, n_xl)
u_l = np.load('u_euler_5e2_11.npy')



lims = [0.75, 3.25]
#fig = plt.figure(figsize=(5, 2))
fig = plt.figure(figsize=(4.5, 2))
for i in range(2):
    ax = fig.add_subplot(1, 2, i+1, projection='3d', computed_zorder=False)
    ax.contour3D(X, T, Y[:,:,i], 300)
    ax.plot(x, np.zeros_like(x), ic_func(x), 'k--', lw=2, zorder=2)
    ax.plot(np.ones_like(t), t, np.ones_like(t), 'r--', lw=2, zorder=2)
    #ax.plot(np.zeros_like(th), th, u_h[i, :, 0], 'b--', lw=2, zorder=2)
    #ax.set_xlabel(' ')
    #ax.set_ylabel(' ')
    #ax.set_zlabel(' ')
    ax.set_zlim(lims)
    ax.tick_params(pad=-2)
    ax.view_init(30, 330)

#plt.tight_layout()

plt.savefig('pinn_inference_3d.svg')
plt.show()



times = [0.1, 1.0]
cs = ['b', 'r', 'g']
#fig, axs = plt.subplots(len(times), 2, figsize=(4.7, 2), sharex=True)#, sharey=True)
fig, axs = plt.subplots(len(times), 2, figsize=(4.5, 2), sharex=True)#, sharey=True)
for i in range(len(times)):
    it1 = np.abs(t - times[i]).argmin()
    it2 = np.abs(th - times[i]).argmin()
    it3 = np.abs(tl - times[i]).argmin()
    for j in range(n_output):
        #axs[i, j].plot(xl, u_l[j, it3, :], color='g', label='FDM_low')
        axs[i, j].plot(xh, u_h[j, it2, :], color='b', label='FDM_high')
        axs[i, j].plot(x, Y[it1, :, j], color='r', label='PINN')
        axs[i, j].plot(x, ic_func(x), 'k--', label='Initial')
        #axs[i, j].set_title(f't={times[i]} s')
        
        axs[i, j].spines['right'].set_visible(False)
        axs[i, j].spines['top'].set_visible(False)
        axs[i, j].set_ylim([0.9, 3.1])
        #if i == 0:
        #    axs[i, j].legend(prop={'size': 8}, ncol=2)
        #if i == len(times) - 1:
        #    axs[i, j].set_xlabel('ρ')

#plt.subplots_adjust(wspace=0.85)
plt.subplots_adjust(wspace=0.5)
plt.savefig('pinn_inference_1d.svg')
plt.show()