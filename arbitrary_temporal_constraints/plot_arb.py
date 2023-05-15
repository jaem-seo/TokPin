import numpy as np
import matplotlib.pyplot as plt
from train import *

# Define IC & FC
def ic_func(x):
    return 2 - x ** 2

def fc_func(x):
    return 1.5 - 0.5 * x


# Load FDM results
n_t = int(5e4) + 1
n_x = 51
tmin, tmax = 0, 1.1
xmin, xmax = 0, 1
t = np.linspace(tmin, tmax, n_t)
x = np.linspace(xmin, xmax, n_x)
u = np.load('ne_5e4_51.npy')[0]

# Load PINN model
n_t = 25
n_x = 21
ts = [-0.1, 0, 0.5, 1]
agent = TransportPinn()
agent.nn.load_weights('save_weights (w_tc2=1)/best.h5')
xt0 = np.zeros([n_t, 2])
xt0[:, 1] = np.linspace(-0.2, 1.1, n_t)
xts = [np.zeros([n_x, 2]) for _ in ts]
for i in range(len(ts)):
    xts[i][:, 0] = np.linspace(0, 1, n_x)
    xts[i][:, 1] = ts[i]
y0 = agent.predict(xt0)
ys = [agent.predict(xt) for xt in xts]

# Load PINN model 2
agent2 = TransportPinn()
agent2.nn.load_weights('save_weights (w_tc2=0.01)/best.h5')
n_t = 15
xt02 = np.zeros([n_t, 2])
xt02[:, 1] = np.linspace(0, 1.1, n_t)
y02 = agent2.predict(xt02)
#ys2 = [agent2.predict(xt) for xt in xts]

# Plot 1
fig = plt.figure(figsize=(4, 2))

ax1 = plt.subplot(1, 1, 1)
ax1.plot(t, u[:, 0], 'b', lw=2, label='FDM solution')
ax1.plot(xt0[:, 1], y0, 'r', lw=2, label='PINN solution')
#ax1.plot(xt02[:, 1], y02, 'r', lw=2, alpha=0.3, label='PINN solution 2')
ax1.scatter([0, 1], [ic_func(0), fc_func(0)], color='k', marker='o', s=75, label='Temporal constraints', zorder=2)
for tt in ts:
    ax1.axvline(tt, color='g', ls='--', zorder=-1)
#ax1.legend()
ax1.set_ylim([0.9, 2.19])
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('$n_{e}$(ρ=0) ($10^{19}m^{-3}$)')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('evol.svg')
plt.show()


# Plot 2

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(4, 3))

it = 0
it1 = np.abs(t - ts[it]).argmin()
axs[0, 0].plot(xts[it][:, 0], ys[it], 'r', lw=2, label='PINN solution')
#axs[0, 0].legend()
axs[0, 0].spines['right'].set_visible(False)
axs[0, 0].spines['top'].set_visible(False)
axs[0, 0].set_title(' ')#f't={ts[it]:.1f} s', color='g')
axs[0, 0].set_ylabel('$n_{e}$ ($10^{19}m^{-3}$)')

it = 1
it1 = np.abs(t - ts[it]).argmin()
axs[0, 1].plot(x, u[it1, :], 'b', lw=2, label='FDM solution')
axs[0, 1].plot(xts[it][:, 0], ys[it], 'r', lw=2, label='PINN solution')
axs[0, 1].plot(x, ic_func(x), 'k--', lw=2, label='Temporal constraint 1\n(Initial condition)')
#axs[0, 1].legend()
axs[0, 1].spines['right'].set_visible(False)
axs[0, 1].spines['top'].set_visible(False)
axs[0, 1].set_title(' ')#f't={ts[it]:.1f} s', color='g')

it = 2
it1 = np.abs(t - ts[it]).argmin()
axs[1, 0].plot(x, u[it1, :], 'b', lw=2, label='FDM solution')
axs[1, 0].plot(xts[it][:, 0], ys[it], 'r', lw=2, label='PINN solution')
#axs[1, 0].legend()
axs[1, 0].spines['right'].set_visible(False)
axs[1, 0].spines['top'].set_visible(False)
axs[1, 0].set_title(' ')#f't={ts[it]:.1f} s', color='g')
axs[1, 0].set_xlabel('ρ')
axs[1, 0].set_ylabel('$n_{e}$ ($10^{19}m^{-3}$)')

it = 3
xf = np.linspace(0, 0.2, 3)
it1 = np.abs(t - ts[it]).argmin()
axs[1, 1].plot(x, u[it1, :], 'b', lw=2, label='FDM solution')
axs[1, 1].plot(xts[it][:, 0], ys[it], 'r', lw=2, label='PINN solution')
axs[1, 1].scatter(xf, fc_func(xf), s=20, color='k', marker='o', lw=2, label='Temporal constraint 2', zorder=2)
#axs[1, 1].legend()
axs[1, 1].spines['right'].set_visible(False)
axs[1, 1].spines['top'].set_visible(False)
axs[1, 1].set_title(' ')#f't={ts[it]:.1f} s', color='g')
axs[1, 1].set_xlabel('ρ')

plt.tight_layout()
plt.savefig('prof.svg')
plt.show()


# Plot 3

n_x, n_t = 51, 51
xx = np.linspace(0, 1, n_x)
tt = np.linspace(-0.1, 1.3, n_t)
X, T = np.meshgrid(xx, tt)

xt = np.zeros([n_x * n_t, 2])
for i in range(n_t):
    xt[i * n_x: (i+1) * n_x, 0] = xx
    xt[i * n_x: (i+1) * n_x, 1] = tt[i]

y = agent.nn.predict(xt)
Y = y.reshape(n_t, n_x, 1)

fig = plt.figure(figsize=(4, 2.5))
ax = fig.add_subplot(1, 1, 1, projection='3d', computed_zorder=False)
ax.contour3D(X, T, Y[:,:,0], 1000)
ax.plot(xx, np.zeros_like(xx), ic_func(xx), 'k--', lw=2, zorder=2)
ax.scatter(xf, np.ones_like(xf), fc_func(xf), s=10, color='k', marker='o', zorder=2, alpha=1)
ax.plot(np.ones_like(tt), tt, np.ones_like(tt), 'r--', lw=2, zorder=2)
ax.set_zlim([0.6, 2.9])
ax.view_init(30, 330)
ax.set_xlabel(' ')
ax.set_ylabel(' ')
ax.set_zlabel(' ')
ax.tick_params(pad=-2)

#plt.tight_layout()
plt.savefig('contour.svg')
plt.show()


'''
X, T = np.meshgrid(x, t)

fig = plt.figure(figsize=(4, 2.5))
ax = fig.add_subplot(1, 1, 1, projection='3d', computed_zorder=False)
ax.contour3D(X, T, u[:,:], 100)
ax.plot(xx, np.zeros_like(xx), ic_func(xx), 'k--', lw=2, zorder=2)
ax.plot(xx, np.ones_like(xx), fc_func(xx), 'k--', lw=2, zorder=2)
ax.plot(np.ones_like(tt), tt, np.ones_like(tt), 'r--', lw=2, zorder=2)
ax.set_zlim([0.6, 2.9])
ax.view_init(30, 330)
ax.set_xlabel(' ')
ax.set_ylabel(' ')
ax.set_zlabel(' ')

#plt.tight_layout()
plt.savefig('contour0.svg')
plt.show()'''