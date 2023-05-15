import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import matplotlib.pyplot as plt

from train import *
from ic_bc_data import ic_func
from coefficients import simple_equil


def gradient(u, x):
    dx = x[1] - x[0]
    up = np.append(u, u[-1])
    um = np.append(u[0], u)
    u_x = (up - um) / dx
    u_x = 0.5 * (u_x[1:] + u_x[:-1])
    return u_x

# Load PINN
agent = TransportPinn()
agent.load_weights("./best_saved/")

t_target = 1
n_x = 101

x = np.linspace(0, 1, n_x)
xt = np.zeros([n_x, 2])
xt[:, 0] = x
xt[:, 1] = t_target
y = agent.nn.predict(xt)
dy_x = np.diff(y, axis=0) / (x[1] - x[0])
dy_x_central = gradient(y[:, 1], x)

btor, rmaj, roc, g11, vr, vr53, mu = simple_equil(y, x, t_target)
eps = roc * x / rmaj
Bp = eps * btor * mu
dp_r = 2 / roc * np.diff(y[:, 0] * y[:, 1]) / (x[1] - x[0]) * 1.6e3 # [Pa / m]
dp_r_central = 2 / roc * gradient(y[:, 0] * y[:, 1], x) * 1.6e3
jbs = -eps[:-1] ** 0.5 * (1 / Bp[:-1]) * dp_r # [A / m2]
jbs_central = -eps ** 0.5 * (1 / Bp) * dp_r_central # [A / m2]


# Load FDM results
u_h = np.load('u_euler_2e5_101.npy')
n_th, n_xh = u_h.shape[1:]
xh = np.linspace(0, 1, n_xh)
du_x_h = np.diff(u_h[:, -1], axis=1) / (xh[1] - xh[0])
du_x_h_central = gradient(u_h[1, -1], xh)

btor, rmaj, roc, g11, vr, vr53, mu = simple_equil(u_h[:, -1], xh, t_target)
eps = roc * xh / rmaj
Bp = eps * btor * mu
dp_r = 2 / roc * np.diff(u_h[0, -1] * u_h[1, -1]) / (xh[1] - xh[0]) * 1.6e3 # [Pa / m]
dp_r_central = 2 / roc * gradient(u_h[0, -1] * u_h[1, -1], xh) * 1.6e3
jbs_h = -eps[:-1] ** 0.5 * (1 / Bp[:-1]) * dp_r # [A / m2]
jbs_h_central = -eps ** 0.5 * (1 / Bp) * dp_r_central # [A / m2]


# Plot

fig, axs = plt.subplots(1, 3, sharex=True, figsize=(8, 2))

axs[0].plot(xh, u_h[1, -1], 'b', label='FDM')
axs[0].plot(x, y[:, 1], 'r', label='PINN')
axs[0].plot(xh, u_h[0, -1], 'b--')
axs[0].plot(x, y[:, 0], 'r--')
axs[0].set_xlabel('ρ')
axs[0].set_ylabel('$n_{e}$ ($10^{19}$/m$^3$), $T_{e}$ (keV)')
#axs[0].legend(loc='lower left')
axs[0].spines['right'].set_visible(False)
axs[0].spines['top'].set_visible(False)

axs[1].plot(xh[:-1], du_x_h[1], 'b', label='FDM')
#axs[1].plot(xh, du_x_h_central, 'b', label='FDM (CD)')
axs[1].plot(x[:-1], dy_x[:, 1], 'r', lw=2, label='PINN')
#axs[1].plot(x, dy_x_central, 'r', lw=2, label='PINN')
#axs[1].set_ylim([-4.5, 1.5])
axs[1].set_xlabel('ρ')
axs[1].set_ylabel('$dT_{e}/dρ$ (a.u.)')
axs[1].legend()
axs[1].spines['right'].set_visible(False)
axs[1].spines['top'].set_visible(False)

axs[2].plot(xh[:-1], 1e-6 * jbs_h, 'b', label='FDM')
#axs[2].plot(xh, 1e-6 * jbs_h_central, 'b', label='FDM (CD)')
axs[2].plot(x[:-1], 1e-6 * jbs, 'r', lw=2, label='PINN')
#axs[2].plot(x, 1e-6 * jbs_central, 'r', lw=2, label='PINN')
axs[2].set_xlabel('ρ')
axs[2].set_ylabel('$j_{BS}$ (MA/m$^2$)')
axs[2].legend()
axs[2].spines['right'].set_visible(False)
axs[2].spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('profile_comparison.svg')
plt.show()



fig, axs = plt.subplots(1, 1, figsize=(1.5, 1.25))

#axs.plot(xh, u_h[1, -1], 'b', label='FDM')
axs.plot(xh, u_h[1, -1], 'b', marker='o', markersize=0.5, label='FDM')
axs.plot(x, y[:, 1], 'r', label='PINN')
axs.set_xlim([0.2, 0.3])
axs.set_ylim([2.45, 2.68])
#axs.set_xlim([0.4, 0.5])
#axs.set_ylim([2., 2.25])
#axs.set_xlabel('ρ')
#axs.set_ylabel('$n_{e}$ ($10^{19}$/m$^3$), $T_{e}$ (keV)')
#axs.legend()
axs.spines['right'].set_visible(False)
axs.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('zoom.svg')
plt.show()