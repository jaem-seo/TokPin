from time import time
import numpy as np
import matplotlib.pyplot as plt

n_t = int(5e4) + 1
n_x = 51
tmin, tmax = 0, 1.1
xmin, xmax = 0, 1
n_output = 2
eps = 1.e-6
A, Z = 2., 1.

t = np.linspace(tmin, tmax, n_t)
x = np.linspace(xmin, xmax, n_x)
u = np.zeros([n_output, n_t, n_x])

dt, dx = t[1] - t[0], x[1] - x[0]

def gradient(u, x):
    dx = x[1] - x[0]
    up = np.append(u, u[-1])
    um = np.append(u[0], u)
    u_x = (up - um) / dx
    u_x = 0.5 * (u_x[1:] + u_x[:-1])
    return u_x

def ic_func(x):
    return 2 - x ** 2

def simple_equil(u, x):
    btor = 1.8544452600061 # [T]
    rmaj = 1.79694367486757 #[m]
    roc = 0.619694589934167 # [m]
    g11 = np.ones_like(x) * 1.05360797714196
    vr = 32.38877421 * (x + 0.1)
    vr53 = vr ** (5/3)
    mu = 1 / (0.64796905 + 1.24131919 * x - 3.97550269 * x ** 2 + 7.01150585 * x ** 3)
    return btor, rmaj, roc, g11, vr, vr53, mu

def hatl(u, x, btor, roc, mu): # Heat conductivity Anomalous by Taroni for L-mode
    ne, te = u[0], u[1]
    dte_x = gradient(te, x)
    dne_x = gradient(ne, x)
    hatl = 0.25 / btor * np.abs(dte_x + te / (ne + eps) * dne_x) / mu ** 2
    return hatl

def hagb(u, x, btor, roc, mu): # Heat conductivity Anomalous gyroBohm
    te = u[1]
    dte_x = gradient(te, x)
    hagb = 0.16 * A ** 0.5 / Z / btor ** 2 / roc * np.sqrt(np.abs(te) + eps) * np.abs(dte_x)
    return hagb

def diffusion(u, x, btor, roc, mu):
    hb = hatl(u, x, btor, roc, mu)
    hgb = hagb(u, x, btor, roc, mu)
    he = hb + hgb
    xi = 2 * hb + hgb
    dn = 0.3 * 0.5 * (he + xi)
    return dn, he, xi # [m^2/s]

def source(u, x):
    return 2 * np.ones_like(x), 0.1 * (1 - x ** 2), 0.1 * (1 - x ** 2) # [10^19/m^3/s], [MW/m^3], [MW/m^3]

def f1(ne, x, d_ne, s, roc, vr, g11):
    Gamma_e = - vr * g11 * d_ne * gradient(ne, x) / roc
    dvrne_t = vr * s - gradient(Gamma_e, x) / roc
    return dvrne_t

def f2(ne, te, x, d_ne, d_te, s, roc, vr, g11):
    vr53 = vr ** (5/3)
    vr23 = vr ** (2/3)
    Gamma_e = - vr * g11 * d_ne * gradient(ne, x) / roc
    q_e = - ne * vr * g11 * d_te * gradient(te, x) / roc
    dvr53nete_t = 625 * 2/3 * vr53 * s - 2/3 * vr23 * gradient(q_e + 2.5 * te * Gamma_e, x) / roc
    return dvr53nete_t


for i in range(n_output):
    u[i, :, :] = ic_func(x)

t0 = time()
for it in range(n_t - 1):
    btor, rmaj, roc, g11, vr, vr53, mu = simple_equil(u[:, it, :], x)
    d_ne, d_te, d_ti = diffusion(u[:, it, :], x, btor, roc, mu)
    s_ne, s_te, s_ti = source(u, x)
    u[0, it + 1, :] = 1 / vr * (vr * u[0, it, :] + dt * f1(u[0, it, :], x, d_ne, s_ne, roc, vr, g11))
    #u[1, it + 1, :] = u[0, it + 1, :]
    #u[1, it + 1, :] = 1 / vr53 / u[0, it + 1, :] * (vr53 * u[0, it, :] * u[1, it, :] + dt * f2(u[0, it, :], u[1, it, :], x, d_ne, d_te, s_te, roc, vr, g11))
    #u[2, it + 1, :] = 1 / vr53 / u[0, it + 1, :] * (vr53 * u[0, it, :] * u[2, it, :] + dt * f2(u[0, it, :], u[2, it, :], x, d_ne, d_ti, s_ti, roc, vr, g11))
    u[:, it + 1, -1] = u[:, 0, -1]
print(f'\nComputation time for Euler method: {time() - t0} seconds')

# Plot results
times = np.array([0.0, 0.1, 0.25, 0.5, 1])
fig, axs = plt.subplots(1, len(times) + 1, figsize=(12, 2), sharey=True)

its = []
for i in range(len(times)):
    it = np.abs(t - times[i]).argmin()
    its.append(it)
    for j in range(n_output):
        axs[i].plot(x, u[j, it, :])
    axs[i].set_title(f't={times[i]}')
    axs[i].set_xlabel('rhon')

for j in range(n_output):
    axs[-1].plot(times, [u[j, i, 0] for i in its])
axs[-1].set_title('Evolution')
axs[-1].set_ylim([np.min(u), 1.25 * np.max(u)])
axs[-1].set_xlabel('Time')
plt.show()