import tensorflow as tf
import numpy as np

# Constants
eps = 1e-4
A, Z = 2., 1.

def simple_equil(u, x, t):
    btor = tf.convert_to_tensor(1.8544452600061) # [T]
    rmaj = tf.convert_to_tensor(1.79694367486757) #[m]
    roc = tf.convert_to_tensor(0.619694589934167) # [m]
    g11 = tf.ones_like(x) * 1.05360797714196
    vr = 32.38877421 * (x + 0.1)
    vr53 = vr ** (5/3)
    mu = 1 / (0.64796905 + 1.24131919 * x - 3.97550269 * x ** 2 + 7.01150585 * x ** 3)
    return btor, rmaj, roc, g11, vr, vr53, mu

def hatl(u, x, t, btor, roc, mu): # Heat conductivity Anomalous by Taroni for L-mode
    ne, te = u[:, 0:1], 2 - x ** 2#u[:, 1:2]
    dte_x = tf.gradients(te, x)
    dne_x = tf.gradients(ne, x)
    hatl = 0.25 / btor * tf.math.abs(dte_x + te / (ne + eps) * dne_x) / mu ** 2
    #hatl = -0.25 / btor * (dte_x + te / (ne + eps) * dne_x) / mu ** 2
    return hatl

def hagb(u, x, t, btor, roc, mu): # Heat conductivity Anomalous gyroBohm
    te = 2 - x ** 2#u[:, 1:2]
    dte_x = tf.gradients(te, x)
    hagb = 0.16 * A ** 0.5 / Z / btor ** 2 / roc * tf.math.sqrt(tf.math.abs(te) + eps) * tf.math.abs(dte_x)
    #hagb = -0.16 * A ** 0.5 / Z / btor ** 2 / roc * tf.math.sqrt(tf.math.abs(te) + eps) * dte_x
    return hagb

def diffusion(u, x, t, btor, roc, mu):
    hb = hatl(u, x, t, btor, roc, mu)
    hgb = hagb(u, x, t, btor, roc, mu)
    he = hb + hgb
    xi = 2 * hb + hgb
    dn = 0.3 * 0.5 * (he + xi)
    return dn, he, xi # [m^2/s]

def source(u, x, t):
    return 2 * tf.ones_like(x), 0.1 * (1 - x ** 2), 0.1 * (1 - x ** 2) # [10^19/m^3/s], [MW/m^3], [MW/m^3]