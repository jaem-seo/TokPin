# Setting initial and boundary conditions
# PINN for Burgers' equation (L-BFGS version)
# coded by St.Watermelon

import tensorflow as tf
import numpy as np

def ic_func(x):
    return 2 - x ** 2

def ic_bc_data():

    # set number of data points
    N_b = 100    # boundary
    N_t = 100    # initial time
    N_c = 1000  # collocation point

    # set boundary
    xmin = 0.0
    xmax = 1.0
    tmin = 0.0
    tmax = 1.0

    # initial condition
    initial_xt = np.linspace([xmin, tmin], [xmax, tmin], N_t)
    initial_u = ic_func(initial_xt[:, 0:1])
    #initial_u = 2.0 * (1.0 - initial_xt[:, 0:1] ** 2) ** 6
    #initial_u = initial_xt[:, 0:1] * (1.0 - initial_xt[:, 0:1])
    
    # boundary condition
    boundary_up = np.linspace([xmax, tmin], [xmax, tmax], N_b)
    boundary_up_sol = np.ones((N_b, 1))
    #boundary_up_sol = 0.5 * np.ones((N_b, 1))
    #boundary_down = np.linspace([xmin, tmin], [xmin, tmax], N_b)
    #boundary_down_sol = np.ones((N_b, 1))

    # collection of initial and boundary condition
    xt_init = initial_xt
    u_init_sol = initial_u
    xt_bnd = boundary_up
    u_bnd_sol = boundary_up_sol
    #xt_bnd = np.concatenate([initial_xt, boundary_up], axis=0)
    #u_bnd_sol = np.concatenate([initial_u, boundary_up_sol], axis=0)
    #xt_bnd = np.concatenate([initial_xt, boundary_up, boundary_down], axis=0)
    #u_bnd_sol = np.concatenate([initial_u, boundary_up_sol, boundary_down_sol], axis=0)

    # collocation point
    t_col_data = np.random.uniform(tmin, tmax, [N_c, 1])
    x_col_data = np.random.uniform(xmin, xmax, [N_c, 1])
    xt_col_data = np.concatenate([x_col_data, t_col_data], axis=1)
    xt_col = xt_col_data
    #xt_col = np.concatenate((xt_col_data, xt_init), axis=0)
    #xt_col = np.concatenate((xt_col_data, xt_bnd), axis=0)
    #xt_col = np.concatenate((xt_col_data, xt_init, xt_bnd), axis=0)

    # convert all to tensors
    xt_col = tf.convert_to_tensor(xt_col, dtype=tf.float32)
    xt_bnd = tf.convert_to_tensor(xt_bnd, dtype=tf.float32)
    u_bnd_sol = tf.convert_to_tensor(u_bnd_sol, dtype=tf.float32)
    xt_init = tf.convert_to_tensor(xt_init, dtype=tf.float32)
    u_init_sol = tf.convert_to_tensor(u_init_sol, dtype=tf.float32)

    return xt_col, xt_bnd, u_bnd_sol, xt_init, u_init_sol
 

 

