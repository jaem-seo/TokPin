import tensorflow as tf
import numpy as np

def ic_func(x):
    return 2 - x ** 2

def fc_func(x):
    return 1.5 - 0.5 * x

def ic_bc_data():

    # set number of data points
    N_b = 200    # boundary
    N_t = 200    # initial time
    N_c = 1000  # collocation point

    # set boundary
    xmin = 0.0
    xmax = 1.0
    tmin = -0.1
    tmax = 1.0
    tic = 0.0
    tfc = 1.0

    # initial condition
    initial_xt = np.linspace([xmin, tic], [xmax, tic], N_t)
    initial_u = ic_func(initial_xt[:, 0:1])
    
    # final condition
    final_xt = np.linspace([xmin, tfc], [0.2, tfc], 3) # 3 core points
    final_u = fc_func(final_xt[:, 0:1])
    
    # boundary condition
    boundary_up = np.linspace([xmax, tmin], [xmax, tmax], N_b)
    boundary_up_sol = np.ones((N_b, 1))

    # collection of initial and boundary condition
    xt_init = initial_xt
    u_init_sol = initial_u
    xt_final = final_xt
    u_final_sol = final_u
    xt_bnd = boundary_up
    u_bnd_sol = boundary_up_sol

    # collocation point
    t_col_data = np.random.uniform(tmin, tmax, [N_c, 1])
    x_col_data = np.random.uniform(xmin, xmax, [N_c, 1])
    xt_col_data = np.concatenate([x_col_data, t_col_data], axis=1)
    #xt_col = xt_col_data
    xt_col = np.concatenate((xt_col_data, initial_xt), axis=0)
    #xt_col = np.concatenate((xt_col_data, initial_xt, xt_bnd), axis=0)
    #xt_col = np.concatenate((xt_col_data, xt_init), axis=0)
    #xt_col = np.concatenate((xt_col_data, xt_bnd), axis=0)
    #xt_col = np.concatenate((xt_col_data, xt_init, xt_bnd), axis=0)

    # convert all to tensors
    xt_col = tf.convert_to_tensor(xt_col, dtype=tf.float32)
    xt_bnd = tf.convert_to_tensor(xt_bnd, dtype=tf.float32)
    u_bnd_sol = tf.convert_to_tensor(u_bnd_sol, dtype=tf.float32)
    xt_init = tf.convert_to_tensor(xt_init, dtype=tf.float32)
    u_init_sol = tf.convert_to_tensor(u_init_sol, dtype=tf.float32)
    xt_final = tf.convert_to_tensor(xt_final, dtype=tf.float32)
    u_final_sol = tf.convert_to_tensor(u_final_sol, dtype=tf.float32)

    return xt_col, xt_bnd, u_bnd_sol, xt_init, u_init_sol, xt_final, u_final_sol