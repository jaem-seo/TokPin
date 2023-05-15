import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from time import time
from ic_bc_data import ic_bc_data, ic_func, fc_func
from coefficients import *

n_output = 1 # [10^19/m^3], [keV], [keV]
seed = 42

np.random.seed(seed)
tf.random.set_seed(seed)

class NN(Model):

    def __init__(self, layers=[20]*3, activation='tanh'):
        super(NN, self).__init__()

        initializer = tf.keras.initializers.GlorotUniform
        self.hs = [Dense(layer, activation=activation, kernel_initializer=initializer) for layer in layers]
        self.u = Dense(n_output, activation='relu')


    def call(self, state):
        x = state
        for h in self.hs:
            x = h(x)
        return self.u(x)


class TransportPinn(object):

    def __init__(self):

        self.nn = NN()
        self.nn.build(input_shape=(None, 2))

        self.train_loss_history = []
        self.iter_count = 0
        self.instant_loss = np.inf
        self.best_loss = np.inf
        self.individual_loss = [0, 0, 0]
        self.weights = [1, 1, 1, 1]
        self.resample_period = 10
        self.iter_tmp = 0

    @tf.function
    def physics_net(self, xt):
        x, t = xt[:, 0:1], xt[:, 1:2]
        xt_t = tf.concat([x, t], 1)
        u = self.nn(xt_t)
        
        btor, rmaj, roc, g11, vr, vr53, mu = simple_equil(u, x, t)
        D = diffusion(u, x, t, btor, roc, mu)
        S = source(u, x, t)
        ne, te, ti = u[:, 0:1], u[:, 1:2], u[:, 2:3]
        pde_electron_particle, pde_electron_energy, pde_ion_energy = 0, 0, 0
        
        dvrne_t = tf.gradients(vr * ne, t)
        dne_x = tf.gradients(ne, x)
        neflux = -g11 * vr * D[0] * dne_x / roc
        dneflux_x = tf.gradients(neflux, x)
        pde_electron_particle = 1e-3 * (1 / vr * dvrne_t + 1 / vr * dneflux_x / roc - S[0])
        #pde_electron_particle = 1 / vr * dvrne_t + 1 / vr * dneflux_x / roc - S[0]
        
        if n_output > 1:
            dte_x = tf.gradients(te, x)
            dnete_t = tf.gradients(vr53 * ne * te, t)
            eeflux = -g11 * vr * ne * D[1] * dte_x / roc + 2.5 * te * neflux
            deeflux_x = tf.gradients(eeflux, x)
            pde_electron_energy = 1.6e-3 * (1.5 / vr53 * dnete_t + 1 / vr * deeflux_x / roc) - S[1]
            #pde_electron_energy = 1.5 / vr53 * dnete_t + 1 / vr * deeflux_x / roc - 625. * S[1]
        
        if n_output > 2:
            ni, niflux = ne / Z, neflux / Z
            dti_x = tf.gradients(ti, x)
            dniti_t = tf.gradients(vr53 * ni * ti, t)
            eiflux = -g11 * vr * ni * D[2] * dti_x / roc + 2.5 * ti * niflux
            deiflux_x = tf.gradients(eiflux, x)
            pde_ion_energy = 1.6e-3 * (1.5 / vr53 * dniti_t + 1 / vr * deeflux_x / roc) - S[2]
            
        return (pde_electron_particle, pde_electron_energy, pde_ion_energy)[:n_output]
    

    def save_weights(self, path):
        self.nn.save_weights(path + 'nn.h5')


    def load_weights(self, path):
        self.nn.load_weights(path + 'nn.h5')


    def compute_loss(self, f, u_bnd_hat, u_bnd_sol, u_init_hat, u_init_sol, u_fin_hat, u_fin_sol):
        loss_col = tf.reduce_mean(tf.square(f))
        loss_bnd = tf.reduce_mean(tf.square(u_bnd_hat - u_bnd_sol))
        loss_init = tf.reduce_mean(tf.square(u_init_hat - u_init_sol))
        loss_fin = tf.reduce_mean(tf.square(u_fin_hat - u_fin_sol))
        #loss_col = tf.reduce_mean(tf.abs(f))
        #loss_bnd = tf.reduce_mean(tf.abs(u_bnd_hat - u_bnd_sol))
        #loss_init = tf.reduce_mean(tf.abs(u_init_hat - u_init_sol))
        self.individual_loss = [loss_col.numpy(), loss_bnd.numpy(), loss_init.numpy(), loss_fin.numpy()]
        
        w1, w2, w3, w4 = self.weights
        loss = w1 * loss_col + w2 * loss_bnd + w3 * loss_init + w4 * loss_fin
        return loss


    def compute_grad(self):
        with tf.GradientTape() as tape:
            f = self.physics_net(self.xt_col)
            u_bnd_hat = self.nn(self.xt_bnd)
            u_init_hat = self.nn(self.xt_init)
            u_fin_hat = self.nn(self.xt_fin)
            loss = self.compute_loss(f, u_bnd_hat, self.u_bnd_sol, u_init_hat, self.u_init_sol, u_fin_hat, self.u_fin_sol)

        grads = tape.gradient(loss, self.nn.trainable_variables)
        return loss, grads


    def callback(self, arg=None):
        if self.iter_count % 10 == 0:
            print('iter=', self.iter_count, ', loss=', self.instant_loss, self.individual_loss)
            self.train_loss_history.append([self.iter_count, self.instant_loss] + self.individual_loss)
            #self.iter_tmp += 1
            #if (self.iter_tmp > 1) and (self.train_loss_history[-1][1] > 0.999 * self.train_loss_history[-2][1]):
            #    print('Resample')
            #    self.xt_col, self.xt_bnd, self.u_bnd_sol, self.xt_init, self.u_init_sol, self.xt_fin, self.u_fin_sol = ic_bc_data()
            #    self.iter_tmp = 0
        
        if self.instant_loss < self.best_loss:
            self.nn.save_weights("./save_weights/best.h5")
            print(f'Best model saved: iter={self.iter_count}')
            self.best_loss = self.instant_loss
            
        if self.iter_count % self.resample_period == 0:
            self.xt_col, self.xt_bnd, self.u_bnd_sol, self.xt_init, self.u_init_sol, self.xt_fin, self.u_fin_sol = ic_bc_data()
            
        self.iter_count += 1


    def train_with_adam(self, adam_num, lr=0.01):
        
        self.lr = lr
        self.opt = Adam(self.lr)

        def learn():
            loss, grads = self.compute_grad()

            self.opt.apply_gradients(zip(grads, self.nn.trainable_variables))

            return loss

        for iter in range(int(adam_num)):

            loss = learn()

            self.instant_loss = loss.numpy()
            self.callback()


    def train_with_lbfgs(self, lbfgs_num):

        def vec_weight():
            # vectorize weights
            weight_vec = []

            # Loop over all weights
            for v in self.nn.trainable_variables:
                weight_vec.extend(v.numpy().flatten())

            weight_vec = tf.convert_to_tensor(weight_vec)
            return weight_vec
        w0 = vec_weight().numpy()

        def restore_weight(weight_vec):
            # restore weight vector to model weights
            idx = 0
            for v in self.nn.trainable_variables:
                vs = v.shape

                # weight matrices
                if len(vs) == 2:
                    sw = vs[0] * vs[1]
                    updated_val = tf.reshape(weight_vec[idx:idx + sw], (vs[0], vs[1]))
                    idx += sw

                # bias vectors
                elif len(vs) == 1:
                    updated_val = weight_vec[idx:idx + vs[0]]
                    idx += vs[0]

                # assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(updated_val, dtype=tf.float32))


        def loss_grad(w):
            # update weights in model
            restore_weight(w)
            loss, grads = self.compute_grad()
            # vectorize gradients
            grad_vec = []
            for g in grads:
                grad_vec.extend(g.numpy().flatten())

            # gradient list to array
            # scipy-routines requires 64-bit floats
            loss = loss.numpy().astype(np.float64)
            self.instant_loss = loss
            grad_vec = np.array(grad_vec, dtype=np.float64)

            return loss, grad_vec

        return scipy.optimize.minimize(fun=loss_grad,
                                       x0=w0,
                                       jac=True,
                                       method='L-BFGS-B',
                                       #method='BFGS',
                                       callback=self.callback,
                                       options={'maxiter': lbfgs_num,
                                                'maxfun': 10000,
                                                'maxcor': 200,
                                                'maxls': 200,
                                                'gtol': np.nan,#1.0 * np.finfo(float).eps,#np.nan,
                                                'ftol': np.nan})#1.0 * np.finfo(float).eps})#np.nan})

    def train_with_bfgs(self, bfgs_num):

        def vec_weight():
            # vectorize weights
            weight_vec = []

            # Loop over all weights
            for v in self.nn.trainable_variables:
                weight_vec.extend(v.numpy().flatten())

            weight_vec = tf.convert_to_tensor(weight_vec)
            return weight_vec
        w0 = vec_weight().numpy()

        def restore_weight(weight_vec):
            # restore weight vector to model weights
            idx = 0
            for v in self.nn.trainable_variables:
                vs = v.shape

                # weight matrices
                if len(vs) == 2:
                    sw = vs[0] * vs[1]
                    updated_val = tf.reshape(weight_vec[idx:idx + sw], (vs[0], vs[1]))
                    idx += sw

                # bias vectors
                elif len(vs) == 1:
                    updated_val = weight_vec[idx:idx + vs[0]]
                    idx += vs[0]

                # assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(updated_val, dtype=tf.float32))


        def loss_grad(w):
            # update weights in model
            restore_weight(w)
            loss, grads = self.compute_grad()
            # vectorize gradients
            grad_vec = []
            for g in grads:
                grad_vec.extend(g.numpy().flatten())

            # gradient list to array
            # scipy-routines requires 64-bit floats
            loss = loss.numpy().astype(np.float64)
            self.instant_loss = loss
            grad_vec = np.array(grad_vec, dtype=np.float64)

            return loss, grad_vec

        return scipy.optimize.minimize(fun=loss_grad,
                                       x0=w0,
                                       jac=True,
                                       method='BFGS',
                                       callback=self.callback,
                                       options={'maxiter': bfgs_num,
                                                'gtol': 0.})


    def predict(self, xt):
        u_pred = self.nn(xt)
        return u_pred


    def train(self, adam_num, bfgs_num, lbfgs_num, weights=[1, 1, 1, 1], repeat=3, plot=True):

        self.xt_col, self.xt_bnd, self.u_bnd_sol, self.xt_init, self.u_init_sol, self.xt_fin, self.u_fin_sol = ic_bc_data()
        self.weights = weights

        # Start timer
        for _ in range(repeat):
            if adam_num > 0:
                t0 = time()
                self.train_with_adam(adam_num)
                # Print computation time
                print('\nComputation time of adam: {} seconds'.format(time() - t0))
            if bfgs_num > 0:
                t1 = time()
                self.train_with_lbfgs(bfgs_num)
                # Print computation time
                print('\nComputation time of BFGS: {} seconds'.format(time() - t1))       
            if lbfgs_num > 0:
                t2 = time()
                self.train_with_lbfgs(lbfgs_num)
                # Print computation time
                print('\nComputation time of L-BFGS-B: {} seconds'.format(time() - t2))    

        self.save_weights("./save_weights/")

        np.savetxt('./save_weights/loss.txt', self.train_loss_history)
        train_loss_history = np.array(self.train_loss_history)
        self.nn.load_weights("./save_weights/best.h5")
        
        if plot:
            labels = ['weighted total', 'pde loss', 'bc loss', 'ic loss']
            for i, lb in enumerate(labels):
                plt.plot(train_loss_history[:, 0], train_loss_history[:, i + 1])
            plt.yscale("log")
            plt.show()


if __name__=="__main__":

    adam_num = 0
    bfgs_num = 0
    lbfgs_num = 1000
    agent = TransportPinn()
    
    agent.train(adam_num, bfgs_num, 50, weights=[0, 1, 1, 1], repeat=1, plot=False)
    agent.best_loss = np.inf
    agent.train(adam_num, bfgs_num, lbfgs_num, weights=[1000, 1, 1, 1], repeat=5)
    
    
    # Plot results
    times = np.array([-0.1, 0.0, 0.1, 0.5, 1])
    cs = ['b', 'r', 'g']
    nr = 20
    xref = np.linspace(0, 1, nr)
    fig, axs = plt.subplots(1, len(times) + 1, figsize=(12, 2), sharey=True)
    xs = [np.zeros([nr, 2]) for i in range(len(times))]
    for i in range(len(times)):
        xs[i][:, 0] = xref
        xs[i][:, 1] = times[i]
    ys = [agent.nn.predict(x) for x in xs]
    
    for i in range(len(ys)):
        for j in range(n_output):
            axs[i].plot(xref, ys[i][:, j], color=cs[j])
        if times[i] == 0:
            axs[i].plot(xref, ic_func(xref), 'k--')
        elif times[i] == 1:
            axs[i].plot(xref, fc_func(xref), 'k--')
        axs[i].set_title(f't={times[i]}')
        axs[i].set_xlabel('rhon')

    ts = np.linspace(-0.1, 1, 101)
    xs1 = np.zeros([len(ts), 2])
    xs1[:, 1] = ts
    ys1 = agent.nn.predict(xs1)
    for j in range(n_output):    
        axs[-1].plot(ts, ys1[:, j], color=cs[j])
        #axs[-1].plot(times, [u[j, i, 0] for i in its], color=cs[j], ls='--')
    axs[-1].set_title('Evolution')
    axs[-1].set_ylim([np.min(ys), 1.25 * np.max(ys)])
    axs[-1].set_xlabel('Time')
    plt.show()

