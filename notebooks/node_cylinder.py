#!/usr/bin/env python

"""
Script for running NODE for flow around a cylinder
"""

### Loading modules
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import scipy
import os
import gc
import argparse
import ipdb

import platform
print("Python "+str(platform.python_version()))

import tensorflow as tf
print("Tensorflow "+ str(tf.__version__))
if tf.__version__ == '1.15.0':
    tf.compat.v1.enable_eager_execution()
elif tf.__version__.split('.')[0] == 2: # in ['2.2.0','2.3.0']:
    print("Setting Keras backend datatype")
    tf.keras.backend.set_floatx('float64')

from tfdiffeq import odeint,odeint_adjoint
from tfdiffeq.models import ODENet
# from tfdiffeq.bfgs_optimizer import BFGSOptimizer
from tfdiffeq.adjoint import odeint as adjoint_odeint
from tfdiffeq import plot_phase_portrait, plot_vector_field, plot_results
tf.keras.backend.set_floatx('float64')
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# tf.debugging.set_log_device_placement(True)
np.random.seed(0)

basedir   = os.getcwd()
podsrcdir = os.path.join(basedir,'../src/podrbf/')
workdir   = os.path.join(basedir,'../notebooks/')
datadir   = os.path.join(basedir,'../data/')
figdir    = os.path.join(basedir,'../figures')
nodedir   = os.path.join(basedir,'../best_models/CYLINDER/')

modeldir = basedir
savedir = nodedir
os.chdir(workdir)


# Options
## --- User specified runtime input arguments ---
parser = argparse.ArgumentParser(description=
                                  'Runs NODE for cylinder example')
parser.add_argument('-adjoint', action='store_true', help='using adjoint method (default=False)')
parser.add_argument('-epochs', default=50000, help='Number of training epochs (default=50,000)')
parser.add_argument('-solver', default='rk4', action='store', type=str, help='ODE solver to use (default=rk4)')
parser.add_argument('-mode', default='eval', action='store', type=str, help='Mode of execution: train, retrain, eval (default=eval)')
parser.add_argument('-scale', action='store_true', help='scale input features (default=False)')
parser.add_argument('-aug', action='store_true', help='using augmented NODE (default=False)')
parser.add_argument('-act', default='tanh', action='store', type=str, help='NN activation function to use (default=tanh)')
parser.add_argument('-nl', default=1, help='Number of network layers (default=1)')
parser.add_argument('-nn', default=256, help='Number of neurons per layer (default=256)')
parser.add_argument('-stk', default='v_x,v_y,p', action='store', type=str, help='Stacking order in latent space (default=v_x,v_y,p)')
parser.add_argument('-lr', default=0.001, help='Initial learning rate (default=0.001)')
parser.add_argument('-lr_steps', default=5001, help='Number of steps for learning rate decay (default=5001)')
parser.add_argument('-lr_rate', default=0.5, help='Rate of learning rate decay (default=0.5)')
parser.add_argument('-dr1', default=3851463, help='Batch ID or parent directory of pretrained model (default=3849292)')
parser.add_argument('-dr2', default=7, help='Run ID or save directory of pretrained model (default=1)')
args = parser.parse_args()


device = 'cpu:0' # select gpu:# or cpu:#
purpose= args.mode #Write 'train' to train a new model and 'eval' to load a pre-trained model for evaluation (make sure you have the correct set of hyperparameters)
pre_trained_dir = savedir+str(args.dr1)+'_'+str(args.dr2)+'/model_weights_cyl/' #If 'eval' specify path for pretrained model
stacking = True #stack or not
stack_order = args.stk #'v_x,v_y,p' #If stacking = True decide the stacking order
scale_time = False #Scale time or not (Normalize)
scale_states = args.scale #Scale states or not (MinMax -1,1)
augmented,aug_dims = (args.aug,5)#Augmented or not and #of dimensions to augment
N_layers = int(args.nl) #Only three layers supported because it's hard coded. I will replace this with a function int he future.
N_neurons = int(args.nn) #Number of neurons per layer
act_f = args.act  #Activation Function ('linear', 'tanh', 'sigmoid',...), default='linear'
learning_rate_decay = True #Use decaying learning rate or not
initial_learning_rate = float(args.lr) #0.001 #If 'learning_rate_decay = False' then this will be the learning rate
decay_steps = int(args.lr_steps) #5001 #Number of decay steps
decay_rate = float(args.lr_rate) #0.5 #Decay rate for number of decay steps
staircase_opt = True #True for staircase decay and False for exponential
optimizer = 'RMSprop' #Adam and RMSprop optimizer only (this can be modified)
adjoint = args.adjoint #False #Use adjoint method or not
solver = args.solver #'dopri5'#Determine solver based on tfdiffeq options
minibatch, batch_size = (False,256) #Use minibatch or not and batch size
epochs = int(args.epochs) #100 #Number of epochs to train on
bfgs = False #Use bfgs optimizer to further fine tune reuslts after training or not (crashes with more than 64 neurons per layer)

print("\n***** Runtime parameters: ******\n")
print(f'Mode = {purpose}, Scaling = {scale_states}, Augmenting = {augmented}, Adjoint = {adjoint}')
print(f'Solver = {solver}, Optimizer = {optimizer}, Stacking order = {stack_order}, Epochs = {epochs}')
print(f'# Layers = {N_layers}, # Neurons per layer = {N_neurons}, Activation fn = {act_f}')
print(f'Init LR = {initial_learning_rate}, # LR decay steps = {decay_steps}, LR decay rate = {decay_rate}')
print('**********************************\n')

### ------ Import Snapshot data -------------------
data = np.load(datadir + 'cylinder_Re100.0_Nn14605_Nt3001.npz')
mesh = np.load(datadir + 'OF_cylinder_mesh_Nn14605_Ne28624.npz')


## ------- Prepare training snapshots ----------------
print('-------Prepare training and testing data---------')
soln_names = ['p', 'v_x', 'v_y']
nodes = mesh['nodes'];  node_ind = mesh['node_ind']
triangles = mesh['elems']; elem_ind = mesh['elem_ind']
snap_start = 1250
T_end = 5.0   ### 5 seconds
snap_data = {}
for key in soln_names:
    snap_data[key] = data[key][:,snap_start:]

times_offline = data['time'][snap_start:]
print('Loaded {0} snapshots of dimension {1} for h,u and v, spanning times [{2}, {3}]'.format(
                    snap_data[soln_names[0]].shape[1],snap_data[soln_names[0]].shape[0],
                    times_offline[0], times_offline[-1]))
## number of steps to skip in selecting training snapshots for SVD basis
snap_incr=4
## Subsample snapshots for building POD basis
snap_end = np.count_nonzero(times_offline[times_offline <= T_end])
snap_train = {};
for key in soln_names:
    snap_train[key] = snap_data[key][:,0:snap_end+1:snap_incr]
times_train=times_offline[0:snap_end+1:snap_incr]
print('Using {0} training snapshots for time interval [{1},{2}]'.format(times_train.shape[0],
                                        times_train[0], times_train[-1]))

del data
del mesh
gc.collect()


### --- Some utility functions for POD latent space  calculations

def compute_pod_multicomponent(S_pod,subtract_mean=True,subtract_initial=False,full_matrices=False):
    """
    Compute standard SVD [Phi,Sigma,W] for all variables stored in dictionary S_til
     where S_til[key] = Phi . Sigma . W is an M[key] by N[key] array
    Input:
    :param: S_pod -- dictionary of snapshots
    :param: subtract_mean -- remove mean or not
    :param: full_matrices -- return Phi and W as (M,M) and (N,N) [True] or (M,min(M,N)) and (min(M,N),N)

    Returns:
    S      : perturbed snapshots if requested, otherwise shallow copy of S_pod
    S_mean : mean of the snapshots
    Phi : left basis vector array
    sigma : singular values
    W   : right basis vectors

    """
    S_mean,S = {},{}
    Phi,sigma,W = {},{},{}

    for key in S_pod.keys():
        if subtract_mean:
            S_mean[key] = np.mean(S_pod[key],1)
            S[key] = S_pod[key].copy()
            S[key]-= np.tile(S_mean[key],(S_pod[key].shape[1],1)).T
            Phi[key],sigma[key],W[key] = scipy.linalg.svd(S[key][:,1:],full_matrices=full_matrices)
        elif subtract_initial:
            S_mean[key] = S_pod[key][:,0]
            S[key] = S_pod[key].copy()
            S[key]-= np.tile(S_mean[key],(S_pod[key].shape[1],1)).T
            Phi[key],sigma[key],W[key] = scipy.linalg.svd(S[key][:,:],full_matrices=full_matrices)
        else:
            S_mean[key] = np.mean(S_pod[key],1)
            S[key] = S_pod[key]
            Phi[key],sigma[key],W[key] = scipy.linalg.svd(S[key][:,:],full_matrices=full_matrices)

    return S,S_mean,Phi,sigma,W


def compute_trunc_basis(D,U,eng_cap = 0.999999):
    """
    Compute the number of modes and truncated basis to use based on getting 99.9999% of the 'energy'
    Input:
    D -- dictionary of singular values for each system component
    U -- dictionary of left singular basis vector arrays
    eng_cap -- fraction of energy to be captured by truncation
    Output:
    nw -- list of number of truncated modes for each component
    U_r -- truncated left basis vector array as a list (indexed in order of dictionary keys in D)
    """

    nw = {}
    for key in D.keys():
        nw[key] = 0
        total_energy = (D[key]**2).sum(); assert total_energy > 0.
        energy = 0.
        while energy/total_energy < eng_cap and nw[key] < D[key].shape[0]-2:
            nw[key] += 1
            energy = (D[key][:nw[key]]**2).sum()
        print('{3} truncation level for {4}% = {0}, \sigma_{1} = {2}'.format(nw[key],nw[key]+1,
                                                        D[key][nw[key]+1],key,eng_cap*100) )

    U_r = {}
    for key in D.keys():
        U_r[key] = U[key][:,:nw[key]]

    return nw, U_r


def project_onto_basis(S,Phi,S_mean,msg=False):
    """
    Convenience function for computing projection of values in high-dimensional space onto
    Orthonormal basis stored in Phi.
    Only projects entries that are in both. Assumes these have compatible dimensions

    Input:
    S -- Dict of High-dimensional snapshots for each component
    Phi -- Dict of POD basis vectors for each component
    S_mean -- Dict of temporal mean for each component
    Output:
    Z -- Dict of modal coefficients for POD-projected snapshots
    """
    soln_names = S.keys()
    S_normalized = {}; Z = {}
    for key in soln_names:
        S_normalized[key] = S[key].copy()
        S_normalized[key] -= np.outer(S_mean[key],np.ones(S[key].shape[1]))
        Z[key] = np.dot(Phi[key].T, S_normalized[key])
        if msg:
            print('{0} projected snapshot matrix size: {1}'.format(key,Z[key].shape))

    return Z


def reconstruct_from_rom(Zpred,Phi,S_mean,nw,msg=False):
    """
    Convenience function for computing projection of values in high-dimensional space onto
    Orthonormal basis stored in Phi.
    Only projects entries that are in both. Assumes these have compatible dimensions

    Input:
    S -- Dict of High-dimensional snapshots for each component
    Phi -- Dict of POD basis vectors for each component
    S_mean -- Dict of temporal mean for each component
    Output:
    Z -- Dict of modal coefficients for POD-projected snapshots
    """
    soln_names = nw.keys()
    S = {};
    ctr= 0
    for key in soln_names:
        S[key] = np.dot(Phi[key],Zpred[key]) + np.outer(S_mean[key],np.ones(Zpred[key].shape[1]))

    return S


### ------ Compute the POD coefficients ------------------
# trunc_lvl = 0.9999995
trunc_lvl = 0.99
snap_norm, snap_mean, U, D, W = compute_pod_multicomponent(snap_train)
nw, U_r = compute_trunc_basis(D, U, eng_cap = trunc_lvl)
Z_train = project_onto_basis(snap_train, U_r, snap_mean)

## Coefficients of training and true prediction snapshots
npod_total = 0
for key in soln_names:
    npod_total+=nw[key]

pred_incr = snap_incr -3
pred_end = -1
snap_pred_true = {};
for key in soln_names:
    snap_pred_true[key] = snap_data[key][:,0:pred_end:pred_incr]
times_predict = times_offline[0:pred_end:pred_incr]
Z_pred_true = project_onto_basis(snap_pred_true, U_r, snap_mean)

true_state_array = np.zeros((times_train.size,npod_total));
true_pred_state_array = np.zeros((times_predict.size, npod_total));
#init_state = true_state_array[0,:]

ctr=0
stack = stack_order.split(',')
for key in stack:
    true_state_array[:,ctr:ctr+nw[key]] = Z_train[key].T
    true_pred_state_array[:,ctr:ctr+nw[key]] = Z_pred_true[key].T
    ctr+=nw[key]

init_state = true_state_array[0,:]
tsteps = np.shape(true_state_array)[0]
state_len = np.shape(true_state_array)[1]
dt_train = (times_train[-1]-times_train[0])/(tsteps-1)
dt_predict = (times_predict[-1]-times_predict[0])/(times_predict.size)
T0 = times_train[0]
print("Training using %d modes for %d time steps with t = {%.4f, %.4f} and dt = %.4f"%(state_len,
                                                tsteps, times_train[0], times_train[-1], dt_train))
print("Predicting using %d modes for %d time steps with t = {%.4f, %.4f} and dt = %.4f"%(state_len,
                                            times_predict.size, times_predict[0], times_predict[-1], dt_predict))



if scale_time == True:
    scale_time = np.amax(times_train)
    times_train = times_train/scale_time


if scale_states == True:
    #scale_mm = MinMaxScaler()  ## Scales to [0,1] for every mode
    #scale_mm = StandardScaler() ## Scales to approx. [-1,1] for every mode
    #scale_mm.fit(true_state_array)
    #true_state_array = scale_mm.transform(true_state_array)

    ## Scale entire vector to [-1,1]^d
    #max_g = true_state_array.max(); min_g = true_state_array.min()
    ## Scale each element between [-1,1]
    max_g = np.amax(true_state_array,axis=0); min_g = np.amin(true_state_array,axis=0)
    scaler = lambda x: (2*(x - min_g)/(max_g - min_g) - 1)
    true_state_array = scaler(true_state_array)

if augmented == True:
    augment_zeros = np.zeros((true_state_array.shape[0],aug_dims))
    true_state_tensor = tf.convert_to_tensor(np.hstack((true_state_array, augment_zeros)))
    times_tensor = tf.convert_to_tensor(times_train)
    init_state = tf.convert_to_tensor(true_state_tensor[0,:],)
elif augmented == False:
    true_state_tensor = tf.convert_to_tensor(true_state_array)
    times_tensor = tf.convert_to_tensor(times_train)
    init_state = true_state_tensor[0,:]
    aug_dims = 0

if learning_rate_decay == True:
    learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps,
                                                    decay_rate, staircase=staircase_opt)
elif learning_rate_decay == False:
    learn_rate = initial_learning_rate
if optimizer == 'Adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate = learn_rate)
elif optimizer == 'RMSprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = learn_rate, momentum = 0.9)



### ------- Define NN and ODE integrator-----------------
# Define NN and ODE integrator

class NN(tf.keras.Model):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if N_layers == 1:

            self.eqn = tf.keras.Sequential([tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros',
                                           input_shape=(state_len+aug_dims,)),
                     tf.keras.layers.Dense(state_len+aug_dims)])

        elif N_layers == 2:

            self.eqn = tf.keras.Sequential([tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros',
                                           input_shape=(state_len+aug_dims,)),
                                           tf.keras.layers.Dense(N_neurons, activation='linear',
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros'),
                     tf.keras.layers.Dense(state_len+aug_dims)])

        elif N_layers == 3:

            self.eqn = tf.keras.Sequential([tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros',
                                           input_shape=(state_len+aug_dims,)),
                                           tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros'),
                                           tf.keras.layers.Dense(N_neurons, activation='linear',
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros'),
                     tf.keras.layers.Dense(state_len+aug_dims)])

        elif N_layers == 4:
            self.eqn =  tf.keras.Sequential([tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros',
                                           input_shape=(state_len+aug_dims,)),
                                           tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros'),
                                           tf.keras.layers.Dense(N_neurons, activation=act_f,
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros'),
                                           tf.keras.layers.Dense(N_neurons, activation='linear',
                                           kernel_initializer = tf.keras.initializers.glorot_uniform(),
                                           bias_initializer='zeros'),
                     tf.keras.layers.Dense(state_len+aug_dims)])

    @tf.function
    def call(self, t, y):
        # Neural ODE component
        i0 = self.eqn(y)

        return i0



### -------- Model Training Loop ---------------------
print('\n------------Begin training---------')
train_loss_results = []
bfgs_loss = []
train_lr = []
saved_ep = []
start_time = time.time()

if adjoint == True:
    int_ode = adjoint_odeint
elif adjoint == False:
    int_ode = odeint

if purpose == 'train':
    if not os.path.exists(savedir+'/current/model_weights_cyl/'):
        os.makedirs(savedir+'/current/model_weights_cyl/')

    if minibatch == True:
        dataset = tf.data.Dataset.from_tensor_slices((true_state_tensor, times_tensor))
        dataset = dataset.batch(128)
        with tf.device(device):
            model = NN()
            for epoch in range(epochs):
                datagen = iter(dataset)
                avg_loss = tf.keras.metrics.Mean()
                for batch, (true_state_trainer, times_trainer) in enumerate(datagen):
                    with tf.GradientTape() as tape:
                        preds = int_ode(model, tf.expand_dims(init_state, axis=0), times_trainer, method=solver)
                        loss = tf.math.reduce_mean(tf.math.square(true_state_trainer - tf.squeeze(preds)))
                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    avg_loss(loss)

                train_loss_results.append(avg_loss.result().numpy())
                print("Epoch %d: Loss = %0.6f" % (epoch + 1, avg_loss.result().numpy()))
                print()

    elif minibatch == False:
        with tf.device(device):
            model = NN()
            print()
            for epoch in range(epochs):

                with tf.GradientTape() as tape:
                    preds = int_ode(model, tf.expand_dims(init_state, axis=0), times_tensor, method=solver)
                    loss = tf.math.reduce_mean(tf.math.square(true_state_tensor - tf.squeeze(preds)))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss_results.append(loss.numpy())
                print("Epoch {0}: Loss = {1:0.6f}, LR = {2:0.6f}".format(epoch+1, loss.numpy(), learn_rate(optimizer.iterations).numpy()))
                print()
                if (epoch+1)%(epochs//4) == 0:
                    print("******Saving model state. Epoch {0}******\n".format(epoch + 1))
                    model.save_weights(savedir+'current/model_weights_cyl/ckpt', save_format='tf')
                    if learning_rate_decay:
                        train_lr.append(learn_rate(optimizer.iterations).numpy())
                    else:
                        train_lr.append(learn_rate)
                    saved_ep.append(epoch+1)
                    np.savez_compressed(savedir+'current/model_weights_cyl/train_lr', lr=train_lr, ep=saved_ep)

    if bfgs == True:
        tolerance = 1e-6
        bfgs_optimizer = BFGSOptimizer(max_iterations=50, tolerance=tolerance)
        def loss_wrapper(model):
            preds = int_ode(model, tf.expand_dims(init_state, axis=0), times_tensor,
                                            atol=1e-6, rtol=1e-6, method=solver)
            loss = tf.math.reduce_mean(tf.math.square(true_state_tensor - tf.squeeze(preds)))
            bfgs_loss.append(loss.numpy())

            return loss
        model = bfgs_optimizer.minimize(loss_wrapper, model)

    model.save_weights(savedir+'current/model_weights_cyl/ckpt', save_format='tf')
    if learning_rate_decay:
        train_lr.append(learn_rate(optimizer.iterations).numpy())
    else:
        train_lr.append(learn_rate)
    saved_ep.append(epoch+1)
    np.savez_compressed(savedir+'current/model_weights_cyl/train_lr', lr=train_lr, ep=saved_ep)
    end_time = time.time()
    print("****Total training time = {0}****\n".format(end_time - start_time))

elif purpose == 'retrain':

    saved_lr = np.load(pre_trained_dir+'train_lr.npz')
    initial_learning_rate = saved_lr['lr'][-1]
    ep = saved_lr['ep'][-1]
    print("Initial lr = {0}".format(initial_learning_rate))
    if not os.path.exists(savedir+'/current/model_weights_cyl/'):
        os.makedirs(savedir+'/current/model_weights_cyl/')

    if learning_rate_decay == True:
        learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps,
                                                    decay_rate, staircase=staircase_opt)
    elif learning_rate_decay == False:
        learn_rate = initial_learning_rate

    if optimizer == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate = learn_rate)
    elif optimizer == 'RMSprop':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate = learn_rate, momentum = 0.9)


    if minibatch == True:
        dataset = tf.data.Dataset.from_tensor_slices((true_state_tensor, times_tensor))
        dataset = dataset.batch(128)

        with tf.device(device):
            model = NN()
            print()
            model.load_weights(pre_trained_dir+'ckpt')

            for epoch in range(epochs):
                datagen = iter(dataset)
                avg_loss = tf.keras.metrics.Mean()
                for batch, (true_state_trainer, times_trainer) in enumerate(datagen):
                    with tf.GradientTape() as tape:
                        preds = int_ode(model, tf.expand_dims(init_state, axis=0), times_trainer, method=solver)
                        loss = tf.math.reduce_mean(tf.math.square(true_state_trainer - tf.squeeze(preds)))

                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                    avg_loss(loss)

                train_loss_results.append(avg_loss.result().numpy())
                print("Epoch %d: Loss = %0.6f, LR = %0.6f" %(epoch + 1, avg_loss.result().numpy(), learn_rate(optimizer.iterations).numpy()))
                print()

    elif minibatch == False:

        with tf.device(device):
            model = NN()
            model.load_weights(pre_trained_dir+'ckpt')
            for epoch in range(epochs):

                with tf.GradientTape() as tape:
                    preds = int_ode(model, tf.expand_dims(init_state, axis=0), times_tensor, method=solver)
                    loss = tf.math.reduce_mean(tf.math.square(true_state_tensor - tf.squeeze(preds)))

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss_results.append(loss.numpy())
                print("Epoch %d: Loss = %0.6f, LR = %0.6f" %(ep+epoch+1, loss.numpy(), learn_rate(optimizer.iterations).numpy()))
                print()
                if (epoch+1)%(epochs//4) == 0:
                    print("Saving model state. Epoch {0}\n".format(epoch + ep + 1))
                    model.save_weights(savedir+'current/model_weights_cyl/ckpt', save_format='tf')
                    if learning_rate_decay:
                        train_lr.append(learn_rate(optimizer.iterations).numpy())
                    else:
                        train_lr.append(learn_rate)
                    saved_ep.append(epoch+ep+1)
                    np.savez_compressed(savedir+'current/model_weights_cyl/train_lr', lr=train_lr, ep=saved_ep)


    if bfgs == True:
        tolerance = 1e-6
        bfgs_optimizer = BFGSOptimizer(max_iterations=50, tolerance=tolerance)

        def loss_wrapper(model):
            preds = int_ode(model, tf.expand_dims(init_state, axis=0), times_tensor, atol=1e-6, rtol=1e-6, method=solver)
            loss = tf.math.reduce_mean(tf.math.square(true_state_tensor - tf.squeeze(preds)))
            bfgs_loss.append(loss.numpy())

            return loss

        model = bfgs_optimizer.minimize(loss_wrapper, model)

    end_time = time.time()
    print("****Total training time = {0}****\n".format(end_time - start_time))

    model.save_weights(savedir+'current/model_weights_cyl/ckpt', save_format='tf')
    if learning_rate_decay:
        train_lr.append(learn_rate(optimizer.iterations).numpy())
    else:
        train_lr.append(learn_rate)
    saved_ep.append(epoch+ep+1)
    np.savez_compressed(savedir+'current/model_weights_cyl/train_lr', lr=train_lr, ep=saved_ep)


elif purpose == 'eval':
    model = NN()
    model.load_weights(pre_trained_dir+'ckpt')



### ----- Predict using trained model ---------------
if scale_time == True:
    times_predict = times_predict/scale_time

if adjoint == True:
    predicted_states = adjoint_odeint(model, tf.expand_dims(init_state, axis=0),
                                        tf.convert_to_tensor(times_predict), method=solver)
    predicted_states = tf.squeeze(predicted_states)
    if augmented == True:
        predicted_states = np.delete(predicted_states,slice(state_len,state_len+aug_dims),axis=1)

elif adjoint == False:
    predicted_states = odeint(model, tf.expand_dims(init_state, axis=0),
                                tf.convert_to_tensor(times_predict), method=solver)
    predicted_states = tf.squeeze(predicted_states)
    if augmented == True:
        predicted_states = np.delete(predicted_states,slice(state_len,state_len+aug_dims),axis=1)


### ---- Post-process predicted states ---------------
if scale_states == True:
    inverse_scaler = lambda z: ((z + 1)*(max_g - min_g)/2 + min_g)
    predicted_states = inverse_scaler(predicted_states)
    true_state_array = inverse_scaler(true_state_array)
    #predicted_states = scale_mm.inverse_transform(predicted_states)

if scale_time == True:
    times_predict = times_predict*scale_time


### ----- Visualize true and predicted POD coefficients -------
viz = False

if viz:
    comp = 0
    # true_state_array = np.load(datadir+'NS_Coefficients_pred_true.npz')['true']

    # Visualization fluff here
    fig, ax = plt.subplots(nrows=3,ncols=1,figsize=(8,15))
    mnum = comp
    for i, key in enumerate(soln_names):
        tt = ax[i].plot(times_predict[:],true_pred_state_array[:,mnum],label='True',marker='o',markevery=20)
        # Visualization of modal evolution using NODE
        ln, = ax[i].plot(times_predict[:],predicted_states[:,mnum],label='NODE',color='orange',marker='D',markevery=25)
        mnum = mnum + nw[key]

        ax[i].set_xlabel('Time')
        sv = str(key)+':'+str(comp)
        ax[i].set_ylabel(sv,fontsize=18)
        ax[i].legend(fontsize=14)



#### ----- Error computations -----------
Z_pred = {}
ctr= 0
for key in stack:
    Z_pred[key] = np.array(predicted_states)[:,ctr:ctr+nw[key]].T
    ctr += nw[key]
urom = reconstruct_from_rom(Z_pred, U_r, snap_mean, nw)

error_p = np.mean(np.square(urom['p']-snap_pred_true['p']))
error_vx = np.mean(np.square(urom['v_x']-snap_pred_true['v_x']))
error_vy = np.mean(np.square(urom['v_y']-snap_pred_true['v_y']))

print('Pr MSE: ' + str(error_p))
print('Vx MSE: ' + str(error_vx))
print('Vy MSE: ' + str(error_vy))


#### ----- Save predicted solutions -------
os.chdir(nodedir+'/current')
print("Saving results in %s"%(os.getcwd()))

np.savez_compressed('cylinder_online_node', p=urom['p'],v_x=urom['v_x'], v_y=urom['v_y'],time=times_predict,loss=train_loss_results)
