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
nodedir   = os.path.join(basedir,'../best_models/')


# Options
## --- User specified runtime input arguments ---
parser = argparse.ArgumentParser(description=
                                  'Runs NODE for shallow water examples')
parser.add_argument('-adjoint', action='store_true', help='using adjoint method (default=False)')
parser.add_argument('-epochs', default=100, help='Number of training epochs')
parser.add_argument('-solver', default='dopri5', action='store', type=str, help='ODE solver to use.')
parser.add_argument('-mode', default='eval', action='store', type=str, help='Mode of execution: train, retrain, eval (default=eval)')
parser.add_argument('-stk', default='S_dep,S_vx,S_vy', action='store', type=str, help='Stacking order to use.')
parser.add_argument('-aug', action='store_true', help='using augmented node (ANODE) (default=False)')
parser.add_argument('-act', default='linear', action='store', type=str, help='Activation function.')
parser.add_argument('-lrs', action='store_true', help='Use learning rate scheduler (default=False)')
parser.add_argument('-lr', default=0.001, help='Initial Learning Rate')
parser.add_argument('-lr_steps', default=5001, action='store', type=str, help='Decay steps')
parser.add_argument('-lr_rate', default=0.5, action='store', type=str, help='Decay rate')
parser.add_argument('-optimus_prime', default='RMSprop', action='store', type=str, help='Optimizer')
parser.add_argument('-minibatch', action='store_true', help='using minibatch method (default=False)')
parser.add_argument('-batch_size', default=64, help='Batch Size')
parser.add_argument('-nl', default=1, help='Number of layers, only 1-3')
parser.add_argument('-nn', default=256, help='Number of neurons per layer')
parser.add_argument('-scale_time', action='store_true', help='Scale time or not (default=False)')
parser.add_argument('-scale_states', action='store_true', help='Scale states or not (default=False)')
parser.add_argument('-sw_model', default='SD', action='store', type=str, help='SW model: Choose between "RED" and "SD" (default)')

args = parser.parse_args()

device = 'cpu:0' # select gpu:# or cpu:#
purpose= args.mode #Write 'train' to train a new model, 'retrain' to retrain a model and 'eval' to load a pre-trained model for evaluation (make sure you have the correct set of hyperparameters)
pre_trained = nodedir+args.sw_model+'/model_weights/' #If 'Evaluate' specify path for pretrained model
stack_order = args.stk #'S_dep,S_vx,S_vy'
scale_time = args.scale_time #Scale time or not (Normalize)
scale_states = args.scale_states #Scale states or not (MinMax -1,1)
augmented,aug_dims = (args.aug,10)#Augmented or not and #of dimensions to augment
N_layers = int(args.nl) #1 #Only three layers supported because it's hard coded. I will replace this with a function int he future.
N_neurons = int(args.nn) #256 #Number of neurons per layer
act_f = args.act  #Activation Function ('linear', 'tanh', 'sigmoid',...)
learning_rate_decay = True #args.lrs #Use decaying learning rate or not
initial_learning_rate = float(args.lr) #float(args.ilr) #If 'learning_rate_decay = False' then this will be the learning rate
decay_steps = int(args.lr_steps) #Number of decay steps
decay_rate = float(args.lr_rate) #Decay rate for number of decay steps
staircase_opt = True #True for staircase decay and False for exponential
optimizer = args.optimus_prime#'RMSprop' #Adam and RMSprop optimizer only (this can be modified)
adjoint = args.adjoint #False #Use adjoint method or not
solver = args.solver #Determine solver based on tfdiffeq options
minibatch, batch_size = (args.minibatch,int(args.batch_size)) #Use minibatch or not and batch size
epochs = int(args.epochs)  #Number of epochs to train on
bfgs = False #Use bfgs optimizer to further fine tune reuslts after training or not (crashes with more than 64 neurons per layer)
model_sw = args.sw_model # SW model to be loaded

nodedir  = nodedir+model_sw
modeldir = basedir
savedir  = nodedir
os.chdir(workdir)

print("\n***** Runtime parameters: ******\n")
print(f'Mode = {purpose}, Scaling states = {scale_states}, Scaling time = {scale_time}, Augmenting = {augmented}')
print(f'Solver = {solver}, Optimizer = {optimizer}, Stacking order = {stack_order}, Epochs = {epochs},  Adjoint = {adjoint}')
print(f'# Layers = {N_layers}, # Neurons per layer = {N_neurons}, Activation fn = {act_f}, Optimizer = {optimizer}')
print(f'Init LR = {initial_learning_rate}, # LR decay steps = {decay_steps}, LR decay rate = {decay_rate}')
print('**********************************\n')


if model_sw =='SD':
    data = np.load(datadir + 'san_diego_tide_snapshots_T4.32e5_nn6311_dt25.npz')
    mesh = np.load(datadir + 'san_diego_mesh.npz')
elif model_sw == 'RED':
    data = np.load(datadir + 'red_river_inset_snapshots_T7.0e4_nn12291_dt10.npz')
    mesh = np.load(datadir + 'red_river_mesh.npz')


## Prepare training snapshots
soln_names = ['S_dep', 'S_vx', 'S_vy']
nodes = mesh['nodes']
triangles = mesh['triangles']

snap_start = 100
if model_sw == 'SD':
    T_end = 50*3600   ### 50 hours in seconds
elif model_sw == 'RED':
    T_end = 3.24e4
snap_end = np.count_nonzero(data['T'][data['T'] <= T_end])

snap_data = {}
for key in soln_names:
    snap_data[key] = data[key][:,snap_start:snap_end+1]

times_offline = data['T'][snap_start:snap_end+1]
print('Loaded {0} snapshots of dimension {1} for h,u and v, spanning times [{2}, {3}]'.format(
                    snap_data[soln_names[0]].shape[1],snap_data[soln_names[0]].shape[0],
                    times_offline[0], times_offline[-1]))


## number of steps to skip in selecting training snapshots for SVD basis
if model_sw == 'SD':
    snap_incr=4
elif model_sw == 'RED':
    snap_incr=3
## Subsample snapshots for building POD basis
snap_train = {};
for key in soln_names:
    snap_train[key] = snap_data[key][:,::snap_incr]

times_train=times_offline[::snap_incr]
print('Using {0} training snapshots for time interval [{1},{2}]'.format(times_train.shape[0],
                                        times_train[0], times_train[-1]))

### Modules for computing POD basis
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
    S = {}; #Z = {}
    ctr= 0
    for key in soln_names:
        #Z[key] = Zpred.T[ctr:ctr+nw[key],:];
        S[key] = np.dot(Phi[key],Zpred[key]) + np.outer(S_mean[key],np.ones(Zpred[key].shape[1]))
    return S

## Compute the POD coefficients
if model_sw == 'SD':
    trunc_lvl = 0.9999995  ### NIROM value
    eps = 0.01
elif model_sw == 'RED':
#     trunc_lvl = 0.999995  ### NIROM value
    trunc_lvl = 0.99
    eps = 0.05

snap_norm, snap_mean, U, D, W = compute_pod_multicomponent(snap_train)
nw, U_r = compute_trunc_basis(D, U, eng_cap = trunc_lvl)
Z_train = project_onto_basis(snap_train, U_r, snap_mean)

## Save POD coefficients of true training snapshots
npod_total = 0
for key in soln_names:
    npod_total+=nw[key]

true_state_array = np.zeros((times_train.size,npod_total));

## Save POD coefficients of snapshots for prediction comparison
tsteps = np.shape(true_state_array)[0]
state_len = np.shape(true_state_array)[1]

batch_tsteps = 50  ## Length of sequence of time steps in each sample inside a mini batch
num_batches = 5   ## Number of samples in a mini batch or batch size
dt = (times_train[-1]-times_train[0])/(tsteps-1)
T0 = times_train[0]

# Time array - fixed
time_array = T0 + dt*np.arange(tsteps)
pred_incr = snap_incr - 2
pred_end = np.count_nonzero(times_offline[times_offline<=T_end])
times_predict = times_offline[0:pred_end:pred_incr]

print("Training using %d modes for %d time steps with t = {%.4f, %.4f} and dt = %.4f"%(state_len,
                                                            tsteps,time_array[0],time_array[-1],dt*snap_incr))
print("Predicting using %d modes for %d time steps with t = {%.4f, %.4f} and dt = %.4f"%(state_len,
                                            times_predict.size,times_predict[0],times_predict[-1],dt*pred_incr))

# DS definition
init_state = true_state_array[0,:]

snap_pred_true = {};
for key in soln_names:
    snap_pred_true[key] = snap_data[key][:,0:pred_end:pred_incr]

true_pred_state_array = np.zeros((times_predict.size,npod_total));
Z_pred_true = project_onto_basis(snap_pred_true, U_r, snap_mean)

ctr=0
stack = stack_order.split(',')
for key in stack:
    true_state_array[:,ctr:ctr+nw[key]] = Z_train[key].T
    true_pred_state_array[:,ctr:ctr+nw[key]] = Z_pred_true[key].T
    ctr+=nw[key]

# np.savez_compressed(datadir + 'SW_Coefficients_pred_true',true=true_pred_state_array)
# np.savez_compressed(datadir + 'SW_Coefficients_train',true=true_state_array)
if scale_time == True:
    time_scaler = np.amax(times_train)
    times_train = times_train/time_scaler

if scale_states == True:
    scale_mm = MaxAbsScaler()
    scale_mm.fit(true_state_array)
    true_state_array = scale_mm.transform(true_state_array)


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

if minibatch == True:
    decay_steps = decay_steps*np.floor(tsteps/batch_size)

if learning_rate_decay == True:
    learn_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate, decay_steps,
                                                    decay_rate, staircase=staircase_opt)
elif learning_rate_decay == False:
    learn_rate = initial_learning_rate

if optimizer == 'Adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate = learn_rate)
elif optimizer == 'RMSprop':
    optimizer = tf.keras.optimizers.RMSprop(learning_rate = learn_rate, momentum = 0.9)
elif optimizer == 'SGD':
    optimizer = tf.keras.optimizers.SGD(learning_rate = learn_rate)
elif optimizer == 'Adadelta':
    optimizer = tf.keras.optimizers.Adadelta(learning_rate = learn_rate)
elif optimizer == 'Adagrad':
    optimizer = tf.keras.optimizers.Adagrad(learning_rate = learn_rate)
elif optimizer == 'Adamax':
    optimizer = tf.keras.optimizers.Adamax(learning_rate = learn_rate)
elif optimizer == 'Nadam':
    optimizer = tf.keras.optimizers.Nadam(learning_rate = learn_rate)
elif optimizer == 'Ftrl':
    optimizer = tf.keras.optimizers.Ftrl(learning_rate = learn_rate)



### ------- Define NN and ODE integrator-----------------
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

  @tf.function
  def call(self, t, y):
    # Neural ODE component
    i0 = self.eqn(y)
    return i0



### -------- Model Training Loop ---------------------

train_loss_results = []
bfgs_loss = []
start_time = time.time()

if adjoint == True:
    int_ode = adjoint_odeint

elif adjoint == False:
    int_ode = odeint

if purpose == 'train':
    if not os.path.exists(savedir+'/current/model_weights/'):
        os.makedirs(savedir+'/current/model_weights/')
    if minibatch == True:

        # Prepare the training dataset.
        dataset = tf.data.Dataset.from_tensor_slices((true_state_tensor, times_tensor))
        dataset = dataset.batch(batch_size)

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

            for epoch in range(epochs):
                with tf.GradientTape() as tape:
                    preds = int_ode(model, tf.expand_dims(init_state, axis=0), times_tensor, method=solver)
                    loss = tf.math.reduce_mean(tf.math.square(true_state_tensor - tf.squeeze(preds)))

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss_results.append(loss.numpy())
                print("Epoch {0}: Loss = {1:0.6f}, LR = {2:0.6f}".format(epoch+1, loss.numpy(), learn_rate(optimizer.iterations).numpy()))
                print()


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
    model.save_weights(savedir+'/current/model_weights/ckpt', save_format='tf')
    if learning_rate_decay:
        train_lr.append(learn_rate(optimizer.iterations).numpy())
    else:
        train_lr.append(learn_rate)
    saved_ep.append(epoch+1)
    np.savez_compressed(savedir+'current/model_weights/train_lr', lr=train_lr, ep=saved_ep)

elif purpose == 'retrain':
    saved_lr = np.load(pre_trained+'train_lr.npz')
    initial_learning_rate = saved_lr['lr'][-1]
    ep = saved_lr['ep'][-1]
    print("Initial lr = {0}".format(initial_learning_rate))
    if not os.path.exists(savedir+'/current/model_weights/'):
        os.makedirs(savedir+'/current/model_weights/')
        
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
        dataset = dataset.batch(batch_size)

        with tf.device(device):
            model = NN()
            model.load_weights(pre_trained+'ckpt')

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
                print("Epoch %d: Loss = %0.6f, LR = %0.6f" %(ep+epoch + 1, avg_loss.result().numpy(), learn_rate(optimizer.iterations).numpy()))
                print()

    elif minibatch == False:

        with tf.device(device):
            model = NN()
            model.load_weights(pre_trained+'ckpt')

            for epoch in range(epochs):

                with tf.GradientTape() as tape:
                    preds = int_ode(model, tf.expand_dims(init_state, axis=0), times_tensor, method=solver)
                    loss = tf.math.reduce_mean(tf.math.square(true_state_tensor - tf.squeeze(preds)))

                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss_results.append(loss.numpy())
                print("Epoch %d: Loss = %0.6f, LR = %0.6f" %(ep+epoch+1, loss.numpy(), learn_rate(optimizer.iterations).numpy()))
                print()


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
    print("****Total training time = {0}****\n".format((end_time - start_time)/3600))
    model.save_weights(savedir+'/current/model_weights/ckpt', save_format='tf')
    if learning_rate_decay:
        train_lr.append(learn_rate(optimizer.iterations).numpy())
    else:
        train_lr.append(learn_rate)
    saved_ep.append(epoch+ep+1)
    np.savez_compressed(savedir+'current/model_weights/train_lr', lr=train_lr, ep=saved_ep)

elif purpose == 'eval':

    model = NN()
    model.load_weights(pre_trained+'ckpt')



### ----- Predict using trained model ---------------
if scale_time == True:
    times_predict = times_predict/time_scaler

predicted_states = int_ode(model, tf.expand_dims(init_state, axis=0),
                                    tf.convert_to_tensor(times_predict), method=solver)
predicted_states = tf.squeeze(predicted_states)
if augmented == True:
    predicted_states = np.delete(predicted_states,slice(20,20+aug_dims),axis=1)



### ---- Post-process predicted states ---------------
if scale_states == True:
    predicted_states = scale_mm.inverse_transform(predicted_states)

if scale_time == True:
    times_predict = times_predict*time_scaler


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

uh = reconstruct_from_rom(Z_pred,U_r,snap_mean,nw)

error_h = np.mean(np.square(uh['S_dep']-snap_pred_true['S_dep']))
error_vx = np.mean(np.square(uh['S_vx']-snap_pred_true['S_vx']))
error_vy = np.mean(np.square(uh['S_vy']-snap_pred_true['S_vy']))

print('H MSE: ' + str(error_h))
print('Vx MSE: ' + str(error_vx))
print('Vy MSE: ' + str(error_vy))


#### ----- Save predicted solutions -------
os.chdir(nodedir+'/current')
print("Saving results in %s"%(os.getcwd()))
if model_sw == 'RED':
    model = 'Red'
elif model_sw == 'SD':
    model = 'SD'
np.savez_compressed('%s_online_node'%(model), S_dep=uh['S_dep'],S_vx = uh['S_vx'], S_vy = uh['S_vy'], time=times_predict)
np.savez_compressed('train_loss', loss=train_loss_results, bfgs_loss=bfgs_loss)
