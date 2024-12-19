'''
Author: Yan Shi
Date: 2023/07
Training code that corresponds to the landing airplane tracking experiment in the paper.
'''
'''
Dependency package
'''
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from internal_gated import * # Importing the gated Bayesian network
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Specify a specific GPU
'''
Parameter settings
'''
batch_size = 8
timestep_size = 200 # State evolution duration
dim_state = 4 # state dimension [x,y,vx,vy]
dim_meas = 2 # measurement dimension
azi_n = 0.3 * np.pi / 180 # Azimuth error (standard deviation)
dis_n = 150 # radial distance error (standard deviation)
_sT = 4. # sampling Interval
m_hidden = 64 # Number of hidden nodes (and the dimension of memory) of EGBRNN
lambda_l2 = 1e-5  # regularization strength
'''
Model knowledge
'''
def F_cv(bt):
    '''
    Get the nominal constant velocity linear motion matrix.
    :param bt: batch size
    :return: CV model in batch
    '''
    temp0 = [1., 0., sT, 0.]
    temp1 = [0., 1., 0., sT]
    temp2 = [0., 0., 1., 0.]
    temp3 = [0., 0., 0., 1.]
    F_c = tf.stack([temp0, temp1, temp2, temp3], axis=0)
    F_c = tf.cast(F_c, tf.float64)
    F = tf.stack([F_c for _ in range(bt)])
    return F
def my_hx(x):
    '''
    Get the nonlinear radar measurement function.
    :param x: current state
    :return: measurement in polar coordinates ([azimuth, radial distance])
    '''
    x = tf.reshape(x,[batch_size,dim_state])
    r0= tf.atan2(x[:,1],x[:,0]) # azimuth
    r1= tf.sqrt(tf.square(x[:,0])+tf.square(x[:,1])) # radial distance
    r = tf.stack([r0,r1],axis = 1)
    return tf.reshape(r,[batch_size,dim_meas,1])
def hx(x_):
    '''
    Get the Jacobi matrix corresponding to the measurement function.
    :param x_: state
    :return: Jacobi matrix
    '''
    x_ = tf.reshape(x_,[batch_size,dim_state])
    for i in range(batch_size):
        x = x_[i]
        temph0 = [-x[1] / (tf.square(x[0]) + tf.square(x[1])),
                  x[0] / (tf.square(x[0]) + tf.square(x[1])), 0., 0.]
        temph1 = [x[0] / tf.sqrt((tf.square(x[0]) + tf.square(x[1]))),
                  x[1] / tf.sqrt((tf.square(x[0]) + tf.square(x[1]))), 0., 0.]
        H_mat = tf.stack([temph0, temph1], axis=0)
        if i == 0:
            H_out = H_mat
        else:
            H_out = tf.concat([H_out,H_mat],axis = 0)
    H_out = tf.reshape(H_out,[batch_size,2,4])
    return H_out
def H_linear(bt):
    '''
    Get the linear measurement matrix (not use here).
    :param bt: batch size
    :return: measurement matrix
    '''
    temp0 = [1., 0., 0., 0.]
    temp1 = [0., 1., 0., 0.]
    H_ = tf.stack([temp0, temp1], axis=0)
    H_ = tf.cast(H_, tf.float64)
    H_bt = tf.stack([H_ for _ in range(bt)])
    return H_bt
def Meas_noise(bt):
    '''
    Get the measurement noise covariance matrix.
    :param bt: batch size
    :return: measurement noise covariance matrix
    '''
    R_mat = np.array([[np.square(azi_n), 0], [0, np.square(dis_n)]], 'float64')
    R_noise = tf.stack(np.array([R_mat for _ in range(bt)], 'float64'))
    return R_noise
def Pro_noise(bt):
    '''
    Get the process noise covariance matrix.
    :param bt: batch size
    :return: process noise covariance matrix
    '''
    Q_mat = np.diag([25,25,25,25])
    Q_noise = tf.stack(np.array([Q_mat for _ in range(bt)], 'float64'))
    return Q_noise
def l2_regularization():
    '''
    L2 regularization loss function
    :return: L2 losss
    '''
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
    return l2_loss
# Get the prior model knowledge
F_bt = F_cv(batch_size) # Dynamic function
H_bt = H_linear(batch_size) # Linear measurement function (not use here)
R_bt = Meas_noise(batch_size) # Measurement noise covariance
Q_bt = Pro_noise(batch_size) # Process noise covariance
H_J = hx # Jacobi matrix of the measurement function
h = my_hx # Nonlinear radar measurement function
'''
Build the TF static graph
'''
# Define the global step variable
with tf.name_scope("globa_lstep"):
    global_steps1 = tf.Variable(0, trainable=False)
# Define input placeholders
with tf.name_scope("place_holder"):
    x = tf.placeholder(tf.float64, shape=(None, timestep_size,2))
    y_ = tf.placeholder(tf.float64, shape=(None, timestep_size-1,4))
    h_start = tf.placeholder(tf.float64, shape=(None,4))
    c_start = tf.placeholder(tf.float64, shape=(None,m_hidden))
    P_start = tf.placeholder(tf.float64, shape=(None,4,4))
# Initialize EGBRNN
cell = LSTMCell(m_hidden,dim_state=dim_state,dim_meas=dim_meas,trans_model=F_bt,
                meas_model=H_bt,meas_func = h,meas_matrix = H_J,meas_noise=R_bt,pro_noise = Q_bt,
                sT = _sT,batch_size = batch_size,activation=tf.nn.relu)
# Initialize memory, state mean and covariance
state_m = LSTMStateTuple_KF(c_start, h_start, tf.reshape(P_start,[batch_size,16]))
# List for storage of results
M_lst = []
# Start Filtering
for time_step in range(1,timestep_size):
    with tf.variable_scope('w_train1', reuse=tf.AUTO_REUSE):
        (pred, state_m) = cell(x[:, time_step, :],state_m) # recursive filtering
        M_lst.append(pred)
# Filtered state
M_arr = tf.transpose(tf.stack(M_lst), [1, 0, 2])
# Loss computation
l2_loss = l2_regularization()  # L2 regularization
loss1 = tf.reduce_mean(tf.square(y_ - M_arr)) # all state loss
loss_pos = tf.reduce_mean(tf.square(y_[:,:,:2]-M_arr[:,:,:2])) # position loss
Total_loss = loss_pos*2+loss1 # + lambda_l2 * l2_loss
# Learning rate decay
learing_rate1 = tf.compat.v1.train.exponential_decay(0.01,
                                          global_step=global_steps1,
                                          decay_steps=1000,
                                          decay_rate=0.9)
# Define Training
train_step1 = tf.compat.v1.train.AdamOptimizer(learing_rate1).minimize(Total_loss, global_step=global_steps1)
saver = tf.train.Saver()
# Read data
data_num = 234 # Data size
data_set = np.load('data/Train_data.npy')[:234]

'''Run the session'''
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Strat training ...')
    for step in range(100000):
        # Random sampling training data
        a = np.array([i for i in range(data_num)])
        b = np.random.choice(a, size=batch_size)
        data = data_set[b]
        state_batch = data[:,:,:].reshape([batch_size,timestep_size,4]) # bt * 200 * 4
        # Generate Polar Measurements
        meas_batch = deepcopy(data[:, :, :2].reshape([batch_size, timestep_size, 2]))
        measurement = np.zeros_like(meas_batch)
        Obser = np.zeros_like(meas_batch)
        # xi = 0.8 # Glint Level (for glint noise)
        for i in range(batch_size):
            # Generate nonlinear radar measurements
            measurement[i, :, 0] = np.arctan2(meas_batch[i, :, 1], meas_batch[i, :, 0]) + np.random.normal(0, azi_n,timestep_size) # azimuth
            measurement[i, :, 1] = np.sqrt(np.square(meas_batch[i, :, 0]) + np.square(meas_batch[i, :, 1])) + np.random.normal(0, dis_n,timestep_size) # radial distance
            # Convert Polar Measurements to Cartesian Coordinates (for check)
            Obser[i, :, 0] = measurement[i, :, 1] * np.cos(measurement[i, :, 0])
            Obser[i, :, 1] = measurement[i, :, 1] * np.sin(measurement[i, :, 0])
            # The commented out content is the generation of glint noise
            # measurement[i, :, 0] = np.arctan2(meas_batch[i, :, 1], meas_batch[i, :, 0]) + (1-xi)*np.random.normal(0, azi_n,timestep_size)+xi*np.random.laplace(0,azi_n,timestep_size)
            # measurement[i, :, 1] = np.sqrt(np.square(meas_batch[i, :, 0]) + np.square(meas_batch[i, :, 1])) + (1-xi)*np.random.normal(0, dis_n,timestep_size) +xi*np.random.laplace(0,dis_n,timestep_size)
        # Initialize the state
        h_start_in = np.zeros([batch_size, dim_state], dtype='float64')
        h_start_in[:, 0] = state_batch[:, 0, 0]
        h_start_in[:, 1] = state_batch[:, 0, 1]
        h_start_in[:, 2] = state_batch[:, 0, 2] #+ np.random.normal(0, 15)
        h_start_in[:, 3] = state_batch[:, 0, 3] #+ np.random.normal(0, 15)
        # Initialize the memory (The mean and variance of the memory are considered to be combined into one vector.)
        c_start_ = np.zeros([batch_size, m_hidden])
        # Initialize the covariance
        P_start_ = np.stack(np.array([10*np.eye(4) for _ in range(batch_size)], "float64"))
        # Training
        _,loss_print,resu,loss_pos_ = sess.run([train_step1,loss1,M_arr,loss_pos],feed_dict={x:measurement,y_:state_batch[:,1:,:],h_start:h_start_in,c_start:c_start_,P_start:P_start_})
        if step % 100 == 0:
            print('**************************************************')
            print('Step %d has been run'%step)
            print('The overall loss:',np.sqrt(loss_print))
            print('Measurement Noise:',np.sqrt(np.mean(np.square(state_batch[:,:,:2]-Obser))))
            print('Position error',np.sqrt(loss_pos_))
            print('**************************************************')
        # Save model
        if step % 1000 == 0:
            saver.save(sess, "./model/model.ckpt")  #
