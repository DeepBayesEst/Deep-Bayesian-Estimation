'''
Author : Yan Shi (NPU Auto)
Date : 2023 / 7 / 15

Training files for the NCLT localization task
'''
'''
Dependency package
'''
import numpy as np
from internal_gated import *
import os
from copy import deepcopy
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Specify a specific GPU
'''
Parameter settings
'''
batch_size = 6
timestep_size = 100
dim_state = 4
dim_meas = 2
meas_noise_ = 2
meas_noise = 1
_sT = 1
m_hidden = 32
'''
Model knowledge
'''
def F_cv(bt):
    '''
    Get the nominal constant velocity linear motion matrix.
    :param bt: batch size
    :return: CV model in batch
    '''
    temp0 = [1., 0., 1., 0.]
    temp1 = [0., 1., 0., 1.]
    temp2 = [0., 0., 1., 0.]
    temp3 = [0., 0., 0., 1.]
    F_c = tf.stack([temp0, temp1, temp2, temp3], axis=0)
    F_c = tf.cast(F_c, tf.float64)
    F = tf.stack([F_c for _ in range(bt)])
    return F

def H_linear(bt):
    '''
    Get the nonlinear radar measurement function.
    :param x: current state
    :return: measurement in polar coordinates ([azimuth, radial distance])
    '''
    temp0 = [0., 0., 1., 0.]
    temp1 = [0., 0., 0., 1.]
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
    R_mat = np.array([[np.square(0.0001), 0], [0, np.square(0.0001)]], 'float64')
    R_noise = tf.stack(np.array([R_mat for _ in range(bt)], 'float64'))
    return R_noise

def Pro_noise(bt):
    '''
    Get the process noise covariance matrix.
    :param bt: batch size
    :return: process noise covariance matrix
    '''
    Q_mat = np.diag([1,1,1,1])
    Q_noise = tf.stack(np.array([Q_mat for _ in range(bt)], 'float64'))
    return Q_noise
# Get the prior model knowledge
F_bt = F_cv(batch_size)
H_bt = H_linear(batch_size)
R_bt = Meas_noise(batch_size)
Q_bt = Pro_noise(batch_size)
'''
Build the TF static graph
'''
# Define the global step variable
with tf.name_scope("globa_lstep"):
    global_steps1 = tf.Variable(0, trainable=False)
    # Define input placeholders
with tf.name_scope("place_holder"):
    x = tf.placeholder(tf.float64, shape=(None, timestep_size,2))  #
    y_ = tf.placeholder(tf.float64, shape=(None, timestep_size-1,2))    #
    h_start = tf.placeholder(tf.float64, shape=(None,4))    #
    c_start = tf.placeholder(tf.float64, shape=(None,m_hidden))    #
    P_start = tf.placeholder(tf.float64, shape=(None,4,4))    #
# Initialize EGBRNN
cell = LSTMCell(m_hidden,dim_state=dim_state,dim_meas=dim_meas,trans_model=F_bt,
                meas_model=H_bt,meas_noise=R_bt,pro_noise = Q_bt,
                sT = _sT,batch_size = batch_size,activation=tf.nn.relu)

# # Initialize the state. The LSTM writing method is retained here, so the state packs the memory, state (h_start) and covariance
initial_state_m = LSTMStateTuple_KF(c_start, h_start, tf.reshape(P_start,[batch_size,16]))
state_m = initial_state_m

# Define a list to save the estimated results
M_lst = []

# Start the loop and perform state estimation
for time_step in range(1, timestep_size):
    with tf.variable_scope('w_train1', reuse=tf.AUTO_REUSE):
        (pred, state_m) = cell(x[:, time_step, :],state_m)
        M_lst.append(pred)

# Get the final estimation result Batch
M_arr = tf.transpose(tf.stack(M_lst), [1, 0, 2])
# Loss function
loss_pos = tf.reduce_mean(tf.square(y_[:,:,:]-M_arr[:,:timestep_size,:2]))
# Learning rate decay
learing_rate1 = tf.train.exponential_decay(0.01,
                                          global_step=global_steps1,
                                          decay_steps=1000,
                                          decay_rate=0.9)

# Define training
train_step1 = tf.train.AdamOptimizer(learing_rate1).minimize(loss_pos, global_step=global_steps1)
saver = tf.train.Saver()

# Data reading
print('Reading data...')
data_all = np.load('data_set.npy')
# data_set = data_all[6:]
data_set = np.concatenate([np.load('data_set.npy')[:30],np.load('data_set.npy')[36:]])

val_data = data_all[30:36]#
print('Read Success',data_set.shape)

loss = []
loss_v = []
Loss = []
Loss_v = []
'''Run the session'''
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Start training...')
    for step in range(20001):
        '''Training set operations'''
        # Randomly extract training data
        a = np.array([i for i in range(data_set.shape[0])])
        b = np.random.choice(a, size=batch_size)
        data = data_set[b]
        # State and measurement sequence packaging
        state_batch = data[:,1:,:2].reshape([batch_size,99,2])
        meas_batch = data[:,:,2:].reshape([batch_size,100,2])
        # Define the initial value of the state
        h_start_in = np.zeros([batch_size, dim_state], dtype='float64')
        h_start_in[:, 0] = state_batch[:, 0, 0] + np.random.normal(0, 1)
        h_start_in[:, 1] = state_batch[:, 0, 1] + np.random.normal(0, 1)
        h_start_in[:, 2] = meas_batch[:, 0, 0]
        h_start_in[:, 3] = meas_batch[:, 0, 1]
        # Define the initial value of memory
        c_start_ = np.zeros([batch_size, m_hidden])
        # Define the initial value of the covariance
        P_start_ = np.stack(np.array([np.eye(4) for _ in range(batch_size)], "float64"))
        '''Validation set operations'''
        # Randomly extract Validation data
        a_v = np.array([i for i in range(6)])
        b_v = np.random.choice(a_v, size=batch_size)
        data_v = val_data[b_v]
        # State and measurement sequence packaging
        state_batch_v = data_v[:, 1:, :2].reshape([batch_size, 99, 2])
        meas_batch_v = data_v[:, :, 2:].reshape([batch_size, 100, 2])
        # Define the initial value of the state
        h_start_in_v = np.zeros([batch_size, dim_state], dtype='float64')
        h_start_in_v[:, 0] = state_batch_v[:, 0, 0] + np.random.normal(0, 1)
        h_start_in_v[:, 1] = state_batch_v[:, 0, 1] + np.random.normal(0, 1)
        h_start_in_v[:, 2] = meas_batch_v[:, 0, 0]
        h_start_in_v[:, 3] = meas_batch_v[:, 0, 1]
        # Define the initial value of memory
        c_start_v = np.zeros([batch_size, m_hidden])
        # Define the initial value of the covariance
        P_start_v = np.stack(np.array([np.eye(4) for _ in range(batch_size)], "float64"))
        # Execution Training
        _,resu,loss_pos_ = sess.run([train_step1,M_arr,loss_pos],feed_dict={x:meas_batch,y_:state_batch[:,:,:],h_start:h_start_in,c_start:c_start_,P_start:P_start_})
        loss.append(np.sqrt(loss_pos_))
        # Output the results of the validation set
        loss_pos_v = sess.run([loss_pos],feed_dict={x: meas_batch_v, y_: state_batch_v, h_start: h_start_in_v,
                                                        c_start: c_start_v, P_start: P_start_v})
        loss_v.append(np.sqrt(loss_pos_v))
        if step % 100 == 0:
            print('**************************************************')
            print('Step %d'%step)
            print('Localization accuracy:',np.sqrt(loss_pos_))
            print('Localization accuracy:(val):',np.sqrt(loss_pos_v))
            Loss.append(np.mean(np.array(loss)))
            Loss_v.append(np.mean(np.array(loss_v)))
            print('**************************************************')
        # Save model
        if step % 1000 == 0 and step > 1:
            saver.save(sess, "./model/model.ckpt")  #
