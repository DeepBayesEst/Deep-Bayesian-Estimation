'''
Author : Yan Shi (NPU Auto)
Date : 2023 / 7 / 15

Testing files for the NCLT localization task
'''

'''
Dependency package
'''
import numpy as np
from internal_gated import * #
import matplotlib.pyplot as plt
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Specify a specific GPU
'''
Parameter settings
'''
batch_size = 6 #
timestep_size = 100 #
dim_state = 4 # state dimension
dim_meas = 2 # measurement dimension
meas_noise = 1 #
_sT = 1 #
m_hidden = 32 #

def get_total_params_num():
    '''
    This function calculates the number of learnable parameters in the network
    :return: the number of learnable parameters
    '''
    total_num = 0
    for var in tf.trainable_variables():
        var_shape = var.get_shape()
        var_params_num = 1
        for dim in var_shape:
            var_params_num *= dim.value
        total_num += var_params_num
    return total_num

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
    F = tf.stack([F_c for _ in range(bt)]) #
    return F


def H_linear(bt):
    '''
    Get the linear measurement matrix (not use here).
    :param bt: batch size
    :return: measurement matrix
    '''
    temp0 = [0., 0., 1., 0.]
    temp1 = [0., 0., 0., 1.]
    H_ = tf.stack([temp0, temp1], axis=0)
    H_ = tf.cast(H_, tf.float64)
    H_bt = tf.stack([H_ for _ in range(bt)])#
    return H_bt

def Meas_noise(bt):
    '''
    Get the measurement noise covariance matrix.
    :param bt: batch size
    :return: measurement noise covariance matrix
    '''
    R_mat = np.array([[np.square(0.0001), 0], [0, np.square(0.0001)]], 'float64')
    R_noise = tf.stack(np.array([R_mat for _ in range(bt)], 'float64')) # 将R矩阵打包成一个Batch
    return R_noise

def Pro_noise(bt):
    '''
    Get the process noise covariance matrix.
    :param bt: batch size
    :return: process noise covariance matrix
    '''
    Q_mat = np.diag([1,1,1,1])
    Q_noise = tf.stack(np.array([Q_mat for _ in range(bt)], 'float64')) # 将Q矩阵打包成一个Batch
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
    x = tf.placeholder(tf.float64, shape=(None, timestep_size,2))
    y_ = tf.placeholder(tf.float64, shape=(None, timestep_size-1,2))
    h_start = tf.placeholder(tf.float64, shape=(None,4))
    c_start = tf.placeholder(tf.float64, shape=(None,m_hidden))
    P_start = tf.placeholder(tf.float64, shape=(None,4,4))

# Initialize EGBRNN
cell = LSTMCell(m_hidden,dim_state=dim_state,dim_meas=dim_meas,trans_model=F_bt,
                meas_model=H_bt,meas_noise=R_bt,pro_noise = Q_bt,
                sT = _sT,batch_size = batch_size,activation=tf.nn.relu)
# Initialize memory, state mean and covariance
initial_state_m = LSTMStateTuple_KF(c_start, h_start, tf.reshape(P_start,[batch_size,16]))
state_m = initial_state_m
# List for storage of results
M_lst = []
P_lst = []
# Start Filtering
for time_step in range(1,timestep_size):
    with tf.variable_scope('w_train1', reuse=tf.AUTO_REUSE):
        (pred, state_m) = cell(x[:, time_step, :],state_m)
        M_lst.append(pred)
# Filtered state
M_arr = tf.transpose(tf.stack(M_lst), [1, 0, 2])
# Loss computation
loss_pos = tf.reduce_mean(tf.square(y_[:,:,:]-M_arr[:,:,:2]))
saver = tf.train.Saver()
# Read data
print('Reading data...')
data_set = np.load('data_set.npy')[-6:]
print('Read success',data_set.shape)
import time
with tf.Session() as sess:
    saver.restore(sess,"./model/model.ckpt")
    print('Strat testing ...')
    for step in range(10):
        print("Total trainable params number:", get_total_params_num())
        data = data_set
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
        # Execution Testing
        start = time.time()
        resu,loss_pos_ = sess.run([M_arr,loss_pos],feed_dict={x:meas_batch,y_:state_batch[:,:,:],h_start:h_start_in,c_start:c_start_,P_start:P_start_})
        end = time.time()
        print(start-end)
        print('**************************************************')
        print('Step %d' % step)
        print('Localization accuracy:', loss_pos_)
        print('**************************************************')



