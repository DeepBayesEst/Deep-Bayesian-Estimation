'''
Author: Yan Shi
Date: 2023/07
Testing code that corresponds to the non-Markov time series filtering experiment in the paper.
'''
'''
Dependency package
'''
import numpy as np
from internal_gated import *
import matplotlib.pyplot as plt
import os
from data.get_data import add_gaussian_noise,noise_std_r
# os.environ['CUDA_VISIBLE_DEVICES'] = '1' # Specify a specific GPU
'''
Parameter settings
'''
batch_size = 64
timestep_size = 100 # State evolution duration
dim_state = 1 # state dimension
dim_meas = 1 # measurement dimension
meas_noise = 6. # measurement noise
process_noise = 2. # process noise
m_hidden = 64 # Number of hidden nodes (and the dimension of memory) of EGBRNN
'''
Model knowledge
'''
def Meas_noise(bt):
    '''
    Get the measurement noise variance.
    :param bt: batch size
    :return: measurement noise variance
    '''
    R_noise = tf.stack(np.array([meas_noise ** 2 for _ in range(bt)], 'float64').reshape([batch_size,1]))
    return R_noise
def Pro_noise(bt):
    '''
    Get the process noise variance.
    :param bt: batch size
    :return: process noise variance
    '''
    Q_noise = tf.stack(np.array([process_noise**2 for _ in range(bt)], 'float64').reshape([batch_size,1]))
    return Q_noise
# Get the prior model knowledge
R_bt = Meas_noise(batch_size) # Measurement noise variance
Q_bt = Pro_noise(batch_size) # Process noise variance

'''
Build the TF static graph
'''
# Define the global step variable
with tf.name_scope("globa_lstep"):
    global_steps1 = tf.Variable(0, trainable=False)
# Define input placeholders
with tf.name_scope("place_holder"):
    x = tf.placeholder(tf.float64, shape=(None, timestep_size,1))
    y_ = tf.placeholder(tf.float64, shape=(None, timestep_size-1,1))
    h_start = tf.placeholder(tf.float64, shape=(None,1))
    c_start = tf.placeholder(tf.float64, shape=(None,m_hidden))
    P_start = tf.placeholder(tf.float64, shape=(None,1))
# Initialize EGBRNN
cell = LSTMCell(m_hidden,dim_state=dim_state,dim_meas=dim_meas,trans_model=0.5,
                meas_model=1,meas_func = 1,meas_matrix = 1,meas_noise=R_bt,pro_noise = Q_bt,
                sT = 1.,batch_size = batch_size,activation=tf.nn.relu)
# Initialize memory, state mean and covariance
state_m = LSTMStateTuple_KF(c_start, tf.reshape(h_start,[batch_size,1]), tf.reshape(P_start,[batch_size,1]))
# List for storage of results
M_lst = []
# Start Filtering
for time_step in range(1,timestep_size):
    with tf.variable_scope('w_train1', reuse=tf.AUTO_REUSE):
        input = tf.reshape(x[:, time_step],[batch_size,1])
        (pred, state_m) = cell(input,state_m)
        M_lst.append(pred)
# Filtered state
M_arr = tf.transpose(tf.stack(M_lst), [1, 0,2])
loss = tf.reduce_mean(tf.square(y_-M_arr))
# Define Training
saver = tf.train.Saver()
MC_num = 100 # Monte Carlo times
# Read data
all_gt = np.load('data/all_gt.npy')
all_noisy = np.load('data/all_noisy.npy')
gt_test,_ = all_gt[640:],all_noisy[640:]
noisy_test = np.zeros([gt_test.shape[0],gt_test.shape[1]])
MGB_res = np.zeros([MC_num,gt_test.shape[0],gt_test.shape[1]-1])
'''
Run the session
'''
with tf.Session() as sess:
    # Read the model
    saver.restore(sess, "./model/model.ckpt")
    print('Strat testing ...')
    Loss_v = []
    for step in range(MC_num):
        print('MC:',step)
        # Using test data with random noise (to conduct multiple Monte Carlo experiments)
        for bt in range(gt_test.shape[0]):
            noisy_test[bt] = add_gaussian_noise(gt_test[bt], noise_std_r)
        # Initialize the state
        h_start_in_v = np.zeros([batch_size], dtype='float64').reshape([batch_size,1])
        h_start_in_v[:] = gt_test[:, 0].reshape([batch_size,1])
        # Initialize the memory (The mean and variance of the memory are considered to be combined into one vector.)
        c_start_v = np.zeros([batch_size, m_hidden])
        # Initialize the variance
        P_start_v = np.array([1 for _ in range(batch_size)], "float64").reshape([batch_size,1])
        # Training
        res,loss_v = sess.run([M_arr,loss],feed_dict={x: noisy_test.reshape([batch_size, timestep_size, 1]),
                                                                     y_: gt_test[:, 1:].reshape(
                                                                         [batch_size, timestep_size - 1, 1]),
                                                                     h_start: h_start_in_v, c_start: c_start_v,
                                                                     P_start: P_start_v})
        Loss_v.append(loss_v)
        MGB_res[step] = res.reshape([batch_size,timestep_size-1])

# np.save('MGB_result_q2_r6_train64.npy',MGB_res)

print('**************************************************')
print('Test loss : ',np.sqrt(np.mean(np.array(Loss_v))))
print('Measurement noise',np.sqrt(np.mean(np.square(gt_test-noisy_test))))
print('**************************************************')

