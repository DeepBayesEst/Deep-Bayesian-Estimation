'''
Author: Yan Shi
Date: 2024/09
Testing code that corresponds to the terminal area tracking experiment in the paper.
'''
'''
Dependency package
'''
import tensorflow as tf
import numpy as np
from lstm_tf_filtering import LSTMStateTuple_KF
from lstm_tf_filtering import LSTMCell as LSTMCell_filter
from lstm_tf_smoothing_attention import LSTMCell as LSTMCell_smooth
import os
from copy import deepcopy

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'# Specify a specific GPU

'''
Parameter settings
'''
batch_size = 8
timestep_size = 120  # State evolution duration
dim_state = 4  # state dimension
dim_meas = 2  # measurement dimension
azi_n = 0.3*np.pi/180  # The noise standard deviation for the azimuth angle.
dis_n = 150  # The noise standard deviation for the distance.
_sT = 4.  # sample time

'''
Model knowledge
'''
def F_cv(bt):
    '''
    Get the state transition matrix.
    :param bt: batch size
    :return: state transition matrix
    '''
    temp0 = [1., 0., _sT, 0.]
    temp1 = [0., 1., 0., _sT]
    temp2 = [0., 0., 1., 0.]
    temp3 = [0., 0., 0., 1.]
    F_c = tf.stack([temp0, temp1, temp2, temp3], axis=0)
    F_c = tf.cast(F_c, tf.float64)
    F = tf.stack([F_c for _ in range(bt)])
    return F

def my_hx(x):
    '''
    Get radar measurement matrix.
    :param x:target state
    :return: radar measurement matrix
    '''
    x = tf.reshape(x,[batch_size,dim_state])
    r0= tf.atan2(x[:,1],x[:,0])
    r1= tf.sqrt(tf.square(x[:,0])+tf.square(x[:,1]))
    r = tf.stack([r0,r1],axis = 1)
    return tf.reshape(r,[batch_size,dim_meas,1])

def hx(x_):
    '''
    Get measure Jacobian matrix.
    :param x_:target state
    :return: measure Jacobian matrix
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
    Get linear measurement matrix.
    :param x_:target state
    :return: linear measurement matrix
    '''
    temp0 = [1., 0., 0., 0.]
    temp1 = [0., 1., 0., 0.]
    H_ = tf.stack([temp0, temp1], axis=0)
    H_ = tf.cast(H_, tf.float64)
    H_bt = tf.stack([H_ for _ in range(bt)])
    return H_bt

def Meas_noise(bt):
    '''
    Get the measurement noise variance.
    :param bt: batch size
    :return: measurement noise variance
    '''
    R_mat = np.array([[np.square(azi_n), 0], [0, np.square(dis_n)]], 'float64')
    R_noise = tf.stack(np.array([R_mat for _ in range(bt)], 'float64'))
    return R_noise

def Pro_noise(bt):
    '''
    Get the process noise variance.
    :param bt: batch size
    :return: process noise variance
    '''
    Q_mat = np.diag([325,325,150,150])
    Q_noise = tf.stack(np.array([Q_mat for _ in range(bt)], 'float64'))
    return Q_noise

# Get the prior model knowledge
F_bt = F_cv(batch_size)
H_bt = H_linear(batch_size)
R_bt = Meas_noise(batch_size)
Q_bt = Pro_noise(batch_size)
H_J = hx
h = my_hx

# Set hidden layer dimension of network
m_hidden = 64

'''
Build the TF static graph
'''
# Define the global step variable
with tf.name_scope("globa_lstep"):
    global_steps1 = tf.Variable(0, trainable=False)

with tf.name_scope("globa_lstep"):
    global_steps2 = tf.Variable(0, trainable=False)

# Define input placeholders
with tf.name_scope("place_holder"):
    x = tf.placeholder(tf.float64, shape=(None, timestep_size,2))
    y_ = tf.placeholder(tf.float64, shape=(None, timestep_size-1,4))
    h_start = tf.placeholder(tf.float64, shape=(None,4))
    c_start = tf.placeholder(tf.float64, shape=(None,m_hidden))
    P_start = tf.placeholder(tf.float64, shape=(None,4,4))

# Initialize filtering section of EGBRNS
cell_filter = LSTMCell_filter(m_hidden,dim_state=dim_state,dim_meas=dim_meas,trans_model=F_bt,
                meas_model=H_bt,meas_func = h,meas_matrix = H_J,meas_noise=R_bt,pro_noise = Q_bt,
                sT = _sT,batch_size = batch_size,activation=tf.nn.relu)

# Initialize smoothing section of EGBRNS
cell_smooth = LSTMCell_smooth(m_hidden,dim_state=dim_state,dim_meas=dim_meas,trans_model=F_bt,
                meas_model=H_bt,meas_func = h,meas_matrix = H_J,meas_noise=R_bt,pro_noise = Q_bt,
                sT = _sT,batch_size = batch_size,activation=tf.nn.relu)

# Initialize memory, state mean and covariance for filtering
initial_state_m = LSTMStateTuple_KF(c_start, h_start, tf.reshape(P_start,[batch_size,16]))
state_m = initial_state_m

# List for storage of results
M_lst = []  # filter update results
smooth_M_lst = []  # smoothing results
m_p_lst = []  # filter prediction results
P_pred = []  # filter prediction covariance matrix
P_filtered = []  # filter update covariance matrix

# Start Filtering
for time_step in range(1,timestep_size):
    with tf.variable_scope('w_train1', reuse=tf.AUTO_REUSE):
        (pred, state_m, m_p, P_f ,P_p) = cell_filter(x[:, time_step, :],state_m)
        M_lst.append(pred)
        m_p_lst.append(m_p)
        P_pred.append(P_p)
        P_filtered.append(P_f)

C_filter,_,_ = state_m  # Filtered memory
M_arr = tf.transpose(tf.stack(M_lst), [1, 0, 2])  # Filtered state

# Loss computation
loss11 = tf.reduce_mean(tf.square(y_-M_arr[:,:,:]))

# Initialize state for smoothing
smooth_M_lst = M_lst
# Start smoothing
for backtime in range(timestep_size-3,-1,-1):
    with tf.variable_scope('w_train2', reuse=tf.AUTO_REUSE):

        input_smooth = tf.concat([
            smooth_M_lst[backtime + 1],
            M_lst[backtime],
            tf.reshape(m_p_lst[backtime + 1], [batch_size, dim_state]),
            P_filtered[backtime],
            tf.reshape(P_pred[backtime + 1], [batch_size, dim_state * dim_state]),
            C_filter
        ], axis=1)
        (smooth_m, state_m) = cell_smooth(input_smooth,state_m)
        smooth_M_lst[backtime] = smooth_m

# Smoothed state
smooth_M_arr = tf.transpose(tf.stack(smooth_M_lst), [1, 0, 2])

# Loss computation
smooth_loss11 = tf.reduce_mean(tf.square(y_-smooth_M_arr))

# Define Training
saver = tf.train.Saver()

# Read data
data_num = 640  # Set data size
data_set = np.load('true_smooth_air_120.npy')[5000:5000+data_num]
'''
Run the session
'''
with tf.Session() as sess:
    # Read the model
    saver.restore(sess, './model/model.ckpt')
    print('Strat testing ...')
    MC_num = 100  # Monte Carlo times
    loss_filtering_list = []
    loss_smoothing_list = []
    tracks = []
    for num in range(MC_num):
        for step in range(data_num // batch_size):

            data = data_set[step*batch_size:(step + 1)*batch_size, :, :]
            state_batch = data[:,:timestep_size,:].reshape([batch_size,timestep_size,4])

            # Generate polar coordinate measurement data
            meas_batch = deepcopy(data[:, :timestep_size, :2].reshape([batch_size, timestep_size, 2]))
            measurement = np.zeros_like(meas_batch)
            Obser = np.zeros_like(meas_batch)

            for i in range(batch_size):
                measurement[i, :, 0] = np.arctan2(meas_batch[i, :, 1], meas_batch[i, :, 0]) + np.random.normal(0, azi_n,timestep_size)
                measurement[i, :, 1] = np.sqrt(
                    np.square(meas_batch[i, :, 0]) + np.square(meas_batch[i, :, 1])) + np.random.normal(0, dis_n,timestep_size)  #

                Obser[i, :, 0] = measurement[i, :, 1] * np.cos(measurement[i, :, 0])
                Obser[i, :, 1] = measurement[i, :, 1] * np.sin(measurement[i, :, 0])

            # Initialize the state
            h_start_in = np.zeros([batch_size, dim_state], dtype='float64')
            h_start_in[:, 0] = state_batch[:, 0, 0]
            h_start_in[:, 1] = state_batch[:, 0, 1]
            h_start_in[:, 2] = state_batch[:, 0, 2]
            h_start_in[:, 3] = state_batch[:, 0, 3]

            # Initialize the memory (The mean and variance of the memory are considered to be combined into one vector.)
            c_start_ = np.zeros([batch_size, m_hidden])
            # Initialize the variance
            P_start_ = np.stack(np.array([1000*np.eye(4) for _ in range(batch_size)], "float64"))

            # Testing
            loss_filtering,smooth_M_arr_v,loss_smoothing = sess.run([loss11, smooth_M_arr, smooth_loss11], feed_dict={x:measurement, y_: state_batch[:, 1:, :], h_start:h_start_in, c_start:c_start_, P_start:P_start_})
            loss_filtering_list.append(np.sqrt(loss_filtering))
            loss_smoothing_list.append(np.sqrt(loss_smoothing))

            # Record test results
            if step == 0 :
                results_step = deepcopy(smooth_M_arr_v)
            else:
                results_step = np.concatenate((results_step, smooth_M_arr_v), axis=0)

            if step % 10 == 0:
                print('**************************************************')
                print('Step %d has been run' % step)
                loss_filtering_array = np.array(loss_filtering_list)
                loss_smoothing_array = np.array(loss_smoothing_list)
                print('Loss after filtering:', np.mean(loss_filtering_array))
                print('Loss after smoothing：', np.mean(loss_smoothing_array))
                print('Measurement Noise:', np.sqrt(np.mean(np.square(state_batch[:, :, :2] - Obser))))
                loss_filtering_list = []
                loss_smoothing_list = []
                print('**************************************************')

        tracks.append(results_step)
    all_tracks_arr = np.array(tracks)
    # np.save('results.npy', all_tracks_arr)
