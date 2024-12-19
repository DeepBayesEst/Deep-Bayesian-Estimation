'''
Author: Yan Shi
Date: 2023/07
Training code that corresponds to the non-Markov time series filtering experiment in the paper.
'''
'''
Dependency package
'''
import numpy as np
from internal_gated import *
import os
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
lambda_l2 = 1e-5  # regularization strength
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
# Loss computation
loss = tf.reduce_mean(tf.square(y_-M_arr))
# Leaning rate decay
learning_rate = tf.train.exponential_decay(0.01,
                                          global_step=global_steps1,
                                          decay_steps=1000,
                                          decay_rate=0.9)
# Define Training
train_step1 = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_steps1)
saver = tf.train.Saver()

# Read data
all_gt = np.load('data/all_gt.npy')
all_noisy = np.load('data/all_noisy.npy')
gt_train,noisy_train = all_gt[:640],all_noisy[:640]
'''
Run the session
'''
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Strat training ...')
    Loss = []
    for step in range(200000):
        # Random sampling training data
        a = np.array([i for i in range(gt_train.shape[0])])
        b = np.random.choice(a, size=batch_size)
        gt = gt_train[b]
        noisy = noisy_train[b]
        # Initialize the state
        h_start_in = np.zeros([batch_size], dtype='float64').reshape([batch_size,1])
        h_start_in[:] = gt[:, 0].reshape([batch_size,1])
        # Initialize the memory (The mean and variance of the memory are considered to be combined into one vector.)
        c_start_ = np.zeros([batch_size, m_hidden])
        # Initialize the variance
        P_start_ = np.array([1 for _ in range(batch_size)], "float64").reshape([batch_size,1])
        # Training
        _,loss_print,resu = sess.run([train_step1,loss,M_arr],
                                                            feed_dict={x:noisy.reshape([batch_size,timestep_size,1]),
                                                                       y_:gt[:,1:].reshape([batch_size,timestep_size-1,1]),
                                                                       h_start:h_start_in,c_start:c_start_,P_start:P_start_})
        Loss.append(loss_print)
        if step % 100 == 0:
            print('**************************************************')
            print('Step %d has been run'%step)
            print('Loss ：',np.sqrt(np.mean(np.array(Loss))))
            print('Measurement Noise:',np.sqrt(np.mean(np.square(gt-noisy))))
            Loss = []
            print('**************************************************')
        # Save model
        if step % 200 == 0:
            saver.save(sess, "./model/model.ckpt")  #
