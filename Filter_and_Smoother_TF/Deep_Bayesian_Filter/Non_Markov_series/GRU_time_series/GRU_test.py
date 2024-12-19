'''
Author: Yan Shi
Date: 2024/01
Testing code of GRU network for non-Markov time series filtering
'''
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import os
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

batch_size = 64
timestep_size = 100
dim_state = 1
dim_meas = 1
m_hidden = 64

with tf.name_scope("globa_lstep"):
    global_steps1 = tf.Variable(0, trainable=False)
with tf.name_scope("place_holder"):
    x = tf.placeholder(tf.float64, shape=(None, timestep_size,1))
    y_ = tf.placeholder(tf.float64, shape=(None, timestep_size,1))

with tf.variable_scope('w_train1', reuse=tf.AUTO_REUSE):
    M_w_1 = tf.get_variable("M_w_1", [m_hidden, 1],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
    M_b_1 = tf.get_variable("M_b_1", [1],dtype=tf.float64)

m_layer = 1
cell_lst = []
for i in range(m_layer):
    cell = rnn.GRUCell(m_hidden,activation=tf.tanh)
    cell_lst.append(cell)
cell = rnn.MultiRNNCell(cell_lst)
initial_state_m = cell.zero_state(batch_size=batch_size, dtype=tf.float64)
state_m = initial_state_m

M_lst = []

for time_step in range(timestep_size):
    with tf.variable_scope('w_train1', reuse=tf.AUTO_REUSE):
        (pred, state_m) = cell(x[:, time_step, :],state_m)
        out = tf.add(tf.matmul(pred, M_w_1), M_b_1)#tf.nn.tanh(
        M_lst.append(out)

M_arr = tf.transpose(tf.stack(M_lst), [1, 0, 2])

loss11 = tf.reduce_mean(tf.square(y_-M_arr[:,:timestep_size,:]))

loss_pos = tf.reduce_mean(tf.square(y_[:,:,:]-M_arr[:,:timestep_size,:]))

saver = tf.train.Saver()
from get_data import add_gaussian_noise,noise_std_r

MC_num = 100
all_gt = np.load('all_gt.npy')
all_noisy = np.load('all_noisy.npy')


gt_test,_ = all_gt[640:],all_noisy[640:]

noisy_test = np.zeros([gt_test.shape[0],gt_test.shape[1]])
LSTM_res = np.zeros([MC_num,gt_test.shape[0],gt_test.shape[1]])

with tf.Session() as sess:
    saver.restore(sess, "./LSTM_1D/train640test64/model.ckpt")
    print('Start training...')
    Loss = []
    Lossv = []

    for step in range(MC_num):
        print('MC:',step)
        for bt in range(gt_test.shape[0]):
            noisy_test[bt] = add_gaussian_noise(gt_test[bt], noise_std_r)
        res,loss_pos_v = sess.run([M_arr,loss_pos],feed_dict={x:noisy_test.reshape([batch_size,timestep_size,dim_meas]),
                                                          y_:gt_test.reshape([batch_size,timestep_size,dim_state])})
        Lossv.append(loss_pos_v)
        LSTM_res[step] = res.reshape([batch_size,timestep_size])

np.save('LSTM_result_q2_r6_train640.npy',LSTM_res)



print('**************************************************')
print('Test loss：',np.sqrt(np.mean(np.array(Lossv))))
print('Measurement noise:',np.sqrt(np.mean(np.square(gt_test-noisy_test))))
print('**************************************************')

