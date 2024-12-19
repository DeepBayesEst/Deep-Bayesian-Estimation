'''
Author: Yan Shi
Date: 2024/01
Testing code of LSTM for the NCLT localization
'''
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

# from data_prepare import get_batch_data
import matplotlib.pyplot as plt
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

batch_size = 1
timestep_size = 100
dim_state = 4
dim_meas = 2
meas_noise = 6

def get_total_params_num():
    total_num = 0
    for var in tf.trainable_variables():
        var_shape = var.get_shape()
        var_params_num = 1
        for dim in var_shape:
            var_params_num *= dim.value
        total_num += var_params_num
    return total_num

_sT = 1


m_hidden = 256

with tf.name_scope("globa_lstep"):
    global_steps1 = tf.Variable(0, trainable=False)
with tf.name_scope("place_holder"):
    x = tf.placeholder(tf.float64, shape=(None, timestep_size,2))
    y_ = tf.placeholder(tf.float64, shape=(None, timestep_size-1,2))
    h_start = tf.placeholder(tf.float64, shape=(None,4))
    c_start = tf.placeholder(tf.float64, shape=(None,m_hidden))
    P_start = tf.placeholder(tf.float64, shape=(None,4,4))

with tf.variable_scope('w_train1', reuse=tf.AUTO_REUSE):
    M_w_1 = tf.get_variable("M_w_1", [m_hidden, 4],initializer=tf.contrib.layers.xavier_initializer(),dtype=tf.float64)
    M_b_1 = tf.get_variable("M_b_1", [4],dtype=tf.float64)

m_layer = 1
output_keep_prob_m = 0.5
input_keep_prob_m = 0.5
cell_lst = []
for i in range(m_layer):
    cell = rnn.GRUCell(m_hidden,activation=tf.tanh)
    cell_drop = rnn.DropoutWrapper(cell, output_keep_prob=output_keep_prob_m)
    cell = cell_drop
    cell_lst.append(cell)
cell = rnn.MultiRNNCell(cell_lst)
initial_state_m = cell.zero_state(batch_size=batch_size, dtype=tf.float64)
state_m = initial_state_m

M_lst = []

for time_step in range(timestep_size):
    with tf.variable_scope('w_train1', reuse=tf.AUTO_REUSE):
        (pred, state_m) = cell(x[:, time_step, :]/10,state_m)
        out = tf.add(tf.matmul(pred, M_w_1), M_b_1)#tf.nn.tanh(
        M_lst.append(out)

print("Total trainable params number:", get_total_params_num())

M_arr = tf.transpose(tf.stack(M_lst), [1, 0, 2])

loss11 = tf.reduce_mean(tf.square(y_-M_arr[:,:timestep_size-1,:2]))

learing_rate1 = tf.train.exponential_decay(0.01,
                                          global_step=global_steps1,
                                          decay_steps=2000,
                                          decay_rate=0.1)

train_step1 = tf.train.AdamOptimizer(learing_rate1).minimize(loss11, global_step=global_steps1)
saver = tf.train.Saver()

data_set = np.load('data_set.npy')[-6:]
import time
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Testing...')
    for step in range(10):
        data = data_set
        state_batch = data[0,1:,:2].reshape([batch_size,99,2])
        meas_batch = data[0,:,2:].reshape([batch_size,100,2])
        c_start_ = np.zeros([batch_size, m_hidden])
        P_start_ = np.stack(np.array([np.eye(4) for _ in range(batch_size)], "float64"))
        start = time.time()
        loss_print,resu = sess.run([loss11,M_arr],feed_dict={x:meas_batch,y_:state_batch[:,:,:],c_start:c_start_,P_start:P_start_})
        end = time.time()
        print('Inference time',end-start)
        print('Loss：', loss_print)
        np.save('LSTMout.npy',resu)


