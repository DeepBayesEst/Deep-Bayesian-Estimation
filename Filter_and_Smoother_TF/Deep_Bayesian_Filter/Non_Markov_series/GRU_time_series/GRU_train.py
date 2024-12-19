'''
Author: Yan Shi
Date: 2024/01
Training code of GRU network for non-Markov time series filtering
'''

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import os
import matplotlib.pyplot as plt

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

learing_rate1 = tf.train.exponential_decay(0.01,
                                          global_step=global_steps1,
                                          decay_steps=5000,
                                          decay_rate=0.5)

train_step1 = tf.train.AdamOptimizer(learing_rate1).minimize(loss11, global_step=global_steps1)
saver = tf.train.Saver()
all_gt = np.load('all_gt_48.npy')
all_noisy = np.load('all_noisy_48.npy')


gt_train,noisy_train = all_gt[:640],all_noisy[:640]
gt_test,noisy_test = all_gt[640:],all_noisy[640:]

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Start training...')
    Loss = []
    Lossv = []
    for step in range(10001):
        a = np.array([i for i in range(gt_train.shape[0])])
        b = np.random.choice(a, size=batch_size)
        gt = gt_train[b]
        noisy = noisy_train[b]
        # 执行训练
        _,loss_print,resu,loss_pos_ = sess.run([train_step1,loss11,M_arr,loss_pos],
                                               feed_dict={x:noisy.reshape([batch_size,timestep_size,1]),
                                                          y_:gt.reshape([batch_size,timestep_size,1])})
        loss_pos_v = sess.run([loss_pos],feed_dict={x:noisy_test.reshape([batch_size,timestep_size,dim_meas]),
                                                          y_:gt_test.reshape([batch_size,timestep_size,dim_state])})
        Loss.append(loss_print)
        Lossv.append(loss_pos_v)
        if step % 100 == 0:
            print('**************************************************')
            print('Step %d'%step)
            print('Loss：',np.sqrt(np.mean(np.array(Loss))))
            print('Val loss：',np.sqrt(np.mean(np.array(Lossv))))
            print('Measurement noise:',np.sqrt(np.mean(np.square(gt-noisy))))
            Loss = []
            loss_v = []
            print('**************************************************')

            # if step % 500 == 0:
            #     saver.save(sess, "./LSTM_1D/train640_48/model.ckpt")  #