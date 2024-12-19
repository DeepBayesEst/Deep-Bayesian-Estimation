import tensorflow as tf
import numpy as np
from internal_gated import *
import matplotlib.pyplot as plt
import os
import math
from copy import deepcopy

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

batch_size = 8
timestep_size = 100
dim_state = 3
dim_meas = 3
meas_noise = 10
m_hidden = 64
_sT = 0.02


def F_lor(x):
    delta_t = 0.02
    C = np.array([[-10, 10, 0],
                  [28, -1, 0],
                  [0, 0, -8 / 3]], 'float64')
    J = 1
    BX = tf.stack([[[0,0,0],
                    [0,0,-x[0,0]],
                    [0,x[0,0],0]],
                   [[0, 0, 0],
                    [0, 0, -x[1, 0]],
                    [0, x[1, 0], 0]],
                   [[0, 0, 0],
                    [0, 0, -x[2, 0]],
                    [0, x[2, 0], 0]],
                   [[0, 0, 0],
                    [0, 0, -x[3, 0]],
                    [0, x[3, 0], 0]],
                   [[0, 0, 0],
                    [0, 0, -x[4, 0]],
                    [0, x[4, 0], 0]],
                   [[0, 0, 0],
                    [0, 0, -x[5, 0]],
                    [0, x[5, 0], 0]],
                   [[0, 0, 0],
                    [0, 0, -x[6, 0]],
                    [0, x[6, 0], 0]],
                   [[0, 0, 0],
                    [0, 0, -x[7, 0]],
                    [0, x[7, 0], 0]]])
    BX = tf.cast(BX, tf.float64)
    Const = tf.stack(C)
    Const = tf.cast(Const, tf.float64)
    A = tf.add(BX, Const)
    A = tf.cast(A, tf.float64)
    F = tf.stack([tf.eye(dim_meas),tf.eye(dim_meas),tf.eye(dim_meas),tf.eye(dim_meas),tf.eye(dim_meas),tf.eye(dim_meas),tf.eye(dim_meas),tf.eye(dim_meas)])
    F = tf.cast(F, tf.float64)
    for j in range(1,J+1):
        Mat = A * delta_t
        if j == 1:
            F_add = Mat/ math.factorial(j)
            F = tf.add(F, F_add)
        # # else:
        # #     F_add = tf.matmul(Mat,Mat) / math.factorial(j)
        # #     F = tf.add(F, F_add)
        # elif j ==2:
        #     F_add = tf.matmul(Mat,Mat) / math.factorial(j)
        #     F = tf.add(F, F_add)
        # elif j==3:
        #     F_add = tf.matmul(tf.matmul(Mat,Mat),Mat) / math.factorial(j)
        #     F = tf.add(F, F_add)
        # elif j==4:
        #     F_add = tf.matmul(tf.matmul(tf.matmul(Mat,Mat),Mat),Mat) / math.factorial(j)
        #     F = tf.add(F, F_add)
        # else:
        #     F_add = tf.matmul(tf.matmul(tf.matmul(tf.matmul(Mat, Mat), Mat), Mat),Mat) / math.factorial(j)
        #     F = tf.add(F, F_add)
    return F


def H_linear(bt):
    #
    roll_deg = yaw_deg = pitch_deg = 10
    #
    roll = roll_deg * (math.pi / 180)
    yaw = yaw_deg * (math.pi / 180)
    pitch = pitch_deg * (math.pi / 180)
    RX = np.array([[1, 0, 0],
                   [0, math.cos(roll), -math.sin(roll)],
                   [0, math.sin(roll), math.cos(roll)]], 'float64')
    RY = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                   [0, 1, 0],
                   [-math.sin(pitch), 0, math.cos(pitch)]], 'float64')
    RZ = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw), math.cos(yaw), 0],
                   [0, 0, 1]], 'float64')
    RotMatrix = RZ @ RY @ RX
    H_Rotate = RotMatrix @ np.eye(dim_state)
    H = H_Rotate.reshape([1, dim_state, dim_state]).repeat(bt,axis=0)# [batch_size, n, n] rotated matrix
    return H

def Meas_noise(bt):
    R_mat = np.array([[meas_noise, 0,0], [0, meas_noise,0],[0,0, meas_noise]], 'float64')
    R_noise = tf.stack(np.array([R_mat for _ in range(bt)], 'float64'))
    return R_noise

def Pro_noise(bt):
    Q_mat = np.diag([0.,0.,0.])
    Q_noise = tf.stack(np.array([Q_mat for _ in range(bt)], 'float64'))
    return Q_noise

F_bt = F_lor
H_bt = H_linear(batch_size)
R_bt = Meas_noise(batch_size)
Q_bt = Pro_noise(batch_size)


with tf.name_scope("globa_lstep"):
    global_steps1 = tf.Variable(0, trainable=False)
with tf.name_scope("place_holder"):
    x = tf.placeholder(tf.float64, shape=(None, timestep_size,3))
    y_ = tf.placeholder(tf.float64, shape=(None, timestep_size-1,3))
    h_start = tf.placeholder(tf.float64, shape=(None,3))
    c_start = tf.placeholder(tf.float64, shape=(None,m_hidden))
    P_start = tf.placeholder(tf.float64, shape=(None,3,3))

cell = LSTMCell(m_hidden,dim_state=dim_state,dim_meas=dim_meas,trans_model=F_bt,
                meas_model=H_bt,meas_noise=R_bt,pro_noise = Q_bt,
                sT = _sT,batch_size = batch_size,activation=tf.nn.relu)

initial_state_m = LSTMStateTuple_KF(c_start, h_start, tf.reshape(P_start,[batch_size,9]))
state_m = initial_state_m

M_lst = []
for time_step in range(1,timestep_size):
    with tf.variable_scope('w_train1', reuse=tf.AUTO_REUSE):
        (pred, state_m,R_est) = cell(x[:, time_step, :],state_m)
        M_lst.append(pred)

M_arr = tf.transpose(tf.stack(M_lst), [1, 0, 2])
loss11 = tf.reduce_mean(tf.square(y_-M_arr[:,:,:]))
learing_rate1 = tf.train.exponential_decay(0.001,
                                          global_step=global_steps1,
                                          decay_steps=1000,
                                          decay_rate=0.9)
train_step1 = tf.train.AdamOptimizer(learing_rate1).minimize(loss11, global_step=global_steps1)
saver = tf.train.Saver()

dir = 'part_f_noR_new/1dB'
Train_data = np.load('data/part_f_noR/10dB/Train_data.npy')



with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('Start training...')

    a = np.array([i for i in range(Train_data.shape[0])])
    for step in range(20000):
        b = np.random.choice(a, size=batch_size)
        data = Train_data[b]
        state_batch = data[:, 1:, 3:].reshape([batch_size, 99, 3])
        meas_batch = data[:, :, :3].reshape([batch_size, 100, 3])
        h_start_in = meas_batch[:, 0, :] #
        c_start_ = np.zeros([batch_size, m_hidden])
        P_start_ = np.stack(np.array([np.eye(3) for _ in range(batch_size)], "float64"))
        _,loss_print,resu = sess.run([train_step1,loss11,M_arr],feed_dict={x:meas_batch,y_:state_batch[:,:,:],h_start:h_start_in,c_start:c_start_,P_start:P_start_})
        if step % 100 == 0:
            print('**************************************************')
            print('Step %d'%step)
            print('Loss：',loss_print)
            print('Measurement noise:',np.mean(np.square(state_batch[:,:,:]-meas_batch[:,1:,:])))
            print('**************************************************')
            # plt.plot(resu[0,:,0],resu[0,:,1],label='est')
            # plt.plot(state_batch[0,:,0],state_batch[0,:,1],label='true')
            # plt.plot(meas_batch[0,:,0],meas_batch[0,:,1],label='meas')
            # plt.legend()
            # plt.show()
        # if step % 500 == 0:
        #     saver.save(sess, "./model/%s/model.ckpt"%dir)  #
