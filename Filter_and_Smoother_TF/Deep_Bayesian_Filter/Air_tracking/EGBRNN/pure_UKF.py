'''
Author: Yan Shi
Date: 2023/07
UKF code for air target tracking.
'''

from filterpy.kalman import JulierSigmaPoints as SP
from filterpy.kalman import UnscentedKalmanFilter as UKF
# from filterpy.kalman import
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

timestep_size = 200
test_size = 26
dim_state = 4
dim_meas = 2
batch_size = 1
sT = 4.
Test_data = np.load('../Train_data.npy')[:26]

def f_cv(x,dt):
    F_cv = np.array([[1, 0, dt, 0],
                 [0, 1, 0, dt],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]], 'float64')
    x_out = F_cv.reshape([dim_state, dim_state]) @ x.reshape([dim_state, 1])
    return x_out.reshape([dim_state])

def my_hx(x):
    """ returns slant range = np.array([[0],[0]]) based on downrange distance and altitude"""
    r0 = np.arctan2(x[1], x[0])
    r1 = np.sqrt(np.square(x[0]) + np.square(x[1]))
    r = np.stack([r0, r1])
    return r

azi_n = 0.3 * np.pi / 180
dis_n = 150
R = np.array([[np.square(azi_n), 0], [0, np.square(dis_n)]], 'float64')

Q = np.diag([8,8,8,8])
MC_num = 1
State_all = np.zeros([MC_num,test_size,timestep_size,dim_state])
Est_all = np.zeros([MC_num,test_size,timestep_size,dim_state])
Obser_all = np.zeros([MC_num,test_size,timestep_size,dim_meas])

result_pos = 0
result_vel = 0

for mc in range(MC_num):
    print('MC:',mc)
    for epoch in range(test_size):
        print('epoch',epoch)
        data = Test_data[epoch].reshape([timestep_size,dim_state])
        # print(data.shape)
        state = data.reshape([timestep_size, dim_state])

        # meas = data[:, :dim_state].reshape([timestep_size, dim_state])
        xi = .2
        meas = np.zeros([timestep_size, 2])  #
        Obser = np.zeros([timestep_size, 2])
        meas_gau = np.zeros([timestep_size, 2])  #
        Obser_gau = np.zeros([timestep_size, 2])
        for t in range(timestep_size):
            meas[t, :] = my_hx(state[t, :])
            meas_gau[t, :] = my_hx(state[t, :])

            meas_gau[t, 0] += np.random.normal(0, azi_n)
            meas_gau[t, 1] += np.random.normal(0, dis_n)
            Obser_gau[t, 0] = meas_gau[t, 1] * np.cos(meas_gau[t, 0])
            Obser_gau[t, 1] = meas_gau[t, 1] * np.sin(meas_gau[t, 0])

            meas[t, 0] += (1-xi)*np.random.normal(0, azi_n)+xi*np.random.laplace(0,azi_n*5)  # 方位角#np.random.normal(0, azi_n)
            meas[t, 1] += (1-xi)*np.random.normal(0, dis_n)+xi*np.random.laplace(0,dis_n*5)#np.random.normal(0, dis_n)
            Obser[t, 0] = meas[t, 1] * np.cos(meas[t, 0])
            Obser[t, 1] = meas[t, 1] * np.sin(meas[t, 0])
        # plt.plot(Obser[:,0],Obser[:,1])
        # plt.plot(Obser_gau[:,0],Obser_gau[:,1])
        #
        # plt.plot(state[:,0],state[:,1])
        # plt.show()
        Obser_all[mc,epoch] = Obser
        State_all[mc,epoch] = state[:, :]
        print('glint:',np.sqrt(np.mean(np.square(state[:, :2] - Obser))))
        print('Gau:', np.sqrt(np.mean(np.square(state[:, :2] - Obser_gau))))
        # UKF初始化
        my_SP = SP(dim_state, kappa=0.)
        kf = UKF(dim_x=dim_state, dim_z=dim_meas, dt=sT,fx=f_cv,hx=my_hx,points=my_SP)
        start_point = data[0].reshape([dim_state])
        P_start = 500*np.eye(dim_state)
        kf.x = start_point
        kf.P = P_start
        # kf.F = F
        # kf.H = H
        kf.Q = Q
        kf.R = R
        est_epoch = np.zeros([timestep_size,dim_state])
        est_epoch[0] = start_point.reshape([dim_state])
        for k in range(1,timestep_size):
            kf.predict()
            kf.update(meas[k])
            est_epoch[k] = kf.x.reshape([dim_state])
        # print(np.mean(np.square(state-est_epoch[1:])))
        # plt.figure()
        # plt.plot(est_epoch[:,0],est_epoch[:,1])
        # plt.plot(state[:,0],state[:,1])
        # plt.show()
        Est_all[mc,epoch] = est_epoch[:]

    loss_pos = np.sqrt(np.mean(np.square(State_all[:,:,:,:2]-Est_all[:,:,:,:2])))
    loss_vel = np.sqrt(np.mean(np.square(np.sqrt(np.square(State_all[:,:,:,2]+State_all[:,:,:,3]))-np.sqrt(np.square(Est_all[:,:,:,2]+Est_all[:,:,:,3])))))
    result_pos+=loss_pos
    result_vel+=loss_vel

print('pos_LOSS:',result_pos/MC_num)
print('vel_LOSS:',result_vel/MC_num)

