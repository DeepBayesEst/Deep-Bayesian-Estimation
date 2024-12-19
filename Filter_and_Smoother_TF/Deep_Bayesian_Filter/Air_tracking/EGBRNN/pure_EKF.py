import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import ExtendedKalmanFilter as EKF
import math
from mpl_toolkits.mplot3d import Axes3D
timestep_size = 200
test_size = 26
dim_state = 4
dim_meas = 2
batch_size = 1
sT = 4.
Test_data = np.load('../Train_data.npy')[:26]
F_cv_mat = np.array([[1, 0, sT, 0], [0, 1, 0, sT], [0, 0, 1, 0], [0, 0, 0, 1]], 'float64')
def f_cv(x,dt):
    F_cv = np.array([[1, 0, dt, 0],
                 [0, 1, 0, dt],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]], 'float64')
    x_out = F_cv.reshape([dim_state, dim_state]) @ x.reshape([dim_state, 1])
    return x_out.reshape([dim_state])
def hx_J(x):
    x = x.reshape([dim_state])
    H_mat  = np.array([[-x[1] / (np.square(x[0]) + np.square(x[1])),
              x[0] / (np.square(x[0]) + np.square(x[1])), 0., 0.],
                      [x[0] / np.sqrt((np.square(x[0]) + np.square(x[1]))),
              x[1] / np.sqrt((np.square(x[0]) + np.square(x[1]))), 0., 0.]])
    return H_mat.reshape([dim_meas,dim_state])
def my_hx(x):
    """ returns slant range = np.array([[0],[0]]) based on downrange distance and altitude"""
    r0 = np.arctan2(x[1], x[0])
    r1 = np.sqrt(np.square(x[0]) + np.square(x[1]))
    r = np.stack([r0, r1])
    return r
azi_n = 0.1 * np.pi / 180
dis_n = 50
R = np.array([[np.square(azi_n), 0], [0, np.square(dis_n)]], 'float64')
Q = np.diag([10,10,10,10])
State_all = np.zeros([test_size,timestep_size,dim_state])
Est_all = np.zeros([test_size,timestep_size,dim_state])
Obser_all = np.zeros([test_size,timestep_size,dim_meas])
result_pos = 0
result_vel = 0
import time
for mc in range(1):
    for epoch in range(test_size):
        print('Epoch',epoch)
        data = Test_data[epoch].reshape([timestep_size,dim_state])
        state = data.reshape([timestep_size, dim_state])
        meas = np.zeros([timestep_size, 2])  # 观测数据
        Obser = np.zeros([timestep_size, 2])
        for t in range(timestep_size):
            meas[t, :] = my_hx(state[t, :])
            meas[t, 0] += np.random.normal(0, azi_n)
            meas[t, 1] += np.random.normal(0, dis_n)
            Obser[t, 0] = meas[t, 1] * np.cos(meas[t, 0])
            Obser[t, 1] = meas[t, 1] * np.sin(meas[t, 0])
        Obser_all[epoch] = Obser
        State_all[epoch] = state[:, :]
        kf = EKF(dim_x=dim_state, dim_z=dim_meas)
        start_point = data[0].reshape([dim_state])
        P_start = 500*np.eye(dim_state)
        kf.x = start_point
        kf.P = P_start
        kf.F = F_cv_mat
        kf.Q = Q
        kf.R = R
        est_epoch = np.zeros([timestep_size,dim_state])
        est_epoch[0] = start_point.reshape([dim_state])
        for k in range(1,timestep_size):
            kf.predict()
            kf.update(meas[k],HJacobian=hx_J,Hx=my_hx)
            est_epoch[k] = kf.x.reshape([dim_state])
        Est_all[epoch,:] = est_epoch[:]
    loss_pos = np.sqrt(np.mean(np.square(State_all[:,:,:2]-Est_all[:,:,:2])))
    loss_vel = np.sqrt(np.mean(np.square(np.sqrt(np.square(State_all[:,:,2]+State_all[:,:,3]))-np.sqrt(np.square(Est_all[:,:,2]+Est_all[:,:,3])))))
    result_pos+=loss_pos
    result_vel+=loss_vel
    np.save('EKF_result.npy', Est_all)
    print(1)
print('pos_LOSS:', result_pos )
print('vel_LOSS:', result_vel )
