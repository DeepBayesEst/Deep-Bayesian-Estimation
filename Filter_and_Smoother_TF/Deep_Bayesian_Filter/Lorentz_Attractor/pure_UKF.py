'''
UKF for tracking
author : Shi Yan
date : 2021/10/25
'''
from filterpy.kalman import JulierSigmaPoints as SP
from filterpy.kalman import UnscentedKalmanFilter as UKF
# from filterpy.kalman import
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D

timestep_size = 2000
test_size = 1
dim_state = 3
dim_meas = 3
batch_size = 1
r2 = 1

dir = 'full/0dB'
T_Data = np.load('data/%s/Test_data.npy'%dir)

def F_lor_part(x,dt):
    x = x.reshape([1,3])
    delta_t = 0.02
    Const = np.array([[-10, 10, 0],
                  [28, -1, 0],
                  [0, 0, -8 / 3]], 'float64')
    J = 5
    BX = np.array([[[0,0,0],
                    [0,0,-x[0,0]],
                    [0,x[0,0],0]]])
    A = np.add(BX, Const)
    F = np.eye(dim_meas)
    for j in range(1,J+1):
        Mat = A * delta_t
        if j == 1:
            F_add = Mat/ math.factorial(j)
            F = np.add(F, F_add)
        # # else:
        # #     F_add = np.matmul(Mat,Mat) / math.factorial(j)
        # #     F = np.add(F, F_add)
        # elif j==2:
        #     F_add = np.matmul(Mat,Mat) / math.factorial(j)
        #     F = np.add(F, F_add)
        # elif j==3:
        #     F_add = np.matmul(np.matmul(Mat,Mat),Mat) / math.factorial(j)
        #     F = np.add(F, F_add)
        # elif j==4:
        #     F_add = np.matmul(np.matmul(np.matmul(Mat,Mat),Mat),Mat) / math.factorial(j)
        #     F = np.add(F, F_add)
        # else:
        #     F_add = np.matmul(np.matmul(np.matmul(np.matmul(Mat, Mat), Mat), Mat),Mat) / math.factorial(j)
        #     F = np.add(F, F_add)
    x = F.reshape([dim_state, dim_state]) @ x.reshape([dim_state, 1])
    return x.reshape([dim_state])

def H_linear():
    roll_deg = yaw_deg = pitch_deg = 10
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
    H = H_Rotate.reshape([dim_state, dim_state])# [batch_size, n, n] rotated matrix
    return H
#
# H =

def my_hx(x):
    H = H_linear()#np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    z = H @ x.reshape([dim_state, 1])
    return z.reshape([dim_meas])

R = np.array([[r2, 0,0], [0, r2,0],[0,0, r2]], 'float64')
Q = np.diag([0.01,0.01,0.01])
State_all = np.zeros([test_size,timestep_size,dim_state])
Est_all = np.zeros([test_size,timestep_size,dim_state])

for epoch in range(test_size):
    print('Epoch %d'%epoch)
    data = T_Data[epoch].reshape([timestep_size,dim_meas+dim_state])
    start = data[0, dim_state:].reshape([dim_state])
    state = data[:, dim_state:].reshape([timestep_size, dim_state])
    meas = data[:, :dim_state].reshape([timestep_size, dim_state])
    State_all[epoch] = state
    # UKF初始化
    my_SP = SP(dim_state, kappa=0.)
    kf = UKF(dim_x=3, dim_z=3, dt=.02,fx=F_lor_part,hx=my_hx,points=my_SP)
    start_point = start.reshape([dim_state])#meas[0].reshape([dim_state])
    P_start = np.eye(dim_state)
    kf.x = start_point
    kf.P = P_start
    # kf.F = F
    # kf.H = H
    kf.Q = Q
    kf.R = R
    est_epoch = np.zeros([timestep_size,dim_meas])
    est_epoch[0] = start_point.reshape([dim_meas])
    for k in range(1,timestep_size):
        kf.predict()
        kf.update(meas[k])
        est_epoch[k] = kf.x.reshape([dim_meas])
    # print(np.mean(np.square(state-est_epoch[1:])))
    # plt.figure()
    # plt.plot(est_epoch[:,0],est_epoch[:,1])
    # plt.plot(state[:,0],state[:,1])
    # plt.show()
    Est_all[epoch] = est_epoch[:]

# np.save('./plot_traj_ukf/noH/ukf_10dB.npy',Est_all)

loss_kf = np.mean(np.square(State_all-Est_all))
print('KF_LOSS:',10*np.log10(loss_kf))
