import numpy as np
from IMM import IMMEstimator
from KF_filter import KalmanFilter as KF
from EKF_filter import ExtendedKalmanFilter as EKF
from filterpy.kalman import JulierSigmaPoints as SP
from filterpy.kalman import UnscentedKalmanFilter as UKF
import time
import matplotlib.pyplot as plt
from copy import deepcopy
import math

Test_data = np.load('Train_data.npy')[234:]

data_len = 200
sT = 4.

dim_state = 4
dim_meas = 2

'''滤波参数'''
MC_num = 100

# 模拟目标运动过程，观测站对目标观测获取距离数据
azi_n = 0.3 * np.pi / 180
dis_n = 150
R_mat = np.array([[np.square(azi_n), 0], [0, np.square(dis_n)]], 'float64')

# 模型数
model_num = 3

# 量测矩阵
# H_mat = np.array([[1., 0., 0., 0.],[0, 1., 0., 0.]],'float64')

def f_cv(x,dt):
    F_cv = np.array([[1, 0, dt, 0],
                 [0, 1, 0, dt],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]], 'float64')
    x_out = F_cv.reshape([dim_state, dim_state]) @ x.reshape([dim_state, 1])
    return x_out.reshape([dim_state])

B_mat = np.array([[0.5 * np.square(sT), 0], [0, 0.5 * np.square(sT)], [sT, 0], [0, sT]])

u1 = np.array([[0],[0]])
u2 = np.array([[20],[20]])
u3 = np.array([[-20],[-20]])

def f_cv1(x,dt):
    F_cv = np.array([[1, 0, dt, 0],
                 [0, 1, 0, dt],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]], 'float64')
    x_out = F_cv.reshape([dim_state, dim_state]) @ x.reshape([dim_state, 1]) + B_mat@u2
    return x_out.reshape([dim_state])

def f_cv2(x,dt):
    F_cv = np.array([[1, 0, dt, 0],
                 [0, 1, 0, dt],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]], 'float64')
    x_out = F_cv.reshape([dim_state, dim_state]) @ x.reshape([dim_state, 1]) + B_mat@u2
    return x_out.reshape([dim_state])

def f_ct1(x,dt):
    w1 = 5 * np.pi / 180
    temp0 = [1, 0, np.sin(w1 * dt) / w1, (np.cos(w1 * dt) - 1) / w1]
    temp1 = [0, 1, -(np.cos(w1 * dt) - 1) / w1, np.sin(w1 * dt) / w1]
    temp2 = [0, 0, np.cos(w1 * dt), -np.sin(w1 * dt)]
    temp3 = [0, 0, np.sin(w1 * dt), np.cos(w1 * dt)]
    F_ct_1 = np.array([temp0, temp1, temp2, temp3], dtype='float64')
    x_out = F_ct_1.reshape([dim_state, dim_state]) @ x.reshape([dim_state, 1])
    return x_out.reshape([dim_state])

def f_ct2(x,dt):
    w2 = -5 * np.pi / 180
    temp0 = [1, 0, np.sin(w2 * dt) / w2, (np.cos(w2 * dt) - 1) / w2]
    temp1 = [0, 1, -(np.cos(w2 * dt) - 1) / w2, np.sin(w2 * dt) / w2]
    temp2 = [0, 0, np.cos(w2 * dt), -np.sin(w2 * dt)]
    temp3 = [0, 0, np.sin(w2 * dt), np.cos(w2 * dt)]
    F_ct_2 = np.array([temp0, temp1, temp2, temp3], dtype='float64')
    x_out = F_ct_2.reshape([dim_state, dim_state]) @ x.reshape([dim_state, 1])
    return x_out.reshape([dim_state])

# 量测雅可比矩阵
def hx_J(x):
    x = x.reshape([dim_state])
    H_mat  = np.array([[-x[1] / (np.square(x[0]) + np.square(x[1])),
              x[0] / (np.square(x[0]) + np.square(x[1])), 0., 0.],
                      [x[0] / np.sqrt((np.square(x[0]) + np.square(x[1]))),
              x[1] / np.sqrt((np.square(x[0]) + np.square(x[1]))), 0., 0.]])
    return H_mat.reshape([dim_meas,dim_state])

# 量测方程
def my_hx(x):
    """ returns slant range = np.array([[0],[0]]) based on downrange distance and altitude"""
    r0 = np.arctan2(x[1], x[0])
    r1 = np.sqrt(np.square(x[0]) + np.square(x[1]))
    r = np.stack([r0, r1])
    return r.reshape([dim_meas])

# 各个状态转移矩阵
F_cv_mat = np.array([[1, 0, sT, 0], [0, 1, 0, sT], [0, 0, 1, 0], [0, 0, 0, 1]], 'float64')

Q11 = np.diag([15,15,15,15])

Imm_result = np.zeros([MC_num,data_len,dim_state])
rmse_x = np.zeros([data_len])
rmse_y = np.zeros([data_len])
rmse_vx = np.zeros([data_len])
rmse_vy = np.zeros([data_len])
rmse_meas_x = np.zeros([data_len])
rmse_meas_y = np.zeros([data_len])
error_x = np.zeros([data_len])
error_y = np.zeros([data_len])
error_vx = np.zeros([data_len])
error_vy = np.zeros([data_len])
error_meas_x = np.zeros([data_len])
error_meas_y = np.zeros([data_len])
Model_prob = np.zeros([MC_num,data_len,model_num])
mostlike_m_index = None
pi_arr = None
result_pos = 0
result_vel = 0

IMM_out = np.zeros([26,100,200,4])
xi = .2
for iii in range(26):
    ori_traj = Test_data[iii]

    traj = ori_traj.reshape([data_len, 4])
    dt = np.zeros([traj.shape[0], 4])
    dt[:, 0] = traj[:, 0]
    dt[:, 1] = traj[:, 1]
    dt[:, 2] = traj[:, 2]
    dt[:, 3] = traj[:, 3]
    State_all = np.zeros([MC_num,data_len-1,dim_state])

    for i in range(MC_num):
        starttime = time.time()

        State_all[i] = dt[1:, :]

        meas_gau = np.zeros([data_len, 2])  # 观测数据
        Obser_gau = np.zeros([data_len, 2])

        meas = np.zeros([data_len, 2])  # 观测数据
        Obser = np.zeros([data_len, 2])
        for t in range(data_len):
            meas[t, :] = my_hx(dt[t, :])

            meas_gau[t, 0] = deepcopy(meas[t, 0])+np.random.normal(0, azi_n)  # 方位角#np.random.normal(0, azi_n)
            meas_gau[t, 1] = deepcopy(meas[t, 1])+np.random.normal(0, dis_n)#np.random.normal(0, dis_n)

            # meas[t, 0] += np.random.normal(0, azi_n)#(1-xi)*np.random.normal(0, azi_n)+xi*np.random.laplace(0,azi_n)  # 方位角#np.random.normal(0, azi_n)
            # meas[t, 1] += np.random.normal(0, dis_n)#(1-xi)*np.random.normal(0, dis_n)+xi*np.random.laplace(0,dis_n)#np.random.normal(0, dis_n)
            meas[t, 0] += (1-xi)*np.random.normal(0, azi_n)+xi*np.random.laplace(0,azi_n*2)  # 方位角#np.random.normal(0, azi_n)
            meas[t, 1] += (1-xi)*np.random.normal(0, dis_n)+xi*np.random.laplace(0,dis_n*2)#np.random.normal(0, dis_n)

            Obser_gau[t, 0] = meas_gau[t, 1] * np.cos(meas_gau[t, 0])
            Obser_gau[t, 1] = meas_gau[t, 1] * np.sin(meas_gau[t, 0])
            Obser[t, 0] = meas[t, 1] * np.cos(meas[t, 0])
            Obser[t, 1] = meas[t, 1] * np.sin(meas[t, 0])

        print(np.sqrt(np.mean(np.square(Obser -dt[:,:2]))))

        my_SP1 = SP(dim_state, kappa=0.)
        my_SP2 = SP(dim_state, kappa=0.)
        my_SP3 = SP(dim_state, kappa=0.)

        # print('Obser',np.sqrt(np.mean(np.square(dt[:, :2] - Obser))))
        # print('Obser_gau',np.sqrt(np.mean(np.square(dt[:, :2] - Obser_gau))))

        # plt.plot(dt[:, 0], dt[:, 1])
        # plt.plot(Obser_gau[:, 0], Obser_gau[:, 1],'r')
        # plt.plot(Obser[:, 0], Obser[:, 1],'b')
        # plt.show()
        kf1 = UKF(dim_x=dim_state, dim_z=dim_meas, dt=sT,fx=f_cv,hx=my_hx,points=my_SP1)
        kf2 = UKF(dim_x=dim_state, dim_z=dim_meas, dt=sT,fx=f_ct1,hx=my_hx,points=my_SP2)
        kf3 = UKF(dim_x=dim_state, dim_z=dim_meas, dt=sT,fx=f_ct2,hx=my_hx,points=my_SP3)


        # 设置各个滤波器的参数
        start_point1 = dt[0].reshape([dim_state,1]) #
        start_point2 = deepcopy(start_point1)
        start_point3 = deepcopy(start_point1)

        kf1.x = start_point1
        kf2.x = start_point2
        kf3.x = start_point3
        kf1.Q = Q11
        kf2.Q = Q11
        kf3.Q = Q11
        kf1.R = R_mat
        kf2.R = R_mat
        kf3.R = R_mat

        kf1.P = np.eye(4)#*500
        kf2.P = np.eye(4)#*500
        kf3.P = np.eye(4)#*500

        filters = [kf1, kf2,kf3]
        # 模型概率
        mu = [0.4, 0.3,0.3]  # each filter is equally likely at the start
        # 转移概率
        trans = np.array([[0.95, 0.025,0.025], [0.025, 0.95,0.025],[0.025,0.025,0.95]])
        # trans = np.array([[1., 0.,0.], [0., 1.,0.],[0.,0.,1.]])

        imm = IMMEstimator(filters=filters, mu=mu, M=trans)
        Imm_result[i,0,:] = start_point1.reshape([dim_state])
        # if i == 0:
        IMM_out[iii,i,0,:] = start_point1.reshape([dim_state])

        # 用于统计RMSE的一些数组
        error_x[0] += np.square(start_point1[0] - dt[0,0])
        error_y[0] += np.square(start_point1[1] - dt[0,1])
        error_vx[0] += np.square(start_point1[2] - dt[0,2])
        error_vy[0] += np.square(start_point1[3] - dt[0,3])
        error_meas_x[0] += np.square(Obser[0,0]- dt[0,0])
        error_meas_y[0] += np.square(Obser[0,1]- dt[0,1])
        # 滑窗所需的储存数组
        mu_slid_lst = []
        like_slid_lst = []
        state1_slid_lst = []
        state2_slid_lst = []
        state3_slid_lst = []
        P1_slid_lst = []
        P2_slid_lst = []
        P3_slid_lst = []
        imm_state_lst = []
        imm_P_lst = []
        sliding_index = 0
        for j in range(1,data_len):
            # 初始化粒子群算法对象
            # 滑窗启动阶段
            # print('step',j)

            kf1.x = kf1.x.reshape([dim_state])
            kf1.predict()
            kf1.update(meas[j,:])
            Imm_result[i,j,:] = kf1.x.reshape([dim_state])
            # if i == 0:
            IMM_out[iii, i,j, :] = kf1.x.reshape([dim_state])
            error_x[j] += np.square(kf1.x[0] - dt[j, 0])
            error_y[j] += np.square(kf1.x[1] - dt[j, 1])
            error_vx[j] += np.square(kf1.x[2] - dt[j, 2])
            error_vy[j] += np.square(kf1.x[3] - dt[j, 3])
            error_meas_x[j] += np.square(Obser[j, 0] - dt[j, 0])
            error_meas_y[j] += np.square(Obser[j, 1] - dt[j, 1])
            Model_prob[i,j,:] = imm.mu
        endtime = time.time()
        print('总共的时间为:', round(endtime - starttime, 2), 'secs')
        print(1)
    loss_pos = np.sqrt(np.mean(np.square(State_all[:,:,:2]-Imm_result[:,1:,:2])))
    loss_vel = np.sqrt(np.mean(np.square(np.sqrt(np.square(State_all[:,:,2]+State_all[:,:,3]))-np.sqrt(np.square(Imm_result[:,1:,2]+Imm_result[:,1:,3])))))
    result_pos+=loss_pos
    result_vel+=loss_vel

np.save('Imm_result_l4_glint_low.npy',IMM_out)

print(result_pos/26)
print(result_vel/26)
