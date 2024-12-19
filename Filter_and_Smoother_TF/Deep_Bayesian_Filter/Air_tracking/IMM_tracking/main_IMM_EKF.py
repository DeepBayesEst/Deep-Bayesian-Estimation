import numpy as np
from IMM import IMMEstimator
from KF_filter import KalmanFilter as KF
from EKF_filter import ExtendedKalmanFilter as EKF

import matplotlib.pyplot as plt
from copy import deepcopy
import math

Test_data = np.load('Train_data.npy')[:26]

data_len = 200
sT = 4.

dim_state = 4
dim_meas = 2

'''滤波参数'''
MC_num = 10

# 模拟目标运动过程，观测站对目标观测获取距离数据
azi_n = 0.05 * np.pi / 180
dis_n = 50
R_mat = np.array([[np.square(azi_n), 0], [0, np.square(dis_n)]], 'float64')

# 模型数
model_num = 3

# 量测矩阵
# H_mat = np.array([[1., 0., 0., 0.],[0, 1., 0., 0.]],'float64')

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
w1 = 1.5 * np.pi / 180
temp0 = [1, 0, np.sin(w1 * sT) / w1, (np.cos(w1 * sT) - 1) / w1]
temp1 = [0, 1, -(np.cos(w1 * sT) - 1) / w1, np.sin(w1 * sT) / w1]
temp2 = [0, 0, np.cos(w1 * sT), -np.sin(w1 * sT)]
temp3 = [0, 0, np.sin(w1 * sT), np.cos(w1 * sT)]
F_ct_1 = np.array([temp0, temp1, temp2, temp3], dtype='float64')
# F_ct_1 = np.array([[1, 0, sT, 0], [0, 1, 0, sT], [0, 0, 1, 0], [0, 0, 0, 1]], 'float64')
w2 = -1.5 * np.pi / 180
temp0 = [1, 0, np.sin(w2 * sT) / w2, (np.cos(w2 * sT) - 1) / w2]
temp1 = [0, 1, -(np.cos(w2 * sT) - 1) / w2, np.sin(w2 * sT) / w2]
temp2 = [0, 0, np.cos(w2 * sT), -np.sin(w2 * sT)]
temp3 = [0, 0, np.sin(w2 * sT), np.cos(w2 * sT)]
F_ct_2 = np.array([temp0, temp1, temp2, temp3], dtype='float64')

B_mat = np.array([[0.5 * np.square(sT), 0], [0, 0.5 * np.square(sT)], [sT, 0], [0, sT]])

u1 = np.array([[0],[0]])
u2 = np.array([[25],[0]])
u3 = np.array([[-25],[0]])

# 过程噪声协方差矩阵
Q11 = np.diag([5,5,5,5])

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

ori_traj = Test_data[1]

traj = ori_traj.reshape([data_len, 4])
dt = np.zeros([traj.shape[0], 4])
dt[:, 0] = traj[:, 0]
dt[:, 1] = traj[:, 1]
dt[:, 2] = traj[:, 2]
dt[:, 3] = traj[:, 3]

for i in range(MC_num):

    meas = np.zeros([data_len, 2])  # 观测数据
    Obser = np.zeros([data_len, 2])
    for t in range(data_len):
        meas[t, :] = my_hx(dt[t, :])
        meas[t, 0] += np.random.normal(0, azi_n)
        meas[t, 1] += np.random.normal(0, dis_n)
        Obser[t, 0] = meas[t, 1] * np.cos(meas[t, 0])
        Obser[t, 1] = meas[t, 1] * np.sin(meas[t, 0])

    kf1 = EKF(dim_x=dim_state, dim_z=dim_meas,HJacobian=hx_J,Hx=my_hx)
    kf2 = EKF(dim_x=dim_state, dim_z=dim_meas,HJacobian=hx_J,Hx=my_hx)
    kf3 = EKF(dim_x=dim_state, dim_z=dim_meas,HJacobian=hx_J,Hx=my_hx)
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
    kf1.F = F_cv_mat
    kf2.F = F_ct_1
    kf3.F = F_ct_2
    kf1.P = np.eye(4)*1000
    kf2.P = np.eye(4)*1000
    kf3.P = np.eye(4)*1000

    filters = [kf1, kf2,kf3]
    # 模型概率
    mu = [0.8, 0.1,0.1]  # each filter is equally likely at the start
    # 转移概率
    # trans = np.array([[0.4, 0.3,0.3], [0.3, 0.4,0.4],[0.4,0.3,0.3]])
    trans = np.array([[0.9, 0.05,0.05], [0.1, 0.8,0.1],[0.1,0.1,0.8]])
    # trans = np.array([[0.6, 0.2,0.2], [0.2, 0.6,0.2],[0.2,0.2,0.6]])

    imm = IMMEstimator(filters=filters, mu=mu, M=trans)
    Imm_result[i,0,:] = start_point1.reshape([dim_state])
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
        print('step',j)
        imm.predict() # 状态交互，滤波器预测
        imm.filter_update(meas[j,:])

        imm.model_probability_update()
        imm.update()
        Imm_result[i,j,:] = imm.x.reshape([dim_state])
        error_x[j] += np.square(imm.x[0] - dt[j, 0])
        error_y[j] += np.square(imm.x[1] - dt[j, 1])
        error_vx[j] += np.square(imm.x[2] - dt[j, 2])
        error_vy[j] += np.square(imm.x[3] - dt[j, 3])
        error_meas_x[j] += np.square(Obser[j, 0] - dt[j, 0])
        error_meas_y[j] += np.square(Obser[j, 1] - dt[j, 1])
        Model_prob[i,j,:] = imm.mu

'''计算和绘制RMSE'''
Rmse_dist = np.zeros([data_len])
Rmse_vel = np.zeros([data_len])
Rmse_meas = np.zeros([data_len])
for j in range(data_len):
    rmse_x[j] = np.sqrt(error_x[j]/MC_num)
    rmse_y[j] = np.sqrt(error_y[j]/MC_num)
    rmse_vx[j] = np.sqrt(error_vx[j]/MC_num)
    rmse_vy[j] = np.sqrt(error_vy[j]/MC_num)
    rmse_meas_x[j] = np.sqrt(error_meas_x[j]/MC_num)
    rmse_meas_y[j] = np.sqrt(error_meas_y[j]/MC_num)
    Rmse_dist[j] = np.sqrt(rmse_x[j]**2+rmse_y[j]**2)
    Rmse_vel[j] = np.sqrt(rmse_vx[j]**2+rmse_vy[j]**2)
    Rmse_meas[j] = np.sqrt(rmse_meas_x[j]**2+rmse_meas_y[j]**2)

result = Imm_result[0]


print(np.mean(Rmse_dist))

print(np.mean(Rmse_vel))

plt.figure()
plt.plot(Rmse_dist)
plt.plot(Rmse_meas)
plt.show()
plt.figure()
plt.plot(Rmse_vel)
plt.show()






