'''
VBAKF的python实现
Author : 闫实
2022 / 5

'''

import numpy as np
from numpy import pi,cos,sin,tan
from Tool_filter import kf,arvbkf,aprivbkf
from copy import deepcopy
import matplotlib.pyplot as plt

'''模型参数'''
nxp = 6#10 # 仿真次数
nx = 4 # 状态维度
nz = 2 # 量测维度
T = 1. # 采样时间间隔
q = 1 #
r = 1 #

# w1 = 10*np.pi/180
# F_ct = np.array([[1, 0, np.sin(w1 * T) / w1, (np.cos(w1 * T) - 1) / w1],
#                              [0, 1, -(np.cos(w1 * T) - 1) / w1, np.sin(w1 * T) / w1],
#                              [0, 0, np.cos(w1 * T), -np.sin(w1 * T)],
#                              [0, 0, np.sin(w1 * T), np.cos(w1 * T)]])
# w2 = 13*np.pi/180
# F_ct2 = np.array([[1, 0, np.sin(w2 * T) / w2, (np.cos(w2 * T) - 1) / w2],
#                              [0, 1, -(np.cos(w2 * T) - 1) / w2, np.sin(w2 * T) / w2],
#                              [0, 0, np.cos(w2 * T), -np.sin(w2 * T)],
#                              [0, 0, np.sin(w2 * T), np.cos(w2 * T)]])

F = np.array([[1,0,T,0],[0,1,0,T],[0,0,1,0],[0,0,0,1]],'float64') #
H = np.array([[0,0,1,0],[0,0,0,1]],'float64') #
Q1 = np.array([[T**3/3,0,T**2/2,0],[0,T**3/3,0,T/2],[T**2/2,0,T,0],[0,T**2/2,0,T]]) #
R1 = r*np.array([[1,0],[0,1]])
ts = 100

'''重要参数选择'''
N = 10 # 变分迭代次数
tao_P = 3
tao_R = 3
rou = 1-np.exp(-4)
alfa = 1
beta = 100

'''存储'''
# 状态和估计状态
xA = np.zeros([nxp,ts,4,1])
xfA = np.zeros([nxp,ts,4,1])
xtfA = np.zeros([nxp,ts,4,1])
xarvA = np.zeros([nxp,ts,4,1])
xaprivA = np.zeros([nxp,ts,4,1])
# MSE
mse_kf_1 = np.zeros([nxp,ts,1])
mse_kf_2 = np.zeros([nxp,ts,1])
mse_ktf_1 = np.zeros([nxp,ts,1])
mse_ktf_2 = np.zeros([nxp,ts,1])
mse_arvbkf_1 = np.zeros([nxp,ts,1])
mse_arvbkf_2 = np.zeros([nxp,ts,1])
mse_aprivbkf_1 = np.zeros([nxp,ts,1])
mse_aprivbkf_2 = np.zeros([nxp,ts,1])

# 数据读取
Test_data = np.load('data_set.npy')[-6:]

import time
for i in range(10):
    for expt in range(nxp):

        print('MC Run in Process = %d\n'%expt)

        #### 系统初值设置 ####
        x = Test_data[expt,0,:].reshape([4,1])#np.array([[100],[100],[10],[10]]) # 真实状态初值
        P = np.diag([100,100,100,100]) # 初始估计误差协方差矩阵
        Skk = np.sqrt(P) # 初始估计误差协方差矩阵的方根

        #### 名义的噪声协方差矩阵
        Q0 = alfa * np.eye(nx)
        R0 = beta * np.eye(nz)

        #### 标准KF初值
        xf = x# + np.matmul(Skk,np.random.randn(nx,1)) # 状态估计初值
        Pf = P

        #### 真实的KF初值 ####
        xtf = xf
        Ptf = Pf

        #### 现有的vbkf-R初值
        xarv = xf
        Parv = Pf
        uarv = (nz+1+tao_R)
        Uarv = tao_R*R0

        #### 提出的ivbkf-PR初值
        xapriv = xf
        Papriv = Pf
        uapriv = (nz+1+tao_R)
        Uapriv = tao_R*R0

        #### 数据存储
        xA[expt,0] = x
        xfA[expt,0] = xf
        xtfA[expt,0] = xtf
        xarvA[expt,0] = xarv
        xaprivA[expt,0] = xapriv

        # MSE存储
        mse_kf_1[expt,0,0] = (xA[expt,0,0,0] - xfA[expt,0,0,0]) ** 2 + (xA[expt,0,1,0] - xfA[expt,0,1,0]) ** 2
        mse_kf_2[expt,0,0] = (xA[expt,0,2,0] - xfA[expt,0,2,0]) ** 2 + (xA[expt,0,3,0] - xfA[expt,0,3,0]) ** 2

        mse_ktf_1[expt,0,0] = (xA[expt,0,0,0] - xtfA[expt,0,0,0]) ** 2 + (xA[expt,0,1,0] - xtfA[expt,0,1,0]) ** 2
        mse_ktf_2[expt,0,0] = (xA[expt,0,2,0] - xtfA[expt,0,2,0]) ** 2 + (xA[expt,0,3,0] - xtfA[expt,0,3,0]) ** 2

        mse_arvbkf_1[expt,0,0] = (xA[expt,0,0,0] - xarvA[expt,0,0,0]) ** 2 + (xA[expt,0,1,0] - xarvA[expt,0,1,0]) ** 2
        mse_arvbkf_2[expt,0,0] = (xA[expt,0,2,0] - xarvA[expt,0,2,0]) ** 2 + (xA[expt,0,3,0] - xarvA[expt,0,3,0]) ** 2

        mse_aprivbkf_1[expt,0,0] = (xA[expt,0,0,0] - xaprivA[expt,0,0,0]) ** 2 + (xA[expt,0,1,0] - xaprivA[expt,0,1,0]) ** 2
        mse_aprivbkf_2[expt,0,0] = (xA[expt,0,2,0] - xaprivA[expt,0,2,0]) ** 2 + (xA[expt,0,3,0] - xaprivA[expt,0,3,0]) ** 2

        '''初始化存储矩阵'''
        QQ = np.zeros([100, 4, 4])
        RR = np.zeros([100, 2, 2])
        QQ[0] = Q1
        RR[0] = R1
        start = time.time()
        for t in range(1,ts):
            #### 仿真真实的状态和量测
            #### Q R 要缓慢变化
            Q = Q1#np.abs((6.5 + 10 * cos(5 * pi * t/ts)) * Q1)
            R = R1#np.abs((0.1 + 1 * cos(5 * pi * t / ts)) * R1)
            QQ[t] = Q
            RR[t] = R

            #### 计算方根矩阵
            SQ = np.sqrt(Q)
            SR = np.sqrt(R)

            #### 仿真真实的状态和量测
            x = Test_data[expt,t,:].reshape([4,1])#np.matmul(F_ct,x) #+ np.matmul(SQ,np.random.randn(nx,1))
            z = Test_data[expt,t,2:].reshape([2,1])#np.matmul(H,x) + np.matmul(SR,np.random.randn(nz,1))

            #### 调用滤波程序
            xf, Pf, Ppf = kf(xf, Pf, F, H, z, Q0, R0)

            xtf, Ptf, Pptf = kf(xtf, Ptf, F, H, z, Q, R)

            xarv, Parv, uarv, Uarv, Pparv, Rarv = arvbkf(xarv, Parv, uarv, Uarv, F, H, z, Q0, R0, N, rou)

            xapriv, Papriv, uapriv, Uapriv, Ppapriv, Rapriv = aprivbkf(xapriv, Papriv, uapriv, Uapriv, F, H, z, Q0, R0, N, tao_P, rou)

            #### 数据存储
            xA[expt,t] = x
            xfA[expt,t] = xf
            xtfA[expt,t] = xtf
            xarvA[expt,t] = xarv
            xaprivA[expt,t] = xapriv

            # #### MSE计算
            # mse_kf_1[expt, t, 0] = (xA[expt, t, 0, 0] - xfA[expt, t, 0, 0]) ** 2 + (
            #             xA[expt, t, 1, 0] - xfA[expt, t, 1, 0]) ** 2
            #
            # mse_kf_2[expt, t, 0] = (xA[expt, t, 2, 0] - xfA[expt, t, 2, 0]) ** 2 + (
            #             xA[expt, t, 3, 0] - xfA[expt, t, 3, 0]) ** 2
            #
            # mse_ktf_1[expt, t, 0] = (xA[expt, t, 0, 0] - xtfA[expt, t, 0, 0]) ** 2 + (
            #             xA[expt, t, 1, 0] - xtfA[expt, t, 1, 0]) ** 2
            #
            # mse_ktf_2[expt, t, 0] = (xA[expt, t, 2, 0] - xtfA[expt, t, 2, 0]) ** 2 + (
            #             xA[expt, t, 3, 0] - xtfA[expt, t, 3, 0]) ** 2
            #
            # mse_arvbkf_1[expt, t, 0] = (xA[expt, t, 0, 0] - xarvA[expt, t, 0, 0]) ** 2 + (
            #             xA[expt, t, 1, 0] - xarvA[expt, t, 1, 0]) ** 2
            #
            # mse_arvbkf_2[expt, t, 0] = (xA[expt, t, 2, 0] - xarvA[expt, t, 2, 0]) ** 2 + (
            #             xA[expt, t, 3, 0] - xarvA[expt, t, 3, 0]) ** 2
            #
            # mse_aprivbkf_1[expt, t, 0] = (xA[expt, t, 0, 0] - xaprivA[expt, t, 0, 0]) ** 2 + (
            #             xA[expt, t, 1, 0] - xaprivA[expt, t, 1, 0]) ** 2
            #
            # mse_aprivbkf_2[expt, t, 0] = (xA[expt, t, 2, 0] - xaprivA[expt, t, 2, 0]) ** 2 + (
            #             xA[expt, t, 3, 0] - xaprivA[expt, t, 3, 0]) ** 2

        end = time.time()

        print(end-start)
#
# # RMSE计算
# rmse_kf_1 = np.sqrt(np.mean(mse_kf_1,axis=0)).reshape([ts])
# rmse_kf_2 = np.sqrt(np.mean(mse_kf_2,axis=0)).reshape([ts])
#
# rmse_ktf_1 = np.sqrt(np.mean(mse_ktf_1,axis=0)).reshape([ts])
# rmse_ktf_2 = np.sqrt(np.mean(mse_ktf_2,axis=0)).reshape([ts])
#
# rmse_arvbkf_1 = np.sqrt(np.mean(mse_arvbkf_1,axis=0)).reshape([ts])
# rmse_arvbkf_2 = np.sqrt(np.mean(mse_arvbkf_2,axis=0)).reshape([ts])
#
# rmse_aprivbkf_1 = np.sqrt(np.mean(mse_aprivbkf_1,axis=0)).reshape([ts])
# rmse_aprivbkf_2 = np.sqrt(np.mean(mse_aprivbkf_2,axis=0)).reshape([ts])
#
# xA = xA.reshape([nxp,ts,4])
# xfA = xfA.reshape([nxp,ts,4])
# xtfA = xtfA.reshape([nxp,ts,4])
# xarvA = xarvA.reshape([nxp,ts,4])
# xaprivA = xaprivA.reshape([nxp,ts,4])
#
# x_axis = [step for step in range(ts)]
# for i in range(nxp):
#     plt.figure()
#     plt.plot(xA[i,:,0],xA[i,:,1],'b')
#     plt.plot(xfA[i,:,0],xfA[i,:,1],'y')
#     plt.plot(xtfA[i,:,0],xtfA[i,:,1],'g')
#     plt.plot(xarvA[i,:,0],xarvA[i,:,1],'grey')
#     plt.plot(xaprivA[i,:,0],xaprivA[i,:,1],'r')
#     plt.show()
#
#     # plt.figure()
#     # # plt.plot(x_axis,rmse_kf_1,'y')
#     # plt.plot(x_axis,rmse_ktf_1,'g')
#     # # plt.plot(x_axis,rmse_arvbkf_1,'grey')
#     # plt.plot(x_axis,rmse_aprivbkf_1,'r')
#     # plt.show()
#
