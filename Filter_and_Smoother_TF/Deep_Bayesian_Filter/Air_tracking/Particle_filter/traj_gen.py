#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
基于DeepMTT源码修改
轨迹生成
Author:闫实 Date: 2021/12/10
'''

import numpy as np
import random as rd
from numpy.linalg import cholesky


# 采样率
sT = 1
# 距离范围
Dist_max = 50000 # 30 * 1.852 * 1000
Dist_min = 10000 # 20 * 1.852 * 1000
# 速度范围
Velo_max1 = 400
Velo_min1 = 100
Velo_max2 = -100
Velo_min2 = -400

## 转弯速率范围
TR_intv = 0.1
TR_min_1 = -10
TR_max_1 = 10
TR_min_2 = -10
TR_max_2 = 10

# 量测误差
# 距离噪声
dis_n = 60
# 角度噪声
azi_n = 0.05 * np.pi / 180

def random_of_ranges(*ranges):
  all_ranges = sum(ranges, [])
  return rd.choice(all_ranges)

# Trajectory generator
def trajectory_creat(data_len, T_matrix1, T_matrix2,T_matrix3, w1, w2,w3):
    state_n = 10  # rd.uniform(2,3)
    s_var = np.square(state_n)
    T2 = np.power(sT, 2)
    T3 = np.power(sT, 3)
    T4 = np.power(sT, 4)
    data = np.array([[0 for i in range(4)] for j in range(data_len)], 'float32')
    distance = np.array([0 for j in range(data_len)], 'float32')
    # Starting point
    sp_distance = rd.uniform(Dist_min, Dist_max)
    # 角度范围
    sp_direction = rd.uniform(-180 * np.pi / 180, 180 * np.pi / 180) # (rd.random()-0.5) * 2 * np.pi #
    d_x = sp_distance * np.cos(sp_direction)  # Target X dirction position
    d_y = sp_distance * np.sin(sp_direction)  # Target Y dirction position
    arr1 = np.random.randint(Velo_min1, Velo_max1)
    arr2 = np.random.randint(Velo_min2, Velo_max2)
    sp_velocity = np.random.choice(np.stack((arr1, arr2)))
    # print('sp_velocity',sp_velocity)
    vel_direction = (rd.random() - 0.5) * 2 * np.pi
    v_x = sp_velocity * np.cos(vel_direction)  # Target X dirction velocity
    v_y = sp_velocity * np.sin(vel_direction)  # Target Y dirction velocity
    X_a = np.array([[d_x, d_y, v_x, v_y]], 'float32')
    a = rd.randint(0, data_len)
    b = rd.randint(0, data_len)
    jump_point1 = 0
    jump_point2 = 0
    T_list = []
    for i in range(data_len):
        data[i, :] = X_a
        distance[i] = np.sqrt(data[i,0]**2+data[i,1]**2)
        # X_a = np.dot(X_a, T_matrix1)
        if i < jump_point1:
            X_a = np.dot(X_a, T_matrix1)
            T_list.append(w1)
        # elif i < jump_point2:
        #     X_a = np.dot(X_a, T_matrix2)
        #     T_list.append(w2)
        else:
            X_a = np.dot(X_a, T_matrix2)
            T_list.append(w2)
    # data_n = data  # + np.dot(np.random.randn(data_len, 4), chol_var)  # 加噪声
    return data, jump_point1,jump_point2,T_list,distance

def trajectory_batch_generator(batch_size, data_len):
    # Initialization
    Traj_r = np.array([[[0 for i in range(4)] for j in range(data_len)] for k in range(batch_size)],
                      'float32')  # Initialization of trajectory
    Obser = np.array([[[0 for i in range(2)] for j in range(data_len)] for k in range(batch_size)],
                     'float32')  # Initialization of observation
    x_start = np.array([[0 for i in range(4)]  for k in range(batch_size)],'float32')  # Initialization of observation
    OB = np.array([[[0 for i in range(3)] for j in range(data_len)] for k in range(batch_size)],
                  'float32')  # Initialization of observation
    w_batch = np.array([[[0 for i in range(1)] for j in range(data_len)] for k in range(batch_size)],
                       'float32')  # Initialization of observation
    Class_array = np.array([0 for k in range(batch_size)],'float32')  # 跳变类别，每个batch一个
    JUMP = np.array([[0 for i in range(1)]for k in range(batch_size)],'float32')  # Initialization of observation
    True_w = np.array([[0 for i in range(data_len)]for k in range(batch_size)],'float32')  # Initialization of observation

    for i in range(batch_size):
        turn_rate1 = rd.randint(TR_min_1 / TR_intv, TR_max_1 / TR_intv) * TR_intv
        turn_rate2 = rd.randint(TR_min_1 / TR_intv, TR_max_1 / TR_intv) * TR_intv
        turn_rate3 = rd.randint(TR_min_1 / TR_intv, TR_max_1 / TR_intv) * TR_intv

        if turn_rate1 == 0:
            F_c1 = np.array([[1, 0, sT, 0], [0, 1, 0, sT], [0, 0, 1, 0], [0, 0, 0, 1]])
        else:
            w1 = turn_rate1 * np.pi / 180
            F_c1 = np.array([[1, 0, np.sin(w1 * sT) / w1, (np.cos(w1 * sT) - 1) / w1],
                             [0, 1, -(np.cos(w1 * sT) - 1) / w1, np.sin(w1 * sT) / w1],
                             [0, 0, np.cos(w1 * sT), -np.sin(w1 * sT)],
                             [0, 0, np.sin(w1 * sT), np.cos(w1 * sT)]])
        F_c1 = np.transpose(F_c1, [1, 0])
        if turn_rate2 == 0:
            F_c2 = np.array([[1, 0, sT, 0], [0, 1, 0, sT], [0, 0, 1, 0], [0, 0, 0, 1]])
        else:
            w2 = turn_rate2 * np.pi / 180
            F_c2 = np.array([[1, 0, np.sin(w2 * sT) / w2, (np.cos(w2 * sT) - 1) / w2],
                             [0, 1, -(np.cos(w2 * sT) - 1) / w2, np.sin(w2 * sT) / w2],
                             [0, 0, np.cos(w2 * sT), -np.sin(w2 * sT)],
                             [0, 0, np.sin(w2 * sT), np.cos(w2 * sT)]])
        F_c2 = np.transpose(F_c2, [1, 0])

        if turn_rate3 == 0:
            F_c3 = np.array([[1, 0, sT, 0], [0, 1, 0, sT], [0, 0, 1, 0], [0, 0, 0, 1]])
        else:
            w3 = turn_rate3 * np.pi / 180
            F_c3 = np.array([[1, 0, np.sin(w3 * sT) / w3, (np.cos(w3 * sT) - 1) / w3],
                             [0, 1, -(np.cos(w3 * sT) - 1) / w3, np.sin(w3 * sT) / w3],
                             [0, 0, np.cos(w3 * sT), -np.sin(w3 * sT)],
                             [0, 0, np.sin(w3 * sT), np.cos(w3 * sT)]])
        F_c3 = np.transpose(F_c3, [1, 0])
        # 轨迹生成
        '''
        现在写的python EKF在计算X负半轴从负角度到正角度的时候arctan计算会出问题，所以我画了个界限
        '''
        J1 = True
        J2 = True
        J3 = True
        J4 = True
        J5 = True
        while J4 or J5:
            dt, jump_point1, jump_point2, T_list,dist = trajectory_creat(data_len, F_c1, F_c2, F_c3,
                                                                        turn_rate1 * np.pi / 180,
                                                                        turn_rate2 * np.pi / 180,
                                                                        turn_rate3 * np.pi / 180)

            J4 = (dist>50000.0).any()
            J5 = (dist<10000.0).any()
            # J1 = (dt[:, 1] > -5000.0).any()
            # J2 = (dt[:, 1] < 5000.0).any()
            # J3 = (dt[:, 0] < 0).any()

        # model_label[i, jump_point2:, :] = label3
        w_batch[i, :jump_point1, :] = turn_rate1 * np.pi / 180
        w_batch[i, jump_point1:, :] = turn_rate2 * np.pi / 180
        # w_batch[i, jump_point2:, :] = turn_rate3 * np.pi / 180
        JUMP[i,0] = jump_point1
        # JUMP[i,1] = jump_point2
        True_w[i] = np.array(T_list)
        # w_batch[i, :, :] = turn_rate1 * np.pi / 180
        Traj_r[i,:,0] = dt[:,0]# + 150000
        Traj_r[i,:,1] = dt[:,1]# + 150000
        Traj_r[i,:,2] = dt[:,2]
        Traj_r[i,:,3] = dt[:,3]

        # 生成极坐标量测
        OB[i, :, 0] = np.arctan2(dt[:, 1], dt[:, 0]) + np.random.normal(0, azi_n,data_len)   # 方位角
        OB[i, :, 1] = np.sqrt(np.square(dt[:, 0]) + np.square(dt[:, 1])) + np.random.normal(0, dis_n,
                                                                                              data_len)   # 径向距
        OB[i, :, 2] = (dt[:,0]*dt[:,2]+dt[:,1]*dt[:,3])/np.sqrt(np.square(dt[:,0])+np.square(dt[:,1])) \
                      + np.random.normal(0, 5, data_len)  # 径向速度
        # 转换为x、y坐标下的量测
        Obser[i, :, 0] = OB[i, :, 1] * np.cos(OB[i, :, 0])
        Obser[i, :, 1] = OB[i, :, 1] * np.sin(OB[i, :, 0])
        # 生成起始点
        x_start[i,0] = Obser[i, 0, 0]
        x_start[i,1] = Obser[i, 0, 1]
        x_start[i,2] = Traj_r[i,0,2] + np.random.normal(0, 20)
        x_start[i,3] = Traj_r[i,0,3] + np.random.normal(0, 20)
    INPUT = OB
    return Traj_r, Obser, w_batch, INPUT,Class_array,x_start,JUMP,True_w,OB










