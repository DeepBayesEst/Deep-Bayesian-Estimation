'''
Particle filter for tracking
author : Shi Yan
date : 2021/10/25
'''
import random
import math
import numpy as np
# import matlab
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from copy import deepcopy
# Random resampling function
def randomR(inIndex,q):
    outIndex = np.zeros([inIndex],'int')
    num = q.shape[0]
    # col = q.shape[1]
    u = np.random.rand(num)
    u.sort()
    l = np.cumsum(q)
    i = 0
    for j in range(num):
        while i < num and u[i] <= l[j]:
            outIndex[i] = j
            i = i + 1
    return outIndex
class Obser_station():
    def __init__(self):
        self.x = None
        self.y = None
class Parameters():
    def __init__(self):
        self.T = None
        self.M = None
        self.Q = None
        self.R = None
        self.F = None
        self.state0 = None

T = 0.02 #
sT = T
dim_meas = 3
dim_state = 3
M = 2000 #
r2=1

dir = 'full/0dB'
# dir_model = 'full_same_len/T%d/20dB'%timestep_size
Test_data = np.load('data/%s/Test_data.npy'%dir)[:,:M]
print(Test_data.shape)
Test_res = np.load('./result_test/%s/Test_result.npy'% dir)

R = np.array([[r2, 0,0], [0, r2,0],[0,0, r2]], 'float64')
Q = np.diag([0.01,0.01,0.01])
H = np.array([[1., 0., 0.],[0., 1., 0.],[0., 0., 1.]])
# def H_linear():
#     roll_deg = yaw_deg = pitch_deg = 10
#     roll = roll_deg * (math.pi / 180)
#     yaw = yaw_deg * (math.pi / 180)
#     pitch = pitch_deg * (math.pi / 180)
#     RX = np.array([[1, 0, 0],
#                    [0, math.cos(roll), -math.sin(roll)],
#                    [0, math.sin(roll), math.cos(roll)]], 'float64')
#     RY = np.array([[math.cos(pitch), 0, math.sin(pitch)],
#                    [0, 1, 0],
#                    [-math.sin(pitch), 0, math.cos(pitch)]], 'float64')
#     RZ = np.array([[math.cos(yaw), -math.sin(yaw), 0],
#                    [math.sin(yaw), math.cos(yaw), 0],
#                    [0, 0, 1]], 'float64')
#     RotMatrix = RZ @ RY @ RX
#     H_Rotate = RotMatrix @ np.eye(dim_state)
#     H = H_Rotate.reshape([dim_state, dim_state])# [batch_size, n, n] rotated matrix
#     return H
#
# H = H_linear()
# def F_lor_part(x,dt):
#     x = x.reshape([1,3])
#     delta_t = 0.02
#     Const = np.array([[-10, 10, 0],
#                   [28, -1, 0],
#                   [0, 0, -8 / 3]], 'float64')
#     J = 5
#     BX = np.array([[[0,0,0],
#                     [0,0,-x[0,0]],
#                     [0,x[0,0],0]]])
#     A = np.add(BX, Const)
#     F = np.eye(dim_meas)
#     for j in range(1,J+1):
#         Mat = A * delta_t
#         if j == 1:
#             F_add = Mat/ math.factorial(j)
#             F = np.add(F, F_add)
#         # else:
#         #     F_add = np.matmul(Mat,Mat) / math.factorial(j)
#         #     F = np.add(F, F_add)
#         elif j==2:
#             F_add = np.matmul(Mat,Mat) / math.factorial(j)
#             F = np.add(F, F_add)
#         elif j==3:
#             F_add = np.matmul(np.matmul(Mat,Mat),Mat) / math.factorial(j)
#             F = np.add(F, F_add)
#         elif j==4:
#             F_add = np.matmul(np.matmul(np.matmul(Mat,Mat),Mat),Mat) / math.factorial(j)
#             F = np.add(F, F_add)
#         else:
#             F_add = np.matmul(np.matmul(np.matmul(np.matmul(Mat, Mat), Mat), Mat),Mat) / math.factorial(j)
#             F = np.add(F, F_add)
#     x = F.reshape([dim_state, dim_state]) @ x.reshape([dim_state, 1])
#     return x.reshape([dim_state,1])

def F_lor_part(x):
    x = x.reshape([1,3])
    delta_t = 0.02
    Const = np.array([[-10, 10, 0],
                  [28, -1, 0],
                  [0, 0, -8 / 3]], 'float64')
    J = 1
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
        # else:
        #     F_add = np.matmul(Mat,Mat) / math.factorial(j)
        #     F = np.add(F, F_add)
        elif j==2:
            F_add = np.matmul(Mat,Mat) / math.factorial(j)
            F = np.add(F, F_add)
        elif j==3:
            F_add = np.matmul(np.matmul(Mat,Mat),Mat) / math.factorial(j)
            F = np.add(F, F_add)
        elif j==4:
            F_add = np.matmul(np.matmul(np.matmul(Mat,Mat),Mat),Mat) / math.factorial(j)
            F = np.add(F, F_add)
        else:
            F_add = np.matmul(np.matmul(np.matmul(np.matmul(Mat, Mat), Mat), Mat),Mat) / math.factorial(j)
            F = np.add(F, F_add)
    return F.reshape([dim_state,dim_state])

######## The target's trajectory  #########
X = np.zeros([M,3]) # Target state
Z = np.zeros([M,3]) # Observational data

MC_num = 1
Result = np.zeros([1,MC_num,M,3])
True_ = np.zeros([1,MC_num,M,3])
Meas = np.zeros([1,MC_num,M,3])

for ii in range(1):# Test_data.shape[0]
    print(ii)
    for mc in range(MC_num):
        X = Test_data[ii,:,dim_state:]# Initialize the target state
        Z = Test_data[ii,:,:dim_state]# Initialize target measurement
        state0 = X[0,:]
        # Target trajectory plotting
        ###### Initialize the particle filter #######
        N = 100   # Number of particles
        zPred = np.zeros([N,3]) # Measurement prediction
        Weight = np.ones([N]) # Weight
        Weight = Weight/np.sum(Weight)
        xparticlePred = np.zeros([N,3]) # State prediction particle
        Xout = np.zeros([M,3]) # State Output
        Xout[0,:] = state0 # Initial state
        xparticle = np.zeros([N,3]) # State Particle

        for j in range(N):
            xparticle[j,:] = state0 # Assign initial values ​​to state particles, all of which are in initial state
        for t in range(1,M): #
            for k in range(N):
                # Prediction
                Trans = F_lor_part(xparticle[k,:])
                temp = Trans@xparticle[k,:].reshape([3,1]) + np.dot(np.sqrt(Q),np.random.randn(3,1))
                xparticlePred[k,:] = temp.reshape([3])
            # Update
            for k in range(N):
                tempz = H@xparticlePred[k].reshape([3,1])
                zPred[k] = tempz.reshape([3])
                z1 = Z[t] - zPred[k]
                # Likelihood
                Weight[k] = 1/np.sqrt(np.linalg.det(2 * np.pi * R)) * np.exp(-0.5 * np.dot(np.dot(z1.T,np.linalg.inv(R)),z1))
                Weight[k]+=1e-20
                #*Weight_last[k] #+ 1e-99
            # print(Weight)
            # Normalized weights
            Weight = np.array(Weight,'float64')
            # print(Weight)
            Weight =Weight/np.sum(Weight) # np.random.dirichlet(Weight) #
            # target = np.zeros([4])
            # for k in range(N):
            #     target +=  Weight[k,0] * xparticlePred[k,:]
            target = np.average(xparticlePred, weights=Weight, axis=0)
            var = np.average((xparticlePred - target) ** 2, weights=Weight, axis=0)
            # Resampling
            outIndex = randomR(N,Weight) #
            xparticle = xparticlePred[outIndex,:] #
            target = np.array([np.mean(xparticle[:,0]),np.mean(xparticle[:,1]),np.mean(xparticle[:,2])])
            Xout[t,:] = target
        Result[ii,mc,:] = Xout
        True_[ii,mc,:] = X
        Meas[ii,mc,:] = Z

loss_pf = np.mean(np.square(Result-True_))
print('KF_LOSS:',10*np.log10(loss_pf))