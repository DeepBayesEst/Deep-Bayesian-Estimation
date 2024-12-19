'''
Particle for target tracking
author : Yan Shi
date : 2021/10/25
This is a particle filter code written in tensorflow for single target tracking.
Batch processing is used to accelerate computing.
'''
import matplotlib.pyplot as plt
import tensorflow as tf
from traj_gen import *
import numpy as np
# import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
batch_size = 64
sT = 1  # 采样周期
T2 = np.power(sT, 2)
T3 = np.power(sT, 3)
T4 = np.power(sT, 4)
timestep_size = 60  # Time duration
state_dim = 4  # State Dimension
meas_dim = 2  # Measurement Dimension
num_particle = 100 # particle number

Traj_r, Obser, w_batch, INPUT, Class_array, x_start, JUMP, True_w, OB = trajectory_batch_generator(batch_size,timestep_size)

# Target state vector
x_true = tf.stack(Traj_r)  # placeholder(dtype=tf.float64,shape=[None, timestep_size,state_dim],name='x_true') # Traj_r
# Target measurement vector
z_meas = tf.stack(OB[:, :, :2])  # tf.placeholder(dtype=tf.float64,shape=[None, timestep_size,meas_dim],name='z_meas') # OB[:,:,:2]
# Target measurement Cartesian coordinate system
z_obser = Obser
# Process noise
ax = 35  # x direction
ay = 35  # y direction

# Process noise covariance matrix
ax2 = np.square(ax)
ay2 = np.square(ay)
Q_mat = np.array([[T4 / 4 * ax2, 0, T3 / 2 * ax2, 0],
                  [0, T4 / 4 * ay2, 0, T3 / 2 * ay2],
                  [T3 / 2 * ax2, 0, T2 * ax2, 0],
                  [0, T3 / 2 * ay2, 0, T2 * ay2]])
Q = np.array([[Q_mat for _ in range(num_particle)] for _ in range(batch_size)], 'float64')

# Radar measurement noise
azi_n = 0.05 * np.pi / 180  # Azimuth noise
dis_n = 60  # Radial noise

# Measurement noise covariance matrix
R_mat = np.array([[np.square(azi_n), 0], [0, np.square(dis_n)]], 'float64')
R = np.array([[R_mat for _ in range(num_particle)] for _ in range(batch_size)], 'float64')

# Weight vector
weight = None

# Particle Prediction
x_particle_pred = None  # tf.zeros([batch_size,num_particle,4])

# State estimation
x_est = []
# Assign initial value to state particle
x_particle = tf.transpose(tf.stack([x_true[:, 0, :] for _ in range(num_particle)]), [1, 0, 2])
weight_norm = None

# Measurement model
def my_hx(x):
    # batch_size * num_particle * 4 * 1
    """ returns slant range = np.array([[0],[0]]) based on downrange distance and altitude"""
    r0 = tf.atan2(x[:,:, 1], x[:,:, 0]) # batch_size * num_particle * 1
    r1 = tf.sqrt(tf.square(x[:,:, 0]) + tf.square(x[:,:, 1])) # batch_size * num_particle * 1
    r = tf.stack([r0, r1], axis=2)# batch_size * num_particle * 2 * 1
    return r


# CT motion model
def get_F(Fw, batch_size):
    F_out = None
    for i in range(batch_size):
        w = Fw[i]
        temp0 = [1, 0, tf.sin(w * sT) / w, (tf.cos(w * sT) - 1) / w]
        temp1 = [0, 1, -(tf.cos(w * sT) - 1) / w, tf.sin(w * sT) / w]
        temp2 = [0, 0, tf.cos(w * sT), -tf.sin(w * sT)]
        temp3 = [0, 0, tf.sin(w * sT), tf.cos(w * sT)]
        F_c = tf.stack([temp0, temp1, temp2, temp3], axis=0)
        if i == 0:
            F_out = F_c
        else:
            F_out = tf.concat([F_out, F_c], axis=0)
    F_out = tf.reshape(F_out, [batch_size, 4, 4])
    return F_out

def F_cv():
    temp0 = [1, 0, sT, 0]
    temp1 = [0, 1, 0, sT]
    temp2 = [0, 0, 1, 0]
    temp3 = [0, 0, 0, 1]
    F_mat = np.array([temp0,temp1,temp2,temp3])
    F_out = tf.stack(np.array([[F_mat for _ in range(num_particle)] for _ in range(batch_size)], 'float64'))
    F_out = tf.reshape(F_out, [batch_size,num_particle, 4, 4])
    return tf.cast(F_out, dtype=tf.float64)

# soft resampling
def resampling(num_particles, prob):
    """
    The implementation of soft-resampling. We implement soft-resampling in a batch-manner.
    :param particles: \{(h_t^i, c_t^i)\}_{i=1}^K for PF-LSTM and \{h_t^i\}_{i=1}^K for PF-GRU.
                    each tensor has a shape: [num_particles * batch_size, h_dim]
    :param prob: weights for particles in the log space. Each tensor has a shape: [num_particles * batch_size, 1]
    :return: resampled particles and weights according to soft-resampling scheme.
    """
    resamp_alpha = 0.5
    resamp_prob = resamp_alpha * tf.exp(prob) + (1 - resamp_alpha) * 1 / num_particles
    indices = tf.multinomial(resamp_prob, num_samples=num_particles)
    return indices

X_state_all = []
for step in range(1, timestep_size):
    print('################################')
    print('Step:', step)
    # # Stage 1. Particle sampling (state prediction) weight update (state update)
    F_ = F_cv()
    # pred_sample batch_size * num_particle * 4 * 1
    x_particle = tf.cast(x_particle,dtype=tf.float64)
    x_particle_pred = tf.matmul(F_, tf.reshape(x_particle, [batch_size,num_particle, 4, 1])) + tf.matmul(tf.sqrt(Q),np.random.randn(
                                                                                                  batch_size, num_particle,4, 1))
    meas_pred = my_hx(x_particle_pred) # batch_size * num_particle * 2 * 1

    z_meas_repeat = tf.reshape(tf.tile(z_meas[:, step, :],[1,num_particle]),[batch_size,num_particle,meas_dim,1])
    z_meas_repeat = tf.cast(z_meas_repeat,dtype=tf.float64)
    z_error = z_meas_repeat - meas_pred # batch_size * num_particle * 2 * 1
    term1 = 1 / tf.sqrt(tf.linalg.det(2 * np.pi * R)) # batchsize * num_particle
    term2 = tf.reshape(tf.exp(-0.5 * tf.matmul(tf.matmul(tf.transpose(z_error, [0,1,3,2]), tf.linalg.inv(R)), z_error)),
                       [batch_size,num_particle])
    weight = tf.reshape(term1 * term2, [batch_size,num_particle, 1])
    weight = tf.reshape(weight, [batch_size, num_particle, 1])

    weight_norm = tf.reshape(tf.reshape(weight,[batch_size, num_particle])/tf.reduce_sum(weight, axis=1),[batch_size,num_particle,1])

    # Stage 2. Weight normalization, resampling, and obtaining estimated states
    x_particle = []
    x_state = []
    # weight_sample = tf.reshape(weight_nor, [batch_size, num_particle])
    res_sam = 0.9
    weight_sample = tf.reshape(res_sam * weight_norm + (1-res_sam) * (1 / num_particle), [batch_size, num_particle,1])

    weight_sample = tf.reshape(tf.reshape(weight_sample,[batch_size, num_particle])/tf.reduce_sum(weight_sample, axis=1),[batch_size,num_particle])

    outIndex_ = tf.multinomial(tf.log(weight_sample), num_particle)

    weight_new = tf.reshape(weight_sample/(res_sam*weight_sample+(1-res_sam)*(1/num_particle)),[batch_size, num_particle,1])

    weight_new = tf.reshape(tf.reshape(weight_new,[batch_size, num_particle])/tf.reduce_sum(weight_new, axis=1),[batch_size,num_particle,1])

    for bt in range(batch_size):
        x_parti = tf.gather(x_particle_pred[bt], outIndex_[bt])
        x_particle.append(x_parti)
        x_state_bt_x = tf.matmul(tf.reshape(weight_new[bt],[1,num_particle]),x_parti[:, 0])[0,0] #tf.reduce_mean(x_parti[:, 0])
        x_state_bt_y = tf.matmul(tf.reshape(weight_new[bt],[1,num_particle]),x_parti[:, 1])[0,0] #tf.reduce_mean(x_parti[:, 1])
        x_state_bt_vx = tf.matmul(tf.reshape(weight_new[bt],[1,num_particle]),x_parti[:, 2])[0,0] #tf.reduce_mean(x_parti[:, 2])
        x_state_bt_vy = tf.matmul(tf.reshape(weight_new[bt],[1,num_particle]),x_parti[:, 3])[0,0] #tf.reduce_mean(x_parti[:, 3])
        # x_state_bt_x = tf.reduce_mean(x_parti[:, 0])
        # x_state_bt_y = tf.reduce_mean(x_parti[:, 1])
        # x_state_bt_vx = tf.reduce_mean(x_parti[:, 2])
        # x_state_bt_vy = tf.reduce_mean(x_parti[:, 3])
        x_state_bt = tf.stack([x_state_bt_x, x_state_bt_y,x_state_bt_vx, x_state_bt_vy])
        x_state.append(x_state_bt)

    x_particle = tf.stack(x_particle)
    x_state = tf.stack(x_state)
    X_state_all.append(x_state)
X_state_all = tf.transpose(tf.stack(X_state_all), [1, 0, 2])

sess = tf.Session()
X = sess.run(X_state_all)
plt.plot(Traj_r[0, :, 0], Traj_r[0, :, 1])
plt.plot(Obser[0, :, 0], Obser[0, :, 1])
plt.plot(X[0, :, 0], X[0, :, 1])
plt.show()

print(np.sqrt(np.mean(np.square(Traj_r[:,1:,:]-X[:,:,:]))))
print(np.sqrt(np.mean(np.square(Traj_r[:,1:,:2]-Obser[:,1:,:2]))))