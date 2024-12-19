import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
num_points = 100
num_points_high = 60
dim = 1  # 状态X的维度
k = 5
k1 = k
k2 = 3
noise_std_q = 2.
noise_std_r = 6. # 噪声的标准差
noise_std_q_high = 0.5
noise_std_r_high = 1.  # 噪声的标准差
F_0_economic_stable = 0.5  # np.random.normal()

def sine_wave(num_points):
    """生成正弦波形的时间序列"""
    return np.cos(np.linspace(0, 4*2 * np.pi, num=num_points))
def exponential_decay(num_points):
    """生成随时间指数衰减的序列"""
    return np.exp(-np.linspace(0, 5, num=num_points))
bar_F_economic_1d = sine_wave(num_points)

bar_F_economic_smoothed = np.load('bar_F_economic_smoothed.npy')

def add_gaussian_noise(sequences, noise_std):
    noisy_sequences = sequences+np.random.normal(0, noise_std, sequences.shape)
    return noisy_sequences

def get_data_single():
    epsilon_moderate = np.random.normal(0, noise_std_q, (num_points))  # 使用适中的标准差
    # 重新生成时间序列，考虑平滑的状态转移矩阵
    X_economic = np.zeros(num_points)
    X_economic_markov = np.zeros(num_points)
    X_economic[0] = np.random.rand() * 0.1  # 经济指标的初始值
    X_economic_markov[0] = deepcopy(X_economic[0])
    for t in range(1, num_points):
        X_t_economic = F_0_economic_stable*X_economic[t-1]
        X_t_economic_mk = F_0_economic_stable*X_economic_markov[t-1]
        X_t_economic = X_t_economic
        X_t_economic_mk = X_t_economic_mk
        for i in range(0, t, k):
            F_i_economic = bar_F_economic_1d[i] if i < num_points-1 else 0#F_0_economic_stable + (
            X_t_economic += F_i_economic*X_economic[i]
        X_economic[t] = X_t_economic + epsilon_moderate[t]
        X_economic_markov[t] = X_t_economic_mk + epsilon_moderate[t]
    # 生成带噪声的量测
    sequence = X_economic  # 选取第一条训练序列
    # 为训练集和测试集添加高斯噪声
    train_sequences_noisy = add_gaussian_noise(sequence, noise_std_r)
    return sequence,train_sequences_noisy

def get_data_batch(batch_size):
    seq_bt = []
    noisy_bt = []
    no_pn = []
    for i in range(batch_size):
        # epsilon_moderate = np.random.normal(0, noise_std_q, (num_points))  # 使用适中的标准差
        # 重新生成时间序列，考虑平滑的状态转移矩阵
        X_economic = np.zeros(num_points)
        X_economic_markov = np.zeros(num_points)
        X_economic[0] = (np.random.rand())*20-10  # 经济指标的初始值
        X_economic_markov[0] = deepcopy(X_economic[0])
        for t in range(1, num_points):
            X_t_economic = F_0_economic_stable*X_economic[t-1]
            X_t_economic_mk = F_0_economic_stable*X_economic_markov[t-1]
            period = 25
            for i in range(t):
                # 使用缩小幅度的周期性函数作为权重
                weight = 0.2#0.1 * (np.sin(2 * np.pi * i / period) + 1)  # 保证权重在0.05到0.1之间
                F_i_economic = bar_F_economic_1d[i] if i < num_points - 1 else 0
                X_t_economic += weight * F_i_economic * X_economic[i]
                X_t_economic_mk += weight * F_i_economic * X_economic_markov[i]
            # for i in range(0, t, k1):
            #     F_i_economic = bar_F_economic_1d[i] if i < num_points-1 else 0#F_0_economic_stable + (
            #     X_t_economic += F_i_economic*X_economic[i]
            #     F_i_economic_mk = bar_F_economic_1d[i] if i < num_points-1 else 0#F_0_economic_stable + (
            #     X_t_economic_mk += F_i_economic_mk*X_economic_markov[i]
            # for i in range(0, t, k2):
            #     F_i_economic = bar_F_economic_1d[i] if i < num_points-1 else 0#F_0_economic_stable + (
            #     X_t_economic -= F_i_economic*X_economic[i]
            X_economic[t] = X_t_economic + np.random.normal(0, noise_std_q)#epsilon_moderate[t]
            X_economic_markov[t] = X_t_economic_mk
        # 生成带噪声的量测
        sequence = X_economic  # 选取第一条训练序列
        # 为训练集和测试集添加高斯噪声
        train_sequences_noisy = add_gaussian_noise(sequence, noise_std_r)
        seq_bt.append(sequence)
        noisy_bt.append(train_sequences_noisy)
        no_pn.append(X_economic_markov)
    return np.array(seq_bt),np.array(noisy_bt),np.array(no_pn)

# 生成带噪声的量测
def add_gaussian_noise_high(sequences, noise_std):
    noisy_sequences = sequences + np.random.normal(0, noise_std, sequences.shape)
    return noisy_sequences
def get_data_high_dim(batch_size):
    num_points = num_points_high
    dim = 4  # 状态X的维度
    seq_bt = []
    noisy_bt = []
    for i in range(batch_size):
        epsilon_moderate = np.random.normal(0, noise_std_q_high, (dim, num_points))  # 使用适中的标准差
        # 重新生成时间序列，考虑平滑的状态转移矩阵
        X_economic = np.zeros((dim, num_points))
        X_economic[:, 0] = np.random.rand(dim) # 经济指标的初始值
        F_0_economic_stable = np.array([[0.5,-0.1,0.2,-0.1],[-0.3,0.2,-0.4,0.2],[0.15,-0.25,0.65,-0.25],[-0.25,0.15,-0.25,0.15]])#np.load('F_0_economic_stable.npy')

        for t in range(1, num_points):
            X_t_economic = F_0_economic_stable @ X_economic[:, t - 1].reshape([4, 1])
            X_t_economic = X_t_economic.reshape([4])
            for i in range(0, t, k):
                F_i_economic = bar_F_economic_smoothed[:, :, i] if i < num_points - 1 else 0  # F_0_economic_stable + (
                X_t_economic += np.dot(F_i_economic, X_economic[:, i])
            X_economic[:, t] = X_t_economic + epsilon_moderate[:, t]
        sequence = X_economic[:, :]  # 选取第一条训练序列
        # 为训练集和测试集添加高斯噪声
        train_sequences_noisy = add_gaussian_noise_high(sequence, noise_std_r_high)
        seq_bt.append(sequence)
        noisy_bt.append(train_sequences_noisy)
    return np.transpose(np.array(seq_bt),[0,2,1]),np.transpose(np.array(noisy_bt),[0,2,1])

# seq,nois,no_pn = get_data_batch(10)
# print(seq.shape)
# print(nois.shape)
# seq = seq[0]
# nois = nois[0]
# no_pn = no_pn[0]
# plt.plot(seq[:, 0], label=f'Noisy Train Sequence {1}')
# plt.plot(nois[:, 0], label=f'Noisy Train Sequence {1}')
# plt.figure(figsize=(10,5))
# plt.plot([], label=f'Ground Truth Sequence', c='b', marker='*')
# plt.plot([], label=f'Noisy Obser Sequence', c='r', marker='^')
# # for i in range(5):
# #     plt.plot(seq[i],c='b',marker='*')
# #     plt.plot(nois[i],c='r',marker='^')
# plt.plot(seq[0],c='b',marker='*')
# plt.plot(nois[0],c='r',marker='^')
# # plt.plot(no_pn[:], label=f'Noisy Train Sequence {1}')
# plt.title('Non-Markov Time Sequences')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.legend()
# plt.show()