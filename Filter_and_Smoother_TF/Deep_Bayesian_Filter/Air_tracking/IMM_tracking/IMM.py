# -*- coding: utf-8 -*-
# pylint: disable=invalid-name, too-many-instance-attributes
"""
Created on Mon Aug  6 07:53:34 2018

@author: rlabbe
"""

from __future__ import (absolute_import, division)
import numpy as np
from numpy import dot, asarray, zeros, outer
from filterpy.common import pretty_str
from copy import deepcopy,copy

dim_state = 4

class IMMEstimator(object):
    """ Implements an Interacting Multiple-Model (IMM) estimator.

    Parameters
    ----------

    filters : (N,) array_like of KalmanFilter objects
        List of N filters. filters[i] is the ith Kalman filter in the
        IMM estimator.

        Each filter must have the same dimension for the state `x` and `P`,
        otherwise the states of each filter cannot be mixed with each other.

    mu : (N,) array_like of float
        mode probability: mu[i] is the probability that
        filter i is the correct one.

    M : (N, N) ndarray of float
        Markov chain transition matrix. M[i,j] is the probability of
        switching from filter j to filter i.


    Attributes
    ----------
    x : numpy.array(dim_x, 1)
        Current state estimate. Any call to update() or predict() updates
        this variable.

    P : numpy.array(dim_x, dim_x)
        Current state covariance matrix. Any call to update() or predict()
        updates this variable.

    x_prior : numpy.array(dim_x, 1)
        Prior (predicted) state estimate. The *_prior and *_post attributes
        are for convienence; they store the  prior and posterior of the
        current epoch. Read Only.

    P_prior : numpy.array(dim_x, dim_x)
        Prior (predicted) state covariance matrix. Read Only.

    x_post : numpy.array(dim_x, 1)
        Posterior (updated) state estimate. Read Only.

    P_post : numpy.array(dim_x, dim_x)
        Posterior (updated) state covariance matrix. Read Only.

    N : int
        number of filters in the filter bank

    mu : (N,) ndarray of float
        mode probability: mu[i] is the probability that
        filter i is the correct one.

    M : (N, N) ndarray of float
        Markov chain transition matrix. M[i,j] is the probability of
        switching from filter j to filter i.

    cbar : (N,) ndarray of float
        Total probability, after interaction, that the target is in state j.
        We use it as the # normalization constant.

    likelihood: (N,) ndarray of float
        Likelihood of each individual filter's last measurement.

    omega : (N, N) ndarray of float
        Mixing probabilitity - omega[i, j] is the probabilility of mixing
        the state of filter i into filter j. Perhaps more understandably,
        it weights the states of each filter by:
            x_j = sum(omega[i,j] * x_i)

        with a similar weighting for P_j


    Examples
    --------

    >>> import numpy as np
    >>> from filterpy.common import kinematic_kf
    >>> from filterpy.kalman import IMMEstimator
    >>> kf1 = kinematic_kf(2, 2)
    >>> kf2 = kinematic_kf(2, 2)
    >>> # do some settings of x, R, P etc. here, I'll just use the defaults
    >>> kf2.Q *= 0   # no prediction error in second filter
    >>>
    >>> filters = [kf1, kf2]
    >>> mu = [0.5, 0.5]  # each filter is equally likely at the start
    >>> trans = np.array([[0.97, 0.03], [0.03, 0.97]])
    >>> imm = IMMEstimator(filters, mu, trans)
    >>>
    >>> for i in range(100):
    >>>     # make some noisy data
    >>>     x = i + np.random.randn()*np.sqrt(kf1.R[0, 0])
    >>>     y = i + np.random.randn()*np.sqrt(kf1.R[1, 1])
    >>>     z = np.array([[x], [y]])
    >>>
    >>>     # perform predict/update cycle
    >>>     imm.predict()
    >>>     imm.update(z)
    >>>     print(imm.x.T)

    For a full explanation and more examples see my book
    Kalman and Bayesian Filters in Python
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python


    References
    ----------

    Bar-Shalom, Y., Li, X-R., and Kirubarajan, T. "Estimation with
    Application to Tracking and Navigation". Wiley-Interscience, 2001.

    Crassidis, J and Junkins, J. "Optimal Estimation of
    Dynamic Systems". CRC Press, second edition. 2012.

    Labbe, R. "Kalman and Bayesian Filters in Python".
    https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
    """

    def __init__(self, filters, mu, M):

        self.filters = filters
        self.mu = asarray(mu) / np.sum(mu)
        self.mu_last = np.zeros_like(self.mu)
        self.last_filt_x_1 = zeros(filters[0].x.shape)
        self.last_filt_x_2 = zeros(filters[0].x.shape)
        self.last_filt_x_3 = zeros(filters[0].x.shape)
        self.last_filt_x = np.array([self.last_filt_x_1,self.last_filt_x_2,self.last_filt_x_3])
        self.last_filt_P_1 = zeros(filters[0].P.shape)
        self.last_filt_P_2 = zeros(filters[0].P.shape)
        self.last_filt_P_3 = zeros(filters[0].P.shape)
        self.last_filt_P = np.array([self.last_filt_P_1,self.last_filt_P_2,self.last_filt_P_3])
        self.M = M
        # x_shape = filters[0].x.shape
        self.x = zeros(filters[0].x.shape)
        self.x_1 = zeros(filters[0].x.shape)
        self.x_2 = zeros(filters[0].x.shape)
        self.x_3 = zeros(filters[0].x.shape)
        self.filt_x = np.array([self.x_1,self.x_2,self.x_3])
        self.x_last = zeros(filters[0].x.shape)
        self.P = zeros(filters[0].P.shape)
        self.P_last = np.zeros_like(self.P)
        self.N = len(filters)  # number of filters
        self.likelihood = zeros(self.N)
        self.omega = zeros((self.N, self.N))
        self._compute_mixing_probabilities()
        # initialize imm state estimate based on current filters
        self._compute_state_estimate()
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        self.xs = []
        self.Ps = []
        self.c = 1
    # def like(self,z,residual=np.subtract):
    #     for i, f in enumerate(self.filters):
    #         y1 = residual(z, f.Hx)
    #         like1 = np.exp(logpdf(x=self.y, cov=self.S))
    #         y2 = residual(z, f.zp)
    def filter_update(self,z):
        '''状态预测后进行更新'''
        # run update on each filter, and save the likelihood
        # 对每个滤波器运行更新，并保存可能性
        for i, f in enumerate(self.filters):
            f.x = f.x.reshape([4])
            f.update(z)
            f.x = f.x.reshape([4])

            # print('量测值',z)
            # print('滤波器更新',f.x)
            # 不是self.y，这个y应该是交互之后的
            # print('$$$$$$$$$$$$$$$$$')
            # print('量测残差y',f.y)
            # print('量测残差协方差S',f.S)
            like = 1/np.sqrt(np.linalg.det(2 * np.pi * f.S)) * np.exp(-0.5 * np.dot(np.dot(f.y.T,np.linalg.inv(f.S)),f.y))
            # print('自己算的likelihood',like)
            # print('程序里的likelihood',f.likelihood)
            # print('$$$$$$$$$$$$$$$$$')
            '''更新完成后得到了似然'''
            self.likelihood[i] = like

            self.filt_x[i] = deepcopy(f.x.reshape([4,1]))

        # print(self.likelihood)
        # print(self.mu)
    def model_probability_update(self):
        self._compute_mixing_probabilities()
        self.mu_last = deepcopy(self.mu)
        self.mu = self.cbar * self.likelihood
        # print(self.mu)
        self.mu /= np.sum(self.mu)  # normalize

    def update(self):
        '''
        这时候可以进行优化了
        '''
        # update mode probabilities from total probability * likelihood
        # 从总概率*可能性中更新模式概率
        # 模型概率更新
        # print('**********************')
        # print(self.likelihood)
        # print(self.cbar)
        # self.mu_last = deepcopy(self.mu)
        # self.mu = self.cbar * self.likelihood
        # # print(self.mu)
        # self.mu /= np.sum(self.mu)  # normalize
        # print(self.mu)
        # print('**********************')
        # 计算每个滤波器的混合概率
        # 计算状态交互概率omega，以在下一步预测中用其进行状态交互
        # self._compute_mixing_probabilities()
        # self._model_probability_update()
        # compute mixed IMM state and covariance and save posterior estimate
        # 计算混合IMM状态和协方差并保存后验估计
        # 模型输出

        self._compute_state_estimate()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        # print('IMM最终更新', self.x)

        for i, f in enumerate(self.filters):

            self.last_filt_x[i] = deepcopy(f.x)
            self.last_filt_P[i] = deepcopy(f.P)
        # print(1)

    def predict(self, u=None):
        self.x_last = self.x
        self.P_last = self.P

        """
        Predict next state (prior) using the IMM state propagation
        equations.
        Parameters
        ----------
        u : np.array, optional
            Control vector. If not `None`, it is multiplied by B
            to create the control input into the system.
        """
        '''预测步先交互'''
        self._compute_mixing_probabilities()
        # compute mixed initial conditions
        # 计算混合初始条件
        self.xs, self.Ps = [], []
        # w 就是 omega
        # 对每个模型进行状态交互
        # print('self.omega',self.omega)
        for i, (f, w) in enumerate(zip(self.filters, self.omega.T)):

            x = zeros(self.x.shape)
            # print('交互之前',f.x)
            for kf, wj in zip(self.filters, w):
                # 状态交互
                x += kf.x * wj
            # 保存状态交互结果
            # print('状态交互',x)
            self.xs.append(x)
            P = zeros(self.P.shape)
            for kf, wj in zip(self.filters, w):
                y = kf.x - x
                # 协方差混合
                P += wj * (outer(y, y) + kf.P)
            # 保存协方差混合结果
            self.Ps.append(P)

        '''得到交互结果后，进行状态预测'''
        # compute each filter's prior using the mixed initial conditions
        # 使用交互初始条件计算每个滤波器的先验
        # 对每个模型用对应的交互初始条件计算预测值
        for i, f in enumerate(self.filters):
            # propagate using the mixed state estimate and covariance
            # 利用混合状态估计和协方差进行传播
            # 将交互后的状态和协方差赋给滤波器
            f.x = self.xs[i].copy()
            f.P = self.Ps[i].copy()
            # 滤波器执行预测
            f.x = f.x.reshape([4])
            f.predict()
            # print('滤波器预测',f.x)
            # '''状态预测后，可以计算似然'''
            # self.likelihood[i] = f.likelihood

        # compute mixed IMM state and covariance and save posterior estimate
        # 计算混合IMM状态和协方差并保存后验估计
        # self._compute_state_estimate()
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
    def mostlike_model(self):
        td = 0.8
        index = None
        mu_arr = np.array(self.mu)
        if (mu_arr>td).any():
            index = np.argmax(mu_arr)
        else:
            index = np.argmax(self.likelihood)
        return index


    def _compute_state_estimate(self):
        #
        """
        Computes the IMM's mixed state estimate from each filter using
        the the mode probability self.mu to weight the estimates.
        使用模式概率 self.mu 对估计进行加权，计算来自每个滤波器的 IMM 混合状态估计。
        """
        self.x.fill(0)
        for f, mu in zip(self.filters, self.mu):
            f.x = f.x.reshape([4,1])

            self.x += f.x * mu

        self.P.fill(0)
        for f, mu in zip(self.filters, self.mu):
            f.x = f.x
            y = f.x - self.x
            self.P += mu * (outer(y, y) + f.P)

    def _compute_mixing_probabilities(self):
        """
        Compute the mixing probability for each filter.
        计算状态混合的概率
        """
        self.cbar = dot(self.mu, self.M)
        for i in range(self.N):
            for j in range(self.N):
                # omega_ij = Pi_ij * miu_i / C_bar_j
                self.omega[i, j] = (self.M[i, j]*self.mu[i]) / self.cbar[j]

    # def _model_probability_update(self):
    #     """
    #     模型概率更新
    #     """
    #     self.c = 0
    #     for i in range(3):
    #         self.c += self.likelihood[i]*self.cbar[i]
    #     for i in range(3):
    #         self.mu[i] = self.likelihood[i]*self.cbar[i]/self.c

    def __repr__(self):
        return '\n'.join([
            'IMMEstimator object',
            pretty_str('x', self.x),
            pretty_str('P', self.P),
            pretty_str('x_prior', self.x_prior),
            pretty_str('P_prior', self.P_prior),
            pretty_str('x_post', self.x_post),
            pretty_str('P_post', self.P_post),
            pretty_str('N', self.N),
            pretty_str('mu', self.mu),
            pretty_str('M', self.M),
            pretty_str('cbar', self.cbar),
            pretty_str('likelihood', self.likelihood),
            pretty_str('omega', self.omega)
            ])
