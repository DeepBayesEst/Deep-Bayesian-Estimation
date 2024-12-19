import numpy as np

def kf(xkk,Pkk,F,H,z,Q,R):
    #### 时间更新（预测）
    xk1k = np.matmul(F,xkk)
    Pk1k = np.matmul(np.matmul(F,Pkk),F.T) + Q

    #### 量测更新
    Pzzk1k = np.matmul(np.matmul(H,Pk1k),H.T)+R

    Pxzk1k = np.matmul(Pk1k,H.T)

    Kk = np.matmul(Pxzk1k,np.linalg.inv(Pzzk1k))

    xkk = xk1k + np.matmul(Kk,z - np.matmul(H,xk1k))

    Pkk = Pk1k - np.matmul(np.matmul(Kk,H),Pk1k)

    return xkk,Pkk,Pk1k

def arvbkf(xkk,Pkk,ukk,Ukk,F,H,z,Q,R,N,rou):
    #### 准备
    nz = z.shape[0]
    nx = xkk.shape[0]

    #### 时间更新 （KF预测）
    xk1k = np.matmul(F,xkk)
    Pk1k = np.matmul(np.matmul(F,Pkk),F.T) + Q
    uk1k = rou * (ukk - nz - 1) + nz + 1
    Uk1k = rou * Ukk

    #### 量测更新
    xkk = xk1k # 初始化x
    Pkk = Pk1k # 初始化P

    for i in range(N):
        #### 更新Rk的分布
        Bk = np.matmul(z - np.matmul(H,xkk),np.transpose(z-np.matmul(H,xkk)))
        ukk = uk1k + 1
        Ukk = Uk1k + Bk
        E_i_Rk = (ukk - nz - 1) * np.linalg.inv(Ukk)

        #### 计算状态估计
        D_R = np.linalg.inv(E_i_Rk)
        zk1k = np.matmul(H,xk1k)
        Pzzk1k = np.matmul(np.matmul(H,Pk1k),H.T) + D_R
        Pxzk1k = np.matmul(Pk1k,H.T)
        Kk = np.matmul(Pxzk1k,np.linalg.inv(Pzzk1k))
        xkk = xk1k + np.matmul(Kk,z - zk1k)
        Pkk = Pk1k - np.matmul(np.matmul(Kk,H),Pk1k)

    return xkk,Pkk,ukk,Ukk,Pk1k,D_R


def aprivbkf(xkk,Pkk,ukk,Ukk,F,H,z,Q,R,N,tao1,rou):
    #### 准备
    nz = z.shape[0]
    nx = xkk.shape[0]

    #### 时间更新 (KF预测)
    xk1k = np.matmul(F,xkk) # 可以非线性
    Pk1k = np.matmul(np.matmul(F,Pkk),F.T) + Q

    #### 量测更新
    # 参数初始化
    tk1k = nx + 1 + tao1
    Tk1k = tao1 * Pk1k
    uk1k = rou * (ukk - nz - 1) + nz + 1
    Uk1k = rou * Ukk
    xkk = xk1k
    Pkk = Pk1k

    # 开始变分迭代N次
    for i in range(N):
        # Step 1 : 更新Pk1k的分布
        Ak = np.matmul(xkk - xk1k,np.transpose(xkk - xk1k)) + Pkk
        tkk = tk1k + 1
        Tkk = Tk1k + Ak
        E_i_Pk1k = (tkk - nx - 1) * np.linalg.inv(Tkk)

        # Step 2 : 更新Rk的分布
        Bk = np.matmul(z - np.matmul(H,xkk),np.transpose(z - np.matmul(H,xkk))) + np.matmul(np.matmul(H,Pkk),H.T)
        ukk = uk1k + 1
        Ukk = Uk1k + Bk
        E_i_Rk = (ukk - nz - 1) * np.linalg.inv(Ukk)

        # Step 3 : 计算状态估计
        D_Pk1k = np.linalg.inv(E_i_Pk1k)
        D_R = np.linalg.inv(E_i_Rk)

        # 下面可以非线性求解
        zk1k = np.matmul(H,xk1k)
        Pzzk1k = np.matmul(np.matmul(H,D_Pk1k),H.T)+D_R # S,新息协方差求解
        Pxzk1k = np.matmul(D_Pk1k,H.T)
        Kk = np.matmul(Pxzk1k,np.linalg.inv(Pzzk1k))
        xkk = xk1k + np.matmul(Kk,z - zk1k)
        Pkk = D_Pk1k - np.matmul(np.matmul(Kk,H),D_Pk1k)

    return xkk,Pkk,ukk,Ukk,D_Pk1k,D_R