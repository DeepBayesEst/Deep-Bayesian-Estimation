import numpy as np

def F_lor_part(xyz,dt):
    # 设置默认参数可以不使用odeint的args参数
    x, y, z = xyz[0],xyz[1],xyz[2]
    dx_dt = -10*x+10*y
    dy_dt = 28*x-y-x*z
    dz_dt = x*y - 8/3*z
    x_ = x+dx_dt*dt
    y_ = y+dy_dt*dt
    z_ = z+dz_dt*dt
    # p, r, b = 10,28,8/3
    # dx = -p*(x-y)
    # dy = r*x-y-x*z
    # dz = -b*z+x*y
    return np.array([x_,y_,z_])

xyz = np.array([0.1,0.2,0.3])
dt = 0.02
Res = []
for i in range(2000):
    xyz = F_lor_part(xyz, dt)
    Res.append(xyz)
Res = np.array(Res)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


plt.figure()
plt.axes(projection='3d')
plt.plot(Res[:, 0], Res[:, 1], Res[:, 2])
plt.show()