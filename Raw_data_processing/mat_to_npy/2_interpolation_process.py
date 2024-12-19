from mpl_toolkits.mplot3d import axes3d
import numpy as np
from scipy import interpolate
import pylab as pl
import matplotlib.pyplot as plt

TRAJ = np.load('TRAJ_month_3_week_1_ori.npy',allow_pickle=True).item()
print(len(TRAJ))
TRAJ_ = dict()
traj_num = len(TRAJ)

index_lst = [i for i in range(traj_num)]

# x y z speed time
con = False
for i in range(traj_num):
    print(i)
    traj = TRAJ[i]
    # print(traj[:,-1])
    start_time = traj[0,-1]
    end_time = traj[-1,-1]
    x = traj[:, -1]

    # Remove stationary trajectories
    for q in range(1,traj.shape[0]):
        if x[q] == x[q-1]:
            con = True
    if con:
        con = False
        continue

    xnew = np.array([i*4 for i in range(start_time, int((end_time-4)/4))])
    y_1 = traj[:, 0]
    y_2 = traj[:, 1]
    y_3 = traj[:, 2]
    y_4 = traj[:, 3]
    # pl.plot(y_1, y_2, "ro")
    # "nearest", "zero", "slinear", "quadratic",
    for kind in ["cubic"]:  # Interpolation method
        # "nearest","zero" are stepwise interpolation
        # slinear is linear interpolation
        # "quadratic","cubic" are 2nd and 3rd degree B-spline interpolation
        f1 = interpolate.interp1d(x, y_1, kind=kind)
        y_1_new = f1(xnew)
    for kind in ["cubic"]:  # Interpolation method
        # "nearest","zero" are stepwise interpolation
        # slinear is linear interpolation
        # "quadratic","cubic" are 2nd and 3rd degree B-spline interpolation
        f2 = interpolate.interp1d(x, y_2, kind=kind)
        y_2_new = f2(xnew)
    for kind in ["cubic"]:  # Interpolation method
        # "nearest","zero" are stepwise interpolation
        # slinear is linear interpolation
        # "quadratic","cubic" are 2nd and 3rd degree B-spline interpolation
        f3 = interpolate.interp1d(x, y_3, kind=kind)
        y_3_new = f3(xnew)
    # Interpolate the speed as well
    for kind in ["cubic"]:  # Interpolation method
        # "nearest","zero" are stepwise interpolation
        # slinear is linear interpolation
        # "quadratic","cubic" are 2nd and 3rd degree B-spline interpolation
        f4 = interpolate.interp1d(x, y_4, kind=kind)
        y_4_new = f4(xnew)
    lst = []
    for t in range(y_1_new.shape[0]):
        lst.append(np.array([y_1_new[t],y_2_new[t],y_3_new[t],y_4_new[t]]))
    traj_out = np.array(lst)
    TRAJ_[index_lst[0]] = traj_out
    index_lst.pop(0)

print(len(TRAJ_))
# x y z v
np.save('TRAJ_month_3_cha.npy',TRAJ_)
