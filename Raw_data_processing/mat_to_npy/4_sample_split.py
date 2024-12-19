import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# x y z v vx vy vz

TRAJ = np.load('Traj_selected_3.npy',allow_pickle=True).item()

TRAJ_ = dict()
term_train = 0
term_test = 0
Train = dict()
Val = dict()
print(len(TRAJ))

# The original trajectory interval is 4 seconds, downsampled to 20 seconds
for i in range(len(TRAJ)):
    Traj_now = []
    for j in range(TRAJ[i].shape[0]):
        if j % 5 == 0:
            Traj_now.append(TRAJ[i][j])

    # TRAJ_[term] = np.array(Traj_now)
    if i < 1600:
        Train[term_train] = np.array(Traj_now)
        term_train += 1
    elif i < 2000:
        Val[term_test] = np.array(Traj_now)
        term_test += 1

print(Val)

# Recalculate the interval velocity for the downsampled trajectories
# The velocity of the first point is inaccurate, remove it
TRAJ_tem = dict()

a = []
Train_res = []
# Sliding window for selecting points
slide_window = 24
# No overlap considered
for i in range(len(Train)):
    print(i)
    data_cha = Train[i]
    if data_cha.shape[0] < slide_window:
        continue
    elif data_cha.shape[0] == slide_window:
        Train_res.append(data_cha)
    else:  # Trajectory length is greater than 48
        for k in range(int(data_cha.shape[0] / slide_window)):
            Train_res.append(data_cha[k * slide_window:(k + 1) * slide_window])
        Train_res.append(data_cha[-slide_window:])

Val_res = []
for i in range(len(Val)):
    print(i)
    data_cha = Val[i]
    if data_cha.shape[0] < slide_window:
        continue
    elif data_cha.shape[0] == slide_window:
        Val_res.append(data_cha)
    else:  # Trajectory length is greater than 48
        for k in range(int(data_cha.shape[0] / slide_window)):
            Val_res.append(data_cha[k * slide_window:(k + 1) * slide_window])
        Val_res.append(data_cha[-slide_window:])

# Consider overlap region

# for i in range(len(Train)):
#     print(i)
#     data_cha = Train[i]
#     if data_cha.shape[0]<slide_window:continue
#     elif data_cha.shape[0] == slide_window:
#         Train_res.append(data_cha)
#     else: # Trajectory length is greater than 48
#         for k in range(int(data_cha.shape[0]/slide_window)):
#             Train_res.append(data_cha[k*slide_window:(k+1)*slide_window])
#         Train_res.append(data_cha[-slide_window:])
# Val_res = []
# for i in range(len(Val)):
#     print(i)
#     data_cha = Val[i]
#     if data_cha.shape[0]<slide_window:continue
#     elif data_cha.shape[0] == slide_window:
#         Val_res.append(data_cha)
#     else: # Trajectory length is greater than 48
#         for k in range(int(data_cha.shape[0]/slide_window)):
#             Val_res.append(data_cha[k*slide_window:(k+1)*slide_window])
#    
