import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

TRAJ = np.load('TRAJ_month_3_cha.npy',allow_pickle=True).item()
TRAJ_ = dict()
term = 0

for i in range(len(TRAJ)):
    if TRAJ[i].shape[0] >= 100:
        print(i)
        Exc = False
        TRAJ_plus_V = np.zeros([TRAJ[i].shape[0],7])
        TRAJ_plus_V[:,:4] = TRAJ[i][:,:4]
        for j in range(1,TRAJ[i].shape[0]):
            TRAJ_plus_V[j,4] = TRAJ[i][j,0] - TRAJ[i][j-1,0]
            TRAJ_plus_V[j,5] = TRAJ[i][j,1] - TRAJ[i][j-1,1]
            TRAJ_plus_V[j,6] = TRAJ[i][j,2] - TRAJ[i][j-1,2]
            # Trajectories with excessive speed are considered abnormal
            if np.sqrt(np.square(TRAJ_plus_V[j,4])+np.square(TRAJ_plus_V[j,5])+np.square(TRAJ_plus_V[j,6])) > 1500:
                Exc = True
            # if np.sqrt(np.square(TRAJ_plus_V[j,6])) > 50:
            #     Exc = True
            if TRAJ_plus_V[j,6] > 400:
                Exc = True
            # TRAJ_plus_V[j, 7] = angular_velocity
            if j > 2:
                # Calculate angular velocity
                delta_t = 1  # You need to provide an appropriate time interval
                v1 = TRAJ_plus_V[j - 1, 4:7]
                v2 = TRAJ_plus_V[j, 4:7]
                angle_cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(angle_cosine, -1.0, 1.0))
                # Turn rate
                angular_velocity = angle / delta_t
                if angular_velocity > 0.4:
                    Exc = True

            # Trajectories with excessive acceleration are also considered abnormal
            # if j >= 2:
            #     ax = (TRAJ[i][j,0] - TRAJ[i][j-1,0]) - (TRAJ[i][j-1,0] - TRAJ[i][j-2,0])
            #     ay = (TRAJ[i][j, 1] - TRAJ[i][j-1, 1]) - (TRAJ[i][j-1, 1] - TRAJ[i][j - 2, 1])
            #     az = (TRAJ[i][j, 2] - TRAJ[i][j-1, 2]) - (TRAJ[i][j-1, 2] - TRAJ[i][j - 2, 2])
            #     print(ax)
            #     print(ay)
            #     print(az)
            #     # if np.sqrt(np.square(ax) + np.square(ay) + np.square(az)) > 1500:
            #     #     Exc = True
        TRAJ_plus_V[0, 4] = TRAJ_plus_V[1, 4]
        TRAJ_plus_V[0, 5] = TRAJ_plus_V[1, 5]
        TRAJ_plus_V[0, 6] = TRAJ_plus_V[1, 6]
        # Exclude trajectories with very low altitude
        if np.any(TRAJ_plus_V[:, 2] < 0):
            Exc = True
        if Exc:
            print('Abnormal trajectory!')
            continue
        TRAJ_[term] = TRAJ_plus_V
        term+=1
print('Saving')
print(len(TRAJ_))
np.save('Traj_selected_3.npy',TRAJ_)
print('Save successful')
        # ax = plt.gca(projection='3d')
        # ax.set_title(i)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        # ax.set_zlabel("z")
        # ax.plot(TRAJ_plus_V[:, 0], TRAJ_plus_V[:, 1], TRAJ_plus_V[:, 2])
        # ax.scatter(TRAJ_plus_V[0, 0], TRAJ_plus_V[0, 1], TRAJ_plus_V[0, 2], c='b')
        # # plt.plot(traj[:, 2], traj[:, 3])
        # plt.show()
