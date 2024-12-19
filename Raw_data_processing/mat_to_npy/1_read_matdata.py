import numpy as np
import scipy.io as io

# Mat_data = []
# TRAJ1 = dict()
# term = 0
#
# for date in range(1,19):
#     try:
#         if date <10:
#             mat_date = io.loadmat('matdata/2006010%d.mat'%date)['Total'][0]
#         else:
#             mat_date = io.loadmat('matdata/200601%d.mat' % date)['Total'][0]
#         Mat_data.append(mat_date)
#         for i in range(mat_date.shape[0]):
#             try:
#                 # Screening of civil airliners
#                 if np.array(['J']) in mat_date[i][1][0] or np.array(['R']) in mat_date[i][1][0]:
#                     # mate_now = mat_date[i]
#                     mata = list(mat_date[i])
#                     mata.pop(0)
#                     mata.pop(0)
#                     mata = mata[0]
#                     traj = []
#                     for j in range(mata.shape[0]):
#                         x = mata[j, 0][0, 0]
#                         y = mata[j, 1][0, 0]
#                         z = mata[j, 2][0, 0]
#                         v = mata[j, 3][0, 0]
#                         t = mata[j, 4][0, 0]
#                         state = np.array([x, y, z, v, t])
#                         traj.append(state)
#                     TRAJ1[term] = np.array(traj)
#                     term+=1
#             except:
#                 continue
#         print('data in Jan，%d over'%date)
#     except:
#         continue
#
# print('traj number in Jan',len(TRAJ1))
# np.save('TRAJ_month_1_ori.npy', TRAJ1)
#
# Mat_data = []
# TRAJ2 = dict()
# term = 0
# for date in range(10,27):
#     try:
#         if date <10:
#             mat_date = io.loadmat('mat数据/2006020%d.mat'%date)['Total'][0]
#         else:
#             mat_date = io.loadmat('mat数据/200602%d.mat' % date)['Total'][0]
#         Mat_data.append(mat_date)
#         for i in range(mat_date.shape[0]):
#             try:
#                 # Screening of civil airliners
#                 if np.array(['J']) in mat_date[i][1][0] or np.array(['R']) in mat_date[i][1][0]:
#                     # mate_now = mat_date[i][0][1]
#                     # print(mate_now)
#                     mata = list(mat_date[i])
#                     mata.pop(0)
#                     mata.pop(0)
#                     mata = mata[0]
#                     traj = []
#                     for j in range(mata.shape[0]):
#                         x = mata[j, 0][0, 0]
#                         y = mata[j, 1][0, 0]
#                         z = mata[j, 2][0, 0]
#                         v = mata[j, 3][0, 0]
#                         t = mata[j, 4][0, 0]
#                         state = np.array([x, y, z, v, t])
#                         traj.append(state)
#                     TRAJ2[term] = np.array(traj)
#                     term+=1
#             except:
#                 continue
#         print('February data，%d over!'%date)
#     except:
#         continue
# print('Number of trajectories inFebruary',len(TRAJ2))
# np.save('TRAJ_month_2_ori.npy', TRAJ2)

Mat_data = []
TRAJ3 = dict()
term = 0
for date in range(1,2):
    try:
        if date <10:
            mat_date = io.loadmat('matdata/2006010%d.mat'%date)['Total'][0]
        else:
            mat_date = io.loadmat('matdata/200601%d.mat' % date)['Total'][0]
        Mat_data.append(mat_date)
        for i in range(mat_date.shape[0]):
            try:
                # Screening of civil airliners
                if np.array(['J']) in mat_date[i][1][0] or np.array(['R']) in mat_date[i][1][0]:
                    # mate_now = mat_date[i][0][1]
                    # print(mate_now)
                    mata = list(mat_date[i])
                    mata.pop(0)
                    mata.pop(0)
                    mata = mata[0]
                    traj = []
                    for j in range(mata.shape[0]):
                        x = mata[j, 0][0, 0]
                        y = mata[j, 1][0, 0]
                        z = mata[j, 2][0, 0]
                        v = mata[j, 3][0, 0]
                        t = mata[j, 4][0, 0]
                        state = np.array([x, y, z, v, t])
                        traj.append(state)
                    TRAJ3[term] = np.array(traj)
                    term+=1
            except:
                continue
        print('%d over!'%date)
    except:
        continue

print('Number of trajectories in week 1',len(TRAJ3))
np.save('TRAJ_month_3_week_1_ori.npy', TRAJ3)




