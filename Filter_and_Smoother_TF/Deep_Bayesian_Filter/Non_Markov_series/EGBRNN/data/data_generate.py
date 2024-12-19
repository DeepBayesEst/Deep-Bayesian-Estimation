import numpy as np

from get_data import get_data_batch

all_gt,all_noisy,_ = get_data_batch(640+64)
np.save('all_gt.npy',all_gt)
np.save('all_noisy.npy',all_noisy)
print('over')
# print(all_gt.shape)