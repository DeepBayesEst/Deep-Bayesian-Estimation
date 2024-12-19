import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class LandingAircraft_Dataset(Dataset):

    def __init__(self, do_train: bool = True, train_DB_model: bool = False) -> None:
        super(LandingAircraft_Dataset, self).__init__()

        self.train_DB_model = train_DB_model
        self.scale_ratio = 0.00001
        # maybe you need change the path
        meas_array  = np.load('/home/yan/Downloads/UP_GIT/Deep_Bayesian_estimation/Filter_Pytorch/EGBRNN_torch/data/meas_trajectories.npy')
        label_array = np.load('/home/yan/Downloads/UP_GIT/Deep_Bayesian_estimation/Filter_Pytorch/EGBRNN_torch/data/true_smooth_air_120.npy')

        train_meas_array = meas_array[:5000, :, :]
        test_meas_array  = meas_array[5000:5320, :, :]

        train_label_array = label_array[:5000, :, :]
        test_label_array  = label_array[5000:5320, :, :]

        self.train_meas_scaler = StandardScaler()
        self.train_label_scaler = StandardScaler()

        '''
        直接将所有数据全部除 10000
        '''
        train_meas_list_scaled = train_meas_array * self.scale_ratio
        test_meas_list_scaled  = test_meas_array * self.scale_ratio
        train_label_list_scaled = train_label_array * self.scale_ratio
        test_label_list_scaled  = test_label_array * self.scale_ratio

        
        '''
        使用 sklearn 的 StandardScaler 对数据进行标准化（线性操作）
        '''
        self.train_meas_scaler.fit(np.vstack(train_meas_array))
        self.train_label_scaler.fit(np.vstack(train_label_array))

        if train_DB_model:
            train_meas_list_scaled = [self.train_meas_scaler.transform(item) for item in train_meas_array]
            test_meas_list_scaled  = [self.train_meas_scaler.transform(item) for item in test_meas_array ]
            train_label_list_scaled = [self.train_label_scaler.transform(item) for item in train_label_array]
            test_label_list_scaled  = [self.train_label_scaler.transform(item) for item in test_label_array ]

        if do_train:
            self.sequence_list = train_meas_list_scaled
            self.target_list = train_label_list_scaled
        else:
            self.sequence_list = test_meas_list_scaled
            self.target_list = test_label_list_scaled

        self.input_size = self.sequence_list[0].shape[1]
        self.output_size = self.target_list[0].shape[1]
        self.sequence_length = self.sequence_list[0].shape[0]

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequence_list[idx], dtype=torch.float32)
        target = torch.tensor(self.target_list[idx], dtype=torch.float32)

        return sequence, target

    def __len__(self):
        return len(self.sequence_list)

    def get_inverse_transform(self, scaler: StandardScaler = None, 
                              input_array: np.array = None, ):
        ''' 对预测结果进行逆归一化操作，只有当使用 StandardScaler 时该才使用该函数

        Args:
            scaler (StandardScaler, optional): 归一化操作对象. Defaults to None.
            input_array (np.array, optional): 模型预测输出数据或标签数据. Defaults to None.

        Returns:
            _type_: 逆归一化结果
        '''
        if self.train_DB_model:
            inversed_trans_list = list()

            for item in input_array:
                inversed_trans_list.append(scaler.inverse_transform(item))
            
            return np.array(inversed_trans_list)
        else:
            inversed_trans_list = list()

            for item in input_array:
                inversed_trans_list.append(item * self.scale_ratio)

            return np.array(inversed_trans_list)


# if __name__ == '__main__':
#
#     a = LandingAircraft_Dataset()
#
#     b = np.random.randn(320, 120, 4)
#     c = a.get_inverse_transform(a.train_label_scaler, input_array=b)
#
#     pass
