import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from mamba_air import Mamba, ModelArgs
from torch.optim.lr_scheduler import StepLR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for computation.")
data_len = 200
# 模型参数
args = ModelArgs(input_size=2, output_size=4, d_model=256)

# 实例化模型
model = torch.load('./mamba_model_air/glint_leve_4_low/model.pth')#Mamba(args).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.004)
scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

# 数据生成函数
def data_pack(gt, noisy):
    inputs = torch.from_numpy(noisy.reshape([noisy.shape[0], data_len, dim_meas])).half()  # 转换为Float
    targets = torch.from_numpy(gt.reshape([gt.shape[0], data_len, dim_state])).half()     # 转换为Float
    return TensorDataset(inputs, targets)
from copy import deepcopy
# 数据加载器
# test_dataset = generate_time_series_data(seq_length, num_samples_test)
timestep_size = data_len
azi_n = 0.3 * np.pi / 180
dis_n = 150
dim_state = 4
dim_meas = 2
batch_size = 8

Resu = np.zeros([26,100,200,4])

for mc in range(100):
    all_gt = np.load('Train_data.npy')

    state_batch = all_gt[:, :, :].reshape([all_gt.shape[0], timestep_size, dim_state])  # bt * 200 * 4

    # 生成极坐标量测
    meas_batch = deepcopy(state_batch[:, :, :2].reshape([state_batch.shape[0], timestep_size, dim_meas]))
    measurement = np.zeros_like(meas_batch)
    all_noisy = np.zeros_like(meas_batch)
    xi = 0.2
    for i in range(all_gt.shape[0]):
        measurement[i, :, 0] = np.arctan2(meas_batch[i, :, 1], meas_batch[i, :, 0]) + (
                1 - xi) * np.random.normal(0, azi_n, timestep_size) + xi * np.random.laplace(0, azi_n * 2,
                                                                                             timestep_size)  # 方位角
        measurement[i, :, 1] = np.sqrt(
            np.square(meas_batch[i, :, 0]) + np.square(meas_batch[i, :, 1])) + (1 - xi) * np.random.normal(0, dis_n,
                                                                                                           timestep_size) + xi * np.random.laplace(
            0, dis_n * 2, timestep_size)
        all_noisy[i, :, 0] = measurement[i, :, 1] * np.cos(measurement[i, :, 0])  # x y huanyuan
        all_noisy[i, :, 1] = measurement[i, :, 1] * np.sin(measurement[i, :, 0])

    all_gt[:, :, 0] /= 10000
    all_gt[:, :, 1] /= 10000
    all_gt[:, :, 2] /= 10000
    all_gt[:, :, 3] /= 10000
    all_noisy[:, :, 0] /= 10000
    all_noisy[:, :, 1] /= 10000
    gt_test, noisy_test = all_gt[234:], all_noisy[234:]
    test_dataset = data_pack(gt_test, noisy_test)

    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=26, shuffle=False)
        Loss_print_test = []
        Loss_pos_test = []
        for inputs_test, targets_test in test_loader:
            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
            outputs_test = model(inputs_test)
            loss = criterion(outputs_test, targets_test)
            outputs_true_test = outputs_test.clone().detach()
            targets_true_test = targets_test[:, :].clone().detach()
            loss_print_test = criterion(outputs_true_test.float(), targets_true_test.float())
            loss_pos_test = criterion(outputs_true_test[:, :, :2].float(), targets_true_test[:, :, :2].float())
            Loss_print_test.append(np.sqrt(loss_print_test.item()))
            Loss_pos_test.append(np.sqrt(loss_pos_test.item()))
        outputs_true = outputs_true_test
        Resu[:,mc,:,:] = outputs_true.cpu().numpy()
        print(f'Tset ,Loss: {np.mean(np.array(Loss_print_test))},Loss_pos: {np.mean(np.array(Loss_pos_test))}')

np.save('Mamba_result_l4_low_glint.npy',Resu)


