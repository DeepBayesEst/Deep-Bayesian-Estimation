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
model = Mamba(args).to(device)

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

all_gt = np.load('Train_data.npy')

state_batch = all_gt[:, :, :].reshape([all_gt.shape[0], timestep_size, dim_state])  # bt * 200 * 4
# 生成极坐标量测
meas_batch = deepcopy(state_batch[:, :, :2].reshape([state_batch.shape[0], timestep_size, dim_meas]))
measurement = np.zeros_like(meas_batch)
all_noisy = np.zeros_like(meas_batch)
xi = 0.2
for i in range(all_gt.shape[0]):
    measurement[i, :, 0] = np.arctan2(meas_batch[i, :, 1], meas_batch[i, :, 0]) + (
                1 - xi) * np.random.normal(0, azi_n,timestep_size) + xi * np.random.laplace(0, azi_n * 5, timestep_size)  # 方位角
    measurement[i, :, 1] = np.sqrt(
        np.square(meas_batch[i, :, 0]) + np.square(meas_batch[i, :, 1])) + (1 - xi) * np.random.normal(0,dis_n,timestep_size) + xi * np.random.laplace(
        0, dis_n * 5, timestep_size)

    all_noisy[i, :, 0] = measurement[i, :, 1] * np.cos(measurement[i, :, 0])  # x y huanyuan
    all_noisy[i, :, 1] = measurement[i, :, 1] * np.sin(measurement[i, :, 0])

all_gt[:, :, 0] /= 10000
all_gt[:, :, 1] /= 10000
all_gt[:, :, 2] /= 10000
all_gt[:, :, 3] /= 10000
all_noisy[:, :, 0] /= 10000
all_noisy[:, :, 1] /= 10000

gt_train, noisy_train = all_gt[:232], all_noisy[:232]
gt_test, noisy_test = all_gt[234:], all_noisy[234:]
train_dataset = data_pack(gt_train, noisy_train)
test_dataset = data_pack(gt_test, noisy_test)

# 训练循环
for epoch in range(5000):  # 训练10个epoch
    model.train()
    dataset = train_dataset # generate_time_series_data(seq_length, num_samples)
    # 接下来是您的数据加载和训练循环代码
    Loss_print = []
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        outputs_true = outputs.clone().detach()
        targets_true = targets[:, :].clone().detach()

        # outputs_true = deepcopy(outputs)
        # targets_true = deepcopy(targets[:, 1:])
        # outputs_true[:, :, :2] *= 100000
        # targets_true[:, :, :2] *= 100000
        # outputs_true[:, :, 2:] *= 100000
        # targets_true[:, :, 2:] *= 100000
        # print(outputs_true.float())
        loss_print = criterion(outputs_true.float(), targets_true.float())
        Loss_print.append(np.sqrt(loss_print.item()))
    scheduler.step()

    lr = optimizer.param_groups[0]["lr"]
    print(f'Epoch {epoch + 1}, lr:{lr},Loss: {np.mean(np.array(Loss_print))}')
    # 测试模型
    model.eval()
    with torch.no_grad():
        test_data = test_dataset#generate_time_series_data(seq_length, 64)
        test_loader = DataLoader(test_data, batch_size=26, shuffle=False)
        Loss_print_test = []
        Loss_pos_test = []
        for inputs_test, targets_test in test_loader:
            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
            outputs_test = model(inputs_test)
            loss = criterion(outputs_test, targets_test)
            outputs_true_test = outputs_test.clone().detach()
            targets_true_test = targets_test[:, :].clone().detach()
            # outputs_true_test[:, :, :2] *= 100000
            # targets_true_test[:, :, :2] *= 100000
            # outputs_true_test[:, :, 2:] *= 100000
            # targets_true_test[:, :, 2:] *= 100000
            loss_print_test = criterion(outputs_true_test.float(), targets_true_test.float())
            loss_pos_test = criterion(outputs_true_test[:, :, :2].float(), targets_true_test[:, :, :2].float())

            Loss_print_test.append(np.sqrt(loss_print_test.item()))
            Loss_pos_test.append(np.sqrt(loss_pos_test.item()))

        print(f'Tset {epoch + 1},Loss: {np.mean(np.array(Loss_print_test))},Loss_pos: {np.mean(np.array(Loss_pos_test))}')


    # if epoch % 100 == 0:
    #     torch.save(model, './mamba_model_air/glint_leve_4_high/model.pth')