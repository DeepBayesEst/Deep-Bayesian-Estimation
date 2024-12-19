'''
Author: Yan Shi
Date: 2024/01
Training code of Transformer for the NCLT localization
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from trans_air import Transformer, ModelArgs
from torch.optim.lr_scheduler import StepLR

data_len = 100
# 模型参数
args = ModelArgs()
# 检查是否有可用的 GPU
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for computation.")
# 模型参数
args = ModelArgs(input_dim=4,output_dim=2,dim=512)

# 实例化模型
model = Transformer(args).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

# 数据加载器
seq_length = data_len
from copy import deepcopy

# train_dataset = generate_time_series_data(seq_length, num_samples_train)
# test_dataset = generate_time_series_data(seq_length, num_samples_test)
timestep_size = data_len
dim_state = 2
dim_meas = 2
batch_size = 6



def data_pack(gt, noisy):
    inputs = torch.from_numpy(noisy.reshape([noisy.shape[0], seq_length, dim_meas])).float()  # 转换为Float
    targets = torch.from_numpy(gt.reshape([gt.shape[0], seq_length, dim_state])).float()  # 转换为Float
    return TensorDataset(inputs, targets)


print('读取数据中...')
data_all = np.load('data_set.npy')
# data_set = np.concatenate([np.load('data_set.npy')[6:24],np.load('data_set.npy')[30:]])

temp1 = np.load('data_set.npy')[42:45]
temp2 = np.load('data_set.npy')[:3]
data_all[:3] = deepcopy(temp1)
data_all[42:45] = deepcopy(temp2)


data_nc = data_all/10

all_gt = data_nc[:,:,:2]
all_noisy = data_nc[:,:,2:]

gt_train, noisy_train = all_gt[:42], all_noisy[:42]
gt_test, noisy_test = all_gt[42:], all_noisy[42:]
train_dataset = data_pack(gt_train, noisy_train)
test_dataset = data_pack(gt_test, noisy_test)

from copy import deepcopy
# 训练循环
for epoch in range(50000):  # 训练10个epoch
    model.train()
    # 接下来是您的数据加载和训练循环代码
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    Loss_print = []
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        print('step : %d / 29')
        outputs = []
        start_pos = 0
        prev_pos = 0
        # prev_out = None
        prev_out = torch.zeros(batch_size, 1, dim_state)
        x0 = torch.reshape(targets[:, 0, :], (batch_size, 1, dim_state))
        H_save = None
        for cur_pos in range(1, seq_length):
            h_pad = torch.zeros((batch_size, (seq_length - 1) - cur_pos, dim_state + dim_meas)).to(device)
            if cur_pos == 1:
                # inputs_0 + 0
                h_now = torch.cat([x0, torch.reshape(inputs[:, cur_pos, :], (batch_size, 1, dim_meas))],
                                  2)  # batch_size,2,1
                H_in = torch.cat([h_now, h_pad], 1)
                H_save = h_now
            else:
                prev_out = torch.reshape(prev_out, (batch_size, 1, dim_state))
                h_now = torch.cat([prev_out, torch.reshape(inputs[:, cur_pos, :], (batch_size, 1, dim_meas))],
                                  2)  # prev_out[:, prev_pos:cur_pos]
                H_save = torch.cat([H_save, h_now], 1)
                H_in = torch.cat([H_save, h_pad], 1)
            logits = model.forward(H_in, prev_pos)  # h [1, 18, 4096], prev_pos: 0
            outputs.append(logits)
            prev_out = logits
            prev_pos = cur_pos

        outputs = torch.transpose(torch.stack(outputs), 1, 0)
        optimizer.zero_grad()
        outputs_true = outputs.clone().detach()
        targets_true = targets[:, 1:].clone().detach()
        outputs_true[:, :, :] *= 10
        targets_true[:, :, :] *= 10
        loss_print = criterion(outputs_true, targets_true)

        Loss_print.append(np.sqrt(loss_print.item()))

        loss = criterion(outputs, targets[:, 1:])
        loss.backward()
        optimizer.step()
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    print(f'Epoch {epoch + 1}, lr:{lr},Loss: {np.mean(np.array(Loss_print))}')

    batch_size2 = 6
    # 测试模型
    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=batch_size2, shuffle=False)
        Loss_print_test = []
        for inputs_test, targets_test in test_loader:
            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
            outputs_test = []
            start_pos = 0
            prev_pos = 0
            # prev_out = None
            prev_out = torch.zeros(batch_size2, 1, dim_state)
            x0 = torch.reshape(targets_test[:, 0, :], (batch_size2, 1, dim_state))
            H_save = None
            for cur_pos in range(1, seq_length):
                h_pad = torch.zeros((batch_size2, (seq_length - 1) - cur_pos, dim_state + dim_meas)).to(device)
                if cur_pos == 1:
                    # inputs_0 + 0
                    h_now = torch.cat([x0, torch.reshape(inputs_test[:, cur_pos, :], (batch_size2, 1, dim_meas))],
                                      2)  # batch_size2,2,1
                    H_in = torch.cat([h_now, h_pad], 1)
                    H_save = h_now
                else:
                    prev_out = torch.reshape(prev_out, (batch_size2, 1, dim_state))
                    h_now = torch.cat([prev_out, torch.reshape(inputs_test[:, cur_pos, :], (batch_size2, 1, dim_meas))],
                                      2)  # prev_out[:, prev_pos:cur_pos]
                    H_save = torch.cat([H_save, h_now], 1)
                    H_in = torch.cat([H_save, h_pad], 1)
                logits = model.forward(H_in, prev_pos)  # h [1, 18, 4096], prev_pos: 0
                outputs_test.append(logits)
                prev_out = logits
                prev_pos = cur_pos
            outputs_test = torch.transpose(torch.stack(outputs_test), 1, 0)
            loss = criterion(outputs_test, targets_test[:, 1:])

            outputs_true_test = outputs_test.clone().detach()
            targets_true_test = targets_test[:, 1:].clone().detach()
            outputs_true_test[:, :, :] *= 10
            targets_true_test[:, :, :] *= 10
            loss_print_test = criterion(outputs_true_test, targets_true_test)
            Loss_print_test.append(np.sqrt(loss_print_test.item()))
        if np.mean(np.array(Loss_print_test)) < 45:
            torch.save(model, './trans_model_nclt/model.pth')
            print('model saved')
        print(f'Tset {epoch + 1},Loss: {np.mean(np.array(Loss_print_test))}')


