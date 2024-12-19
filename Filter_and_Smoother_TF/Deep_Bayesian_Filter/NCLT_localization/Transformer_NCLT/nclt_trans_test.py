'''
Author: Yan Shi
Date: 2024/01
Testing code of Transformer for the NCLT localization
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from trans_air import Transformer, ModelArgs
from torch.optim.lr_scheduler import StepLR

data_len = 100
args = ModelArgs()
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for computation.")
args = ModelArgs(input_dim=4,output_dim=2,dim=512)
model = torch.load('./trans_model_nclt/model.pth').to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

seq_length = data_len
from copy import deepcopy

timestep_size = data_len
dim_state = 2
dim_meas = 2
batch_size = 6


def data_pack(gt, noisy):
    inputs = torch.from_numpy(noisy.reshape([noisy.shape[0], seq_length, dim_meas])).float()  # 转换为Float
    targets = torch.from_numpy(gt.reshape([gt.shape[0], seq_length, dim_state])).float()  # 转换为Float
    return TensorDataset(inputs, targets)


print('Reading data...')
data_all = np.load('data_set.npy')

data_nc = data_all/10

all_gt = data_nc[:,:,:2]
all_noisy = data_nc[:,:,2:]

gt_test, noisy_test = all_gt[42:], all_noisy[42:]
test_dataset = data_pack(gt_test, noisy_test)
import time
for epoch in range(10):  #
    batch_size2 = 6
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
            start = time.time()
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
            end = time.time()
            print(end-start)
            outputs_test = torch.transpose(torch.stack(outputs_test), 1, 0)
            loss = criterion(outputs_test, targets_test[:, 1:])

            outputs_true_test = outputs_test.clone().detach().cpu()
            targets_true_test = targets_test[:, 1:].clone().detach().cpu()

            outputs_true_test[:, :, :] *= 10
            targets_true_test[:, :, :] *= 10

            outputs_true_ = outputs_true_test.numpy()
            targets_true_ = targets_true_test.numpy()
            loss_print_test = criterion(outputs_true_test, targets_true_test)
            Loss_print_test.append(np.sqrt(loss_print_test.item()))
            np.save('trans_ncly.npy',outputs_true_)
        print(f'Tset {epoch + 1},Loss: {np.mean(np.array(Loss_print_test))}')



