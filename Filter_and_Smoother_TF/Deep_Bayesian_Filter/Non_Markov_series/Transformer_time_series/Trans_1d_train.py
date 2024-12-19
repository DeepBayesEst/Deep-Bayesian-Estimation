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
from trans import Transformer,ModelArgs
from torch.optim.lr_scheduler import StepLR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for computation.")

data_len = 100
args = ModelArgs()

model = Transformer(args).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.004)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

def data_pack(gt, noisy):
    inputs = torch.from_numpy(noisy.reshape([noisy.shape[0], seq_length, 1])).float()  # 转换为Float
    targets = torch.from_numpy(gt.reshape([gt.shape[0], seq_length, 1])).float()    # 转换为Float
    return TensorDataset(inputs, targets)
# 数据加载器
seq_length = data_len
num_samples_train = 640
num_samples_test = 64



all_gt = np.load('all_gt.npy')
all_noisy = np.load('all_noisy.npy')
print(all_gt.shape)
gt_train,noisy_train = all_gt[:640],all_noisy[:640]
gt_test,noisy_test = all_gt[640:],all_noisy[640:]
train_dataset = data_pack(gt_train, noisy_train)
test_dataset = data_pack(gt_test, noisy_test)


from copy import deepcopy

for epoch in range(200):
    model.train()
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = []
        start_pos = 0
        prev_pos = 0
        # prev_out = None
        prev_out = torch.zeros(64, 1, 1)
        x0 = torch.reshape(targets[:,0,:],(64,1,1))
        H_save = None
        for cur_pos in range(1,seq_length):
            h_pad = torch.zeros((64,(seq_length-1)-cur_pos,2)).to(device)
            if cur_pos == 1:
                # inputs_0 + 0
                h_now = torch.cat([x0,torch.reshape(inputs[:,cur_pos,:],(64,1,1))], 2)  # 64,2,1
                H_in = torch.cat([h_now,h_pad],1)
                H_save = h_now
            else:
                prev_out = torch.reshape(prev_out,(64,1,1))
                h_now = torch.cat([prev_out,torch.reshape(inputs[:,cur_pos,:],(64,1,1))], 2)#prev_out[:, prev_pos:cur_pos]
                H_save = torch.cat([H_save,h_now],1)
                H_in = torch.cat([H_save,h_pad],1)
            logits = model.forward(H_in, prev_pos)  # h [1, 18, 4096], prev_pos: 0
            outputs.append(logits)
            prev_out = logits
            prev_pos = cur_pos
        outputs = torch.transpose(torch.stack(outputs),1,0)
        optimizer.zero_grad()
        loss = criterion(outputs, targets[:,1:])
        loss.backward()
        optimizer.step()
    scheduler.step()
    lr = optimizer.param_groups[0]["lr"]
    print(f'Epoch {epoch+1}, lr:{lr},Loss: {np.sqrt(loss.item())}')

    # if epoch % 10 ==0:
    #     torch.save(model, 'trans_model/train640_48/model.pth')

    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        for inputs_test, targets_test in test_loader:
            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
            outputs_test = []
            start_pos = 0
            prev_pos = 0
            # prev_out = None
            prev_out = torch.zeros(64, 1, 1)
            x0 = torch.reshape(targets_test[:, 0, :], (64, 1, 1))
            H_save = None
            for cur_pos in range(1, seq_length):
                h_pad = torch.zeros((64, (seq_length - 1) - cur_pos, 2)).to(device)
                if cur_pos == 1:
                    # inputs_0 + 0
                    h_now = torch.cat([x0, torch.reshape(inputs_test[:, cur_pos, :], (64, 1, 1))], 2)  # 64,2,1
                    H_in = torch.cat([h_now, h_pad], 1)
                    H_save = h_now
                else:
                    prev_out = torch.reshape(prev_out, (64, 1, 1))
                    h_now = torch.cat([prev_out, torch.reshape(inputs_test[:, cur_pos, :], (64, 1, 1))],
                                      2)  # prev_out[:, prev_pos:cur_pos]
                    H_save = torch.cat([H_save, h_now], 1)
                    H_in = torch.cat([H_save, h_pad], 1)
                print(H_in.shape)
                print(prev_pos)
                logits = model.forward(H_in, prev_pos)  # h [1, 18, 4096], prev_pos: 0
                outputs_test.append(logits)
                prev_out = logits
                prev_pos = cur_pos
            outputs_test = torch.transpose(torch.stack(outputs_test), 1, 0)
            loss = criterion(outputs_test, targets_test[:, 1:])
        print(f'Tset {epoch + 1},Loss: {np.sqrt(loss.item())}')


