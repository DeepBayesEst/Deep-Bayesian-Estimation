'''
Author: Yan Shi
Date: 2024/01
Testing code of Transformer for the non-Markov time series filtering
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
model = torch.load('trans_model/train64test64/model.pth').to(device)#Transformer(args)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.004)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)


def data_pack(gt, noisy):
    inputs = torch.from_numpy(noisy.reshape([noisy.shape[0], seq_length, 1])).float()  # 转换为Float
    targets = torch.from_numpy(gt.reshape([gt.shape[0], seq_length, 1])).float()    # 转换为Float
    return TensorDataset(inputs, targets)
def add_gaussian_noise(sequences, noise_std):
    noisy_sequences = sequences+np.random.normal(0, noise_std, sequences.shape)
    return noisy_sequences

seq_length = 100
num_samples = 640
MC_num = 100
noise_std_r = 6.
all_gt = np.load('all_gt.npy')
all_noisy = np.load('all_noisy.npy')
gt_test,_ = all_gt[640:],all_noisy[640:]

noisy_test = np.zeros([gt_test.shape[0],gt_test.shape[1]])
Trans_res = np.zeros([MC_num,gt_test.shape[0],gt_test.shape[1]-1])

for epoch in range(100):
    model.eval()
    with torch.no_grad():
        for bt in range(gt_test.shape[0]):
            noisy_test[bt] = add_gaussian_noise(gt_test[bt], noise_std_r)
        test_dataset = data_pack(gt_test, noisy_test)

        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        for inputs_test, targets_test in test_loader:
            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
            outputs_test = []
            start_pos = 0
            prev_pos = 0
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
                logits = model.forward(H_in, prev_pos)  # h [1, 18, 4096], prev_pos: 0
                outputs_test.append(logits)
                prev_out = logits
                prev_pos = cur_pos
            outputs_test = torch.transpose(torch.stack(outputs_test), 1, 0)

            outputs_test_ = outputs_test.clone().detach().cpu().numpy()
            outputs_test_ = outputs_test_.reshape([64,99])
            Trans_res[epoch] = outputs_test_

            loss = criterion(outputs_test, targets_test[:, 1:])
        print(f'Tset {epoch + 1},Loss: {np.sqrt(loss.item())}')

np.save('Trans_result_q2_r6_train64.npy',Trans_res)

