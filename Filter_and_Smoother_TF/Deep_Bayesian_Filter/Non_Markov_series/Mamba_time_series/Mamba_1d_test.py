'''
Author: Yan Shi
Date: 2024/01
Testing code of Mamba for the non-Markov time series filtering
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model_my import Mamba, ModelArgs
from torch.optim.lr_scheduler import StepLR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for computation.")

args = ModelArgs(input_size=1, output_size=1, d_model=256)
model = torch.load('mamba_model/train64test64/model.pth').to(device)#Mamba(args)
criterion = nn.MSELoss()

def data_pack(gt, noisy):
    inputs = torch.from_numpy(noisy.reshape([noisy.shape[0], seq_length, 1])).half()  # 转换为Float
    targets = torch.from_numpy(gt.reshape([gt.shape[0], seq_length, 1])).half()     # 转换为Float
    return TensorDataset(inputs, targets)
def add_gaussian_noise(sequences, noise_std):
    noisy_sequences = sequences+np.random.normal(0, noise_std, sequences.shape)
    return noisy_sequences
seq_length = 100
num_samples = 640
MC_num = 100
noise_std_r = 6
all_gt = np.load('all_gt.npy')
all_noisy = np.load('all_noisy.npy')
gt_test,_ = all_gt[640:],all_noisy[640:]
noisy_test = np.zeros([gt_test.shape[0],gt_test.shape[1]])
MAM_res = np.zeros([MC_num,gt_test.shape[0],gt_test.shape[1]])

for epoch in range(MC_num):  #
    print(epoch)
    model.eval()
    with torch.no_grad():
        for bt in range(gt_test.shape[0]):
            noisy_test[bt] = add_gaussian_noise(gt_test[bt], noise_std_r)

        test_dataset = data_pack(gt_test, noisy_test)

        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        for inputs_test, targets_test in test_loader:
            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
            outputs_test = model(inputs_test)
            outputs_test_ = outputs_test.clone().detach().cpu().numpy()
            outputs_test_ = outputs_test_.reshape([64,100])
            MAM_res[epoch] = outputs_test_
            # MAM_res[epoch,]
            loss = criterion(outputs_test, targets_test)
            print(f'Tset {epoch + 1},Loss: {np.sqrt(loss.item())}')
np.save('MAM_result_q2_r6_train64.npy',MAM_res)
