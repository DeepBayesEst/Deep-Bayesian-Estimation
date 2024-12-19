'''
Author: Yan Shi
Date: 2024/01
Training code of Mamba for the non-Markov time series filtering
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

model = Mamba(args).to(device)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)
scheduler = StepLR(optimizer, step_size=100, gamma=0.5)


def data_pack(gt, noisy):
    inputs = torch.from_numpy(noisy.reshape([noisy.shape[0], seq_length, 1])).half()  #
    targets = torch.from_numpy(gt.reshape([gt.shape[0], seq_length, 1])).half()     #
    return TensorDataset(inputs, targets)

seq_length = 100
num_samples = 640

all_gt = np.load('all_gt.npy')
all_noisy = np.load('all_noisy.npy')
print(all_gt.shape)
gt_train,noisy_train = all_gt[:640],all_noisy[:640]
gt_test,noisy_test = all_gt[640:],all_noisy[640:]
train_dataset = data_pack(gt_train, noisy_train)
test_dataset = data_pack(gt_test, noisy_test)

for epoch in range(500):  #
    model.train()
    dataset = train_dataset
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    scheduler.step()

    lr = optimizer.param_groups[0]["lr"]
    print(f'Epoch {epoch+1}, lr:{lr},Loss: {np.sqrt(loss.item())}')

    model.eval()
    with torch.no_grad():
        test_data = test_dataset
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)
        for inputs_test, targets_test in test_loader:
            inputs_test, targets_test = inputs_test.to(device), targets_test.to(device)
            outputs_test = model(inputs_test)
            loss = criterion(outputs_test, targets_test)
            print(f'Tset {epoch + 1},Loss: {np.sqrt(loss.item())}')
