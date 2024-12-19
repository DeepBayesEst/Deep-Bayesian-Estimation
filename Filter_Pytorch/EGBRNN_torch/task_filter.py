'''
Author: Yan Shi
Date: 2024/01
Training code of EGBRNN for terminal area aircraft tracking
'''

import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math


from torch import amp
from torch import amp
from tqdm import tqdm
from torch.utils.data import DataLoader
from air_dataset import LandingAircraft_Dataset 
from Tools import EarlyStopping
from neural_network_filter import BLSTM,EGBRNN


from torch.utils.tensorboard import SummaryWriter

np.set_printoptions(threshold=np.inf)

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rc('font', family='Times New Roman')

torch.set_default_dtype(torch.float32)
torch.cuda.empty_cache()


'''
Typical Mixed Precision Training
'''
def train(model: nn.Module,
          train_loader: DataLoader,
          test_loader: DataLoader,
          learning_rate: float,
          epochs: int,
          step_lr: bool = True,
          lr_change_step: int = 3,
          gamma: float = 0.996,
          device: str = 'cuda',
          writer: SummaryWriter = None, 
          model_params: str = None):
    '''
    Model and optimizer configuration
    '''
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, lr_change_step, gamma=gamma)
    loss_caculate = nn.MSELoss().to(device)
    scaler = amp.GradScaler(enabled=True)

    '''
    Create a folder to save the weight file. 
    Each time you save a weight file, first clear all the files in the folder and then save the weight.
    '''
    weight_file_path = __file__.split('/')[-1].split('.')[0] + '_' + model.name() + '_' + model_params

    '''
    early_stop logic: Each train epoch will be validated on the test set, and the model with the lowest loss will always be saved. If the optimal model is saved in the i th epoch, and the loss on the test set is still greater than the optimal loss after patience epochs, the training will stop.
    '''
    early_stop = EarlyStopping(patience=1000)

    if not os.path.exists(weight_file_path):
        os.mkdir(weight_file_path)

    with tqdm(total=epochs, desc='Model Training', leave=False) as pbar:
        for epoch in range(1, epochs + 1):
            train_total_loss = 0.
            test_total_loss = 0.

            '''
            Train model
            '''
            model.train()
            loss = None

            for i in train_loader:
                epoch_loss = 0
                sequence = i[0].to(device)
                target = i[1].to(device)

                optimizer.zero_grad(set_to_none=True, target=target)

                '''
                amp.autocast is automatic mixed precision training, which is enabled when enabled == True
                '''
                with amp.autocast(enabled=True, device_type='cuda'):
                    outputs = model(sequence, target)
                    loss = loss_caculate(outputs[:, :, :], target[:, :, :])

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_total_loss += loss.item() / train_loader.__len__()
                epoch_loss += loss.item() / train_loader.__len__()

            if step_lr:
                scheduler.step()

            '''
            Evaluate model
            '''
            model.eval()
            loss = None

            with torch.no_grad():
                for i in test_loader:
                    sequence = i[0].to(device)
                    target = i[1].to(device)

                    outputs = model(sequence, target=target)
                    loss = loss_caculate(outputs[:, :, :2], target[:, :, :2])
                    test_total_loss += loss.item()  / test_loader.__len__()

                '''
                Perform early_stop judgment based on the historical best result. If the current result is better than the historical best, delete the historical best result and save the current result as the historical best result.
                '''
                early_stop(val_loss=test_total_loss, model=model, 
                           path=weight_file_path + '/' + '{}_{}_{:.10f}.pt'.format(model.name(), epoch, test_total_loss, ), 
                           filepath=weight_file_path)

            tqdm.write('Epoch: {:5} | Train Loss: {:8} | Test Loss: {:8} | LR: {:8}'.format(epoch, train_total_loss, test_total_loss, scheduler.get_last_lr()[0]))
            pbar.update(1)

            if early_stop.early_stop:
                print('Early stopping')
                break



# def test_dl(model: nn.Module,
#             test_dataset: LandingAircraft_Dataset,
#             device: str = 'cuda'):
#     model.to(device)
#     model.eval()
#
#     target_array_list = list()
#     output_array_list = list()
#
#     for idx in range(test_dataset.__len__()):
#         input_tensor = test_dataset[idx][0].unsqueeze(0).to(device)
#         target_array = test_dataset[idx][1].numpy()
#
#         output_array = model(input_tensor).squeeze(0).detach().cuda().numpy()
#
#         target_array_list.append(target_array)
#         output_array_list.append(output_array)
#
#     target_array = np.array(target_array_list)
#     output_array = np.array(output_array_list)
#
#     target_array = test_dataset.get_inverse_transform(test_dataset.train_label_scaler, target_array)
#     output_array = test_dataset.get_inverse_transform(test_dataset.train_label_scaler, output_array)
#
#     np.save('./data/predict_result.npy', [target_array, output_array])
def test_dl(model: EGBRNN,
                test_dataset: LandingAircraft_Dataset,
                device: str = 'cuda'):
    model.to(device)
    model.eval()

    target_array_list = list()
    output_array_list = list()

    for idx in range(test_dataset.__len__()):
        input_tensor  = test_dataset[idx][0].unsqueeze(0).to(device)
        target_tensor = test_dataset[idx][1].unsqueeze(0).to(device)
        target_array = test_dataset[idx][1].numpy()

        output_array = model(input_tensor, target_tensor).squeeze(0).detach().cuda().numpy()

        target_array_list.append(target_array)
        output_array_list.append(output_array)

    target_array = np.array(target_array_list)
    output_array = np.array(output_array_list)

    target_array = test_dataset.get_inverse_transform(test_dataset.train_label_scaler, target_array)
    output_array = test_dataset.get_inverse_transform(test_dataset.train_label_scaler, output_array)

    np.save('./data/predict_result.npy', [target_array, output_array])

def Generate_transition_F_CV(sT: float = 1.):
    row_0 = np.array([1., 0., sT, 0.])
    row_1 = np.array([0., 1., 0., sT])
    row_2 = np.array([0., 0., 1., 0.])
    row_3 = np.array([0., 0., 0., 1.])

    mat_F = np.stack([row_0, row_1, row_2, row_3])

    return mat_F


def Generate_measuremet_H_Linear():
    row_0 = np.array([1., 0., 0., 0.])
    row_1 = np.array([0., 1., 0., 0.])

    mat_H = np.stack([row_0, row_1])

    return mat_H

def Generate_process_noise():
    # row_0 = np.array([1.8125,     0., 0.6042,     0.])
    # row_1 = np.array([    0., 1.8125,     0., 0.6042])
    # row_2 = np.array([0.6042,     0., 0.2014,     0.])
    # row_3 = np.array([    0., 0.6042,     0., 0.2014])

    # row_0 = np.array([1.3794e4,       0., 551.7477,       0.])
    # row_1 = np.array([      0., 1.3794e4,       0., 551.7477])
    # row_2 = np.array([551.7477,       0.,  22.0699,       0.])
    # row_3 = np.array([      0., 551.7477,       0.,  22.0699])

    # matrix_Q = np.stack([row_0, row_1, row_2, row_3])

    mat_Q = np.diag([1, 1, 1, 1]) * 1

    return mat_Q

def Generate_measurement_noise():
    row_0 = np.array([1., 0.])
    row_1 = np.array([0., 1.])

    matrix_R = np.stack([row_0, row_1]) * 1

    return matrix_R

def Generate_delta_k_f_transition_F(sT: float = 1.):
    row_0 = np.array([sT,  0, (1/2) * (sT**2),               0])
    row_1 = np.array([ 0, sT,               0, (1/2) * (sT**2)])
    row_2 = np.array([ 0,  0,              sT,               0])
    row_3 = np.array([ 0,  0,               0,              sT])

    matrix_delta_F = np.stack([row_0, row_1, row_2, row_3])

    return matrix_delta_F


if __name__ == '__main__':

    device = torch.device('cuda:0')
    torch.set_num_threads(8)

    '''
    Hyperparameters
    '''
    batch_size = 1000
    learning_rate = 1e-4
    epochs = 50000

    do_train = True
    do_test  = not do_train

    '''
    - Train EGBRNN --------------------------------------------------------------------------------------
    '''

    mat_F = Generate_transition_F_CV(sT=1)
    mat_H = Generate_measuremet_H_Linear()
    mat_Q = Generate_process_noise()
    mat_R = Generate_measurement_noise()
    mat_delta_F = Generate_delta_k_f_transition_F(sT=1)

    # EGBRNN's parameters
    memory_dim = 32
    state_dim = 4
    measurement_dim = 2

    model_params_str = '{}_{}_{}'.format(memory_dim, state_dim, measurement_dim)

    train_dataset = LandingAircraft_Dataset(do_train=True , train_DB_model=False)
    test_dataset  = LandingAircraft_Dataset(do_train=False, train_DB_model=False)

    train_loader = DataLoader(train_dataset, shuffle=True , batch_size=batch_size, drop_last=False, ) 
    test_loader  = DataLoader(test_dataset , shuffle=False, batch_size=1, drop_last=False, )

    model = EGBRNN(input_size=train_dataset.input_size,
                   hidden_dim=memory_dim,
                   output_size=train_dataset.output_size,
                   state_dim=state_dim, meas_dim=measurement_dim,
                   transition_model=mat_F, measurement_model=mat_H,
                   F_first_order_transition_model=mat_delta_F,
                   process_noise=mat_Q, measurement_noise=mat_R, device=device)
    
    if do_train:
        train(model=model, 
              train_loader=train_loader, 
              test_loader=test_loader, 
              learning_rate=learning_rate, 
              epochs=epochs, 
              device=device, 
              model_params=model_params_str)
    if do_test:
        print(__file__)
        folder_path = f'./{os.path.basename(__file__).split(".")[0]}_{model.name}_{model_params_str}/'
        for name in os.listdir(folder_path):
            if name.split('_')[0] == model.name:
                weight_path = folder_path + name
        model.load_state_dict(torch.load(weight_path, weights_only=True))
        test_dl(model=model, test_dataset=test_dataset)