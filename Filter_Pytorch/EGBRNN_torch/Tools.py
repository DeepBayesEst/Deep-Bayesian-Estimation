import torch
import numpy as np
import os


def Generate_transition_F_CV(batch_size: int, ):
    ''' Generate the transition matrix F of CV model

    Args:
        batch_size (int): batch size

    Returns:
        _type_: transition matrix F with a shape of [batch_size, 4, 4], dtype = torch.float64 (double)
    ''' 

    row_0 = torch.tensor([1., 0., 1., 0.])
    row_1 = torch.tensor([0., 1., 0., 1.])
    row_2 = torch.tensor([0., 0., 1., 0.])
    row_3 = torch.tensor([0., 0., 0., 1.])

    matrix_F = torch.stack([row_0, row_1, row_2, row_3]).repeat(batch_size, 1, 1).double()

    return matrix_F.requires_grad_(False)

def Generate_measuremet_H_Linear(batch_size: int, ):
    '''Generate the measurement matrix H of linear measurement model

    Args:
        batch_size (int): batch size

    Returns:
        _type_: measurement matrix H with a shape of [batch_size, 2, 4], dtype = torch.float64 (double)
    '''

    row_0 = torch.tensor([1., 0., 0., 0.])
    row_1 = torch.tensor([0., 1., 0., 0.])

    matrix_H = torch.stack([row_0, row_1]).repeat(batch_size, 1, 1).double()

    return matrix_H.requires_grad_(False)


class EarlyStopping:

    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path, filepath, step: float = None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, filepath)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path, filepath)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, filepath):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        for name in os.listdir(filepath):
            file = filepath + '/' + name
            if name.split('-')[0] == 'loss_curve':
                continue
            os.remove(file)

        # print(path)
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
        
