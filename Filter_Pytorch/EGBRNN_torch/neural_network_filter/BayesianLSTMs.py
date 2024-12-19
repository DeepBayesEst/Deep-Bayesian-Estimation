import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLSTM(nn.Module):

    def __init__(self,
                 input_size: int,
                 output_size: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 mc_dropout_prob: float = 0.3,
                 linear_dropout_prob: float = 0.1
                 ) -> None:
        super(BayesianLSTM, self).__init__()

        self.i_size = input_size
        self.o_size = output_size

        self.lstm = nn.LSTM(self.i_size, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=False,
                            batch_first=True, dropout=mc_dropout_prob)

        self.linear_dropout = nn.Dropout(linear_dropout_prob)
        self.linear = nn.Linear(hidden_dim, self.o_size)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)

        # output = self.linear_dropout(output)
        output = self.linear(output)

        return output

    def name(self):
        return self.__class__.__name__


class BayesianBLSTM(nn.Module):

    def __init__(self, 
                 input_size: int,
                 output_size: int,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 mc_dropout_prob: float = 0.5,
                 linear_dropout_prob: float = 0.1
                 ) -> None:
        super(BayesianBLSTM, self).__init__()

        self.i_size = input_size
        self.o_size = output_size

        self.blstm = nn.LSTM(self.i_size, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=mc_dropout_prob)
        
        self.linear_dropout = nn.Dropout(linear_dropout_prob)
        self.linear = nn.Linear(hidden_dim * 2, self.o_size)

    def forward(self, x):
        output, (h_n, c_n) = self.blstm(x)

        output = self.linear_dropout(output)
        output = self.linear(output)

        return output
    
    def name(self):
        return self.__class__.__name__
        