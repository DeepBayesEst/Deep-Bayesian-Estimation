import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from typing import Tuple
import math
import cmath
import scipy

torch.set_default_dtype(torch.float64)


class EGBRNN(nn.Module):

    def __init__(self, input_size: int, hidden_dim: int, output_size: int,
                 state_dim: int, meas_dim: int, 
                 transition_model: np.array, measurement_model: np.array, 
                 F_first_order_transition_model: np.array,
                 process_noise: np.array, measurement_noise: np.array,
                 device: str = 'cuda') -> None:
        super(EGBRNN, self).__init__()

        self.__i_size = input_size
        self.__h_size = hidden_dim
        self.__o_size = output_size

        self.__state_dim = state_dim
        self.__meas_dim = meas_dim

        self.__acti_fun = torch.tanh

        self.__trans_model = torch.Tensor(transition_model).requires_grad_(False).to(device)
        self.__meas_model = torch.Tensor(measurement_model).requires_grad_(False).to(device)

        self.__F_fom_trans_model = torch.Tensor(F_first_order_transition_model).requires_grad_(False).to(device)

        self.__pro_noise = torch.Tensor(process_noise).requires_grad_(False).to(device)
        self.__meas_noise = torch.Tensor(measurement_noise).requires_grad_(False).to(device)

        '''
        Replace all matrix multiplications with linear layers
        '''

        ''' MUG '''
        # self._kernel_memory = nn.Parameter(torch.Tensor(self.__state_dim + self.__h_size), self.__h_size)
        # self._kernel_memory_bias = nn.Parameter(torch.Tensor(self.__h_size))

        self.__kernel_memory_l0 = nn.Linear(in_features=self.__state_dim + self.__h_size, out_features=self.__h_size, bias=True)
        self.__kernel_memory_l1 = nn.Linear(in_features=self.__state_dim + self.__h_size + self.__meas_dim, out_features=self.__h_size, bias=True)

        ''' SPG '''
        # self._F_first_order_moment_l0 = nn.Parameter(torch.Tensor(self.__state_dim + self.__h_size, self.__h_size))
        # self._F_first_order_moment_l1 = nn.Parameter(torch.Tensor(self.__h_size, self.__o_size))
        # self._F_first_order_moment_l0_bias = nn.Parameter(torch.Tensor(self.__h_size))
        # self._F_first_order_moment_l1_bias = nn.Parameter(torch.Tensor(self.__o_size))

        self.__F_first_order_moment_l0 = nn.Linear(in_features=self.__state_dim + self.__h_size, out_features=self.__h_size, bias=True)
        self.__F_first_order_moment_l1 = nn.Linear(in_features=self.__h_size, out_features=self.__F_fom_trans_model.shape[-1], bias=True)

        # self._F_second_order_moment_l0 = nn.Parameter(torch.Tensor(self.__state_dim + self.__h_size, self.__h_size))
        # self._F_second_order_moment_l1 = nn.Parameter(torch.Tensor(self.__h_size, self.__state_dim))
        # self._F_second_order_moment_l0_bias = nn.Parameter(torch.Tensor(self.__h_size))
        # self._F_second_order_moment_l1_bias = nn.Parameter(torch.Tensor(self.__state_dim))

        self.__F_second_order_moment_l0 = nn.Linear(in_features=self.__state_dim + self.__h_size, out_features=self.__h_size, bias=True)
        self.__F_second_order_moment_l1 = nn.Linear(in_features=self.__h_size, out_features=self.__state_dim, bias=True)

        ''' SUG '''
        # self._H_second_order_moment_l0 = nn.Parameter(torch.Tensor(self.__state_dim + self.__h_size + self.__meas_dim, self.__h_size))
        # self._H_second_order_moment_l1 = nn.Parameter(torch.Tensor(self.__h_size, self.__state_dim))
        # self._H_second_order_moment_l0_bias = nn.Parameter(torch.Tensor(self.__h_size))
        # self._H_second_order_moment_l1_bias = nn.Parameter(torch.Tensor(self.__state_dim))

        self.__H_second_order_moment_l0 = nn.Linear(in_features=self.__state_dim + self.__h_size + self.__meas_dim, out_features=self.__h_size, bias=True)
        self.__H_second_order_moment_l1 = nn.Linear(in_features=self.__h_size, out_features=self.__meas_dim, bias=True)

    def forward(self, x: torch.Tensor, target: torch.Tensor):

        batch_size, seq_len, input_dim = x.shape

        '''
        Initialize the memory unit c_start and use xavier_uniform_ to initialize its weight parameters
        '''
        c_start = torch.empty(batch_size, 1, self.__h_size).to(x.device)
        nn.init.xavier_uniform_(c_start, gain=nn.init.calculate_gain('sigmoid'))

        '''
        Initialize the target state. The target state of the first frame is initialized directly using the true value label. Since batch training is used, the original batch dimension needs to be expanded.
        '''
        state_start = torch.zeros(batch_size, 1, self.__state_dim).to(x.device)
        state_start[:, 0, 0] = target[:, 0, 0]
        state_start[:, 0, 1] = target[:, 0, 1]
        state_start[:, 0, 2] = target[:, 0, 2]
        state_start[:, 0, 3] = target[:, 0, 3]

        '''
        Initialize the state error covariance matrix
        '''
        p_start = torch.eye(self.__state_dim).repeat(batch_size, 1, 1).to(x.device)

        '''
        Package the initialized memory, state, and covariance and iterate them in the following loop
        '''
        mgbrnn_state_per_time_step = (c_start, state_start, p_start)

        state_output_list = list()

        for time_step in range(seq_len):
            measurement_per_step = x[:, time_step, :].unsqueeze(1)

            state_per_step, mgbrnn_state_per_time_step = self.forward_one_step(meas_input=measurement_per_step,
                                                                               state_input=mgbrnn_state_per_time_step)
            state_output_list.append(state_per_step)

        output = torch.cat(state_output_list, dim=1)

        return output

    # meas_input shape: [batch_size, 1, meas_dim]
    def forward_one_step(self, meas_input: torch.Tensor, 
                         state_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        
        c_prev, state_prev, p_prev = state_input

        '''
        Memory update gate
        '''
        c_update = self.__acti_fun(self.__kernel_memory_l0(torch.cat((c_prev, state_prev / 60), dim=-1)))

        '''
        State prediction gate
        '''
        delta_k_f = self.__acti_fun(self.__F_first_order_moment_l0(torch.cat((c_update, state_prev / 60), dim=-1)))
        delta_k_f = self.__F_first_order_moment_l1(delta_k_f)

        state_pred = (self.__trans_model @ state_prev.permute([0, 2, 1])).permute([0, 2, 1]) + \
                     (self.__F_fom_trans_model @ delta_k_f.permute([0, 2, 1])).permute([0, 2, 1])
        
        p_k_f_vector = self.__acti_fun(self.__F_second_order_moment_l0(torch.cat((c_update, state_prev / 60), dim=-1)))
        p_k_f_vector = self.__F_second_order_moment_l1(p_k_f_vector)
        p_k_f = torch.bmm(p_k_f_vector.permute([0, 2, 1]), p_k_f_vector)

        p_pred = self.__trans_model @ p_prev @ (self.__trans_model).permute([1, 0]) + \
                 p_k_f + \
                 self.__pro_noise
        
        '''
        State update gate
        '''
        inna_info = meas_input - (self.__meas_model @ state_pred.permute([0, 2, 1])).permute([0, 2, 1])

        p_k_h_vector = self.__acti_fun(self.__H_second_order_moment_l0(torch.cat((c_update, state_prev / 60, meas_input / 60), dim=-1)))
        p_k_h_vector = self.__H_second_order_moment_l1(p_k_h_vector)
        p_k_h = torch.bmm(p_k_h_vector.permute([0, 2, 1]), p_k_h_vector)

        s = self.__meas_model @ p_pred @ (self.__meas_model).permute([1, 0]) + \
            p_k_h + \
            self.__meas_noise
        
        k_gain = torch.bmm(p_pred @ (self.__meas_model).permute([1, 0]), torch.linalg.inv(s))

        state_update = state_pred + torch.bmm(k_gain, inna_info.permute([0, 2, 1])).permute([0, 2, 1])

        p_update = p_pred - torch.matmul(torch.matmul(k_gain, s), k_gain.permute([0, 2, 1]))

        '''
        Memory update
        '''
        c_update = self.__acti_fun(self.__kernel_memory_l1(torch.cat((c_update, meas_input / 60, state_update / 60), dim=-1)))

        return state_update, (c_update, state_update, p_update)

    def name(self):
        return self.__class__.__name__


class EGBLSTM(nn.Module):
    '''
    Different from the above EGBRNN code, MGBLSTM replaces the unit that updates the memory in EGBRNN from the linear layer to LSTM to obtain more stable long-term memory.
    '''

    def __init__(self, input_size: int, hidden_dim: int, output_size: int,
                 state_dim: int, meas_dim: int, 
                 transition_model: np.array, measurement_model: np.array, 
                 F_first_order_transition_model: np.array,
                 process_noise: np.array, measurement_noise: np.array,
                 device: str = 'cuda') -> None:
        super(EGBLSTM, self).__init__()

        self.__i_size = input_size
        self.__h_size = hidden_dim
        self.__o_size = output_size

        self.__state_dim = state_dim
        self.__meas_dim = meas_dim

        self.__acti_fun = torch.tanh

        self.__trans_model = torch.Tensor(transition_model).requires_grad_(False).to(device)
        self.__meas_model = torch.Tensor(measurement_model).requires_grad_(False).to(device)

        self.__F_fom_trans_model = torch.Tensor(F_first_order_transition_model).requires_grad_(False).to(device)

        self.__pro_noise = torch.Tensor(process_noise).requires_grad_(False).to(device)
        self.__meas_noise = torch.Tensor(measurement_noise).requires_grad_(False).to(device)

        ''' MUG '''
        # self._kernel_memory = nn.Parameter(torch.Tensor(self.__state_dim + self.__h_size), self.__h_size)
        # self._kernel_memory_bias = nn.Parameter(torch.Tensor(self.__h_size))

        # self.__kernel_memory_l0 = nn.Linear(in_features=self.__state_dim + self.__h_size, out_features=self.__h_size, bias=True)
        # self.__kernel_memory_l1 = nn.Linear(in_features=self.__state_dim + self.__h_size + self.__meas_dim, out_features=self.__h_size, bias=True)

        self.__kernel_memory = nn.LSTMCell(input_size=self.__state_dim + self.__h_size, hidden_size=self.__h_size)

        ''' SPG '''
        # self._F_first_order_moment_l0 = nn.Parameter(torch.Tensor(self.__state_dim + self.__h_size, self.__h_size))
        # self._F_first_order_moment_l1 = nn.Parameter(torch.Tensor(self.__h_size, self.__o_size))
        # self._F_first_order_moment_l0_bias = nn.Parameter(torch.Tensor(self.__h_size))
        # self._F_first_order_moment_l1_bias = nn.Parameter(torch.Tensor(self.__o_size))

        self.__F_first_order_moment_l0 = nn.Linear(in_features=self.__state_dim + self.__h_size, out_features=self.__h_size, bias=True)
        self.__F_first_order_moment_l1 = nn.Linear(in_features=self.__h_size, out_features=self.__F_fom_trans_model.shape[-1], bias=True)

        # self._F_second_order_moment_l0 = nn.Parameter(torch.Tensor(self.__state_dim + self.__h_size, self.__h_size))
        # self._F_second_order_moment_l1 = nn.Parameter(torch.Tensor(self.__h_size, self.__state_dim))
        # self._F_second_order_moment_l0_bias = nn.Parameter(torch.Tensor(self.__h_size))
        # self._F_second_order_moment_l1_bias = nn.Parameter(torch.Tensor(self.__state_dim))

        self.__F_second_order_moment_l0 = nn.Linear(in_features=self.__state_dim + self.__h_size, out_features=self.__h_size, bias=True)
        self.__F_second_order_moment_l1 = nn.Linear(in_features=self.__h_size, out_features=self.__state_dim, bias=True)

        ''' SUG '''
        # self._H_second_order_moment_l0 = nn.Parameter(torch.Tensor(self.__state_dim + self.__h_size + self.__meas_dim, self.__h_size))
        # self._H_second_order_moment_l1 = nn.Parameter(torch.Tensor(self.__h_size, self.__state_dim))
        # self._H_second_order_moment_l0_bias = nn.Parameter(torch.Tensor(self.__h_size))
        # self._H_second_order_moment_l1_bias = nn.Parameter(torch.Tensor(self.__state_dim))

        self.__H_second_order_moment_l0 = nn.Linear(in_features=self.__state_dim + self.__h_size + self.__meas_dim, out_features=self.__h_size, bias=True)
        self.__H_second_order_moment_l1 = nn.Linear(in_features=self.__h_size, out_features=self.__meas_dim, bias=True)

    def forward(self, x: torch.Tensor, target: torch.Tensor):

        batch_size, seq_len, input_dim = x.shape

        c_start = torch.empty(batch_size, 1, self.__h_size).to(x.device)
        nn.init.xavier_uniform_(c_start, gain=nn.init.calculate_gain('sigmoid'))

        lstm_h_start = torch.empty(batch_size, self.__h_size).to(x.device)
        nn.init.xavier_uniform_(lstm_h_start, gain=nn.init.calculate_gain('sigmoid'))
        lstm_c_start = torch.empty(batch_size, self.__h_size).to(x.device)
        nn.init.xavier_uniform_(lstm_c_start, gain=nn.init.calculate_gain('sigmoid'))
        
        state_start = torch.zeros(batch_size, 1, self.__state_dim).to(x.device)
        state_start[:, 0, 0] = target[:, 0, 0]
        state_start[:, 0, 1] = target[:, 0, 1]
        state_start[:, 0, 2] = target[:, 0, 2]
        state_start[:, 0, 3] = target[:, 0, 3]

        p_start = torch.eye(self.__state_dim).repeat(batch_size, 1, 1).to(x.device)

        mgbrnn_state_per_time_step = (c_start, state_start, p_start, lstm_h_start, lstm_c_start)

        state_output_list = list()

        for time_step in range(seq_len):
            measurement_per_step = x[:, time_step, :].unsqueeze(1)

            state_per_step, mgbrnn_state_per_time_step = self.forward_one_step(meas_input=measurement_per_step,
                                                                               state_input=mgbrnn_state_per_time_step)
            state_output_list.append(state_per_step)

        output = torch.cat(state_output_list, dim=1)

        return output

    # meas_input shape: [batch_size, 1, meas_dim]
    def forward_one_step(self, meas_input: torch.Tensor, 
                         state_input: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        
        c_prev, state_prev, p_prev, lstm_h_prev, lstm_c_prev = state_input

        '''
        Memory update gate
        '''
        lstm_h_prev, lstm_c_prev = self.__kernel_memory(torch.cat((c_prev, state_prev / 60), dim=-1).squeeze(1), (lstm_h_prev, lstm_c_prev))
        c_update = lstm_h_prev.unsqueeze(1)

        '''
        State prediction gate
        '''
        delta_k_f = self.__acti_fun(self.__F_first_order_moment_l0(torch.cat((c_update, state_prev / 60), dim=-1)))
        delta_k_f = self.__F_first_order_moment_l1(delta_k_f)

        state_pred = (self.__trans_model @ state_prev.permute([0, 2, 1])).permute([0, 2, 1]) + \
                     (self.__F_fom_trans_model @ delta_k_f.permute([0, 2, 1])).permute([0, 2, 1])
        
        p_k_f_vector = self.__acti_fun(self.__F_second_order_moment_l0(torch.cat((c_update, state_prev / 60), dim=-1)))
        p_k_f_vector = self.__F_second_order_moment_l1(p_k_f_vector)
        p_k_f = torch.bmm(p_k_f_vector.permute([0, 2, 1]), p_k_f_vector)

        p_pred = self.__trans_model @ p_prev @ (self.__trans_model).permute([1, 0]) + \
                 p_k_f + \
                 self.__pro_noise
        
        '''
        State update gate
        '''
        inna_info = meas_input - (self.__meas_model @ state_pred.permute([0, 2, 1])).permute([0, 2, 1])

        p_k_h_vector = self.__acti_fun(self.__H_second_order_moment_l0(torch.cat((c_update, state_prev / 60, meas_input / 60), dim=-1)))
        p_k_h_vector = self.__H_second_order_moment_l1(p_k_h_vector)
        p_k_h = torch.bmm(p_k_h_vector.permute([0, 2, 1]), p_k_h_vector)

        s = self.__meas_model @ p_pred @ (self.__meas_model).permute([1, 0]) + \
            p_k_h + \
            self.__meas_noise
        
        k_gain = torch.bmm(p_pred @ (self.__meas_model).permute([1, 0]), torch.linalg.inv(s))

        state_update = state_pred + torch.bmm(k_gain, inna_info.permute([0, 2, 1])).permute([0, 2, 1])

        p_update = p_pred - torch.matmul(torch.matmul(k_gain, s), k_gain.permute([0, 2, 1]))

        '''
        Memory update
        '''
        # c_update = self.__acti_fun(self.__kernel_memory_l1(torch.cat((c_update, meas_input / 60, state_update / 60), dim=-1)))

        return state_update, (c_update, state_update, p_update, lstm_h_prev, lstm_c_prev)

    def name(self):
        return self.__class__.__name__
    