# -*- coding: utf-8 -*-
"""
@author: hongyuan
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from cont_time_cell import CTLSTMCell

from torch.distributions.multinomial import Multinomial
class NeuralHawkes(nn.Module):

    def __init__(self, *,
        event_num, group_num,
        hidden_dim=32, beta=1.0, device=None):
        super(NeuralHawkes, self).__init__()

        self.event_num = event_num
        self.group_num = group_num
        self.hidden_dim = hidden_dim

        self.idx_BOS = {}
        for i in range(self.group_num):
            self.idx_BOS[i] = self.event_num + i
        self.idx_EOS = self.event_num + self.group_num
        self.idx_PAD = self.event_num + self.group_num + 1

        self.beta = beta

        device = device or 'cpu'
        self.device = torch.device(device)

        self.Emb = nn.Embedding(
            self.event_num + self.group_num + 2, self.hidden_dim)

        self.rnn_cell = CTLSTMCell(
            self.hidden_dim, beta=self.beta, device=self.device)

        self.hidden_lambda = nn.Linear(
            self.hidden_dim, self.event_num, bias=False)

        self.init_h = torch.zeros(size=[hidden_dim],
                                  dtype=torch.float32, device=self.device)
        self.init_c = torch.zeros(size=[hidden_dim],
                                  dtype=torch.float32, device=self.device)
        self.init_cb = torch.zeros(size=[hidden_dim],
                                   dtype=torch.float32, device=self.device)

        self.eps = np.finfo(float).eps
        self.max = np.finfo(float).max

    def cuda(self, device=None):
        device = device or 'cuda:0'
        self.device = torch.device(device)
        assert self.device.type == 'cuda'
        super().cuda(self.device)

    def cpu(self):
        self.device = torch.device('cpu')
        super().cuda(self.device)

    def getStates(self, event, dtime):
        batch_size, num_groups, T_plus_2 = event.size()
        cell_t_i_minus = self.init_c.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_groups, self.hidden_dim)
        cell_bar_im1 = self.init_cb.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_groups, self.hidden_dim)
        hidden_t_i_minus = self.init_h.unsqueeze(0).unsqueeze(0).expand(
            batch_size, num_groups, self.hidden_dim)

        all_cell, all_cell_bar = [], []
        all_gate_output, all_gate_decay = [], []
        all_hidden = []
        all_hidden_after_update = []
#把event一个一个放入
        for i in range(T_plus_2 - 1):
            # only BOS to last event update LSTM
            # <s> CT-LSTM
#256,1,16
            emb_i = self.Emb(event[:, :, i ])
            dtime_i = dtime[:, :, i + 1 ] # need to carefully check here

            cell_i, cell_bar_i, gate_decay_i, gate_output_i = self.rnn_cell(
                emb_i, hidden_t_i_minus, cell_t_i_minus, cell_bar_im1
            )
            #执行公式7
            _, hidden_t_i_plus = self.rnn_cell.decay(
                cell_i, cell_bar_i, gate_decay_i, gate_output_i,
                torch.zeros_like(dtime_i)
            )
            cell_t_ip1_minus, hidden_t_ip1_minus = self.rnn_cell.decay(
                cell_i, cell_bar_i, gate_decay_i, gate_output_i,
                dtime_i
            )
            all_cell.append(cell_i)
            all_cell_bar.append(cell_bar_i)
            all_gate_decay.append(gate_decay_i)
            all_gate_output.append(gate_output_i)
            all_hidden.append(hidden_t_ip1_minus)
            all_hidden_after_update.append(hidden_t_i_plus)
            cell_t_i_minus = cell_t_ip1_minus
            cell_bar_im1 = cell_bar_i
            hidden_t_i_minus = hidden_t_ip1_minus
            # </s> CT-LSTM
        # these tensors shape : batch_size, num_groups, T+1, hidden_dim
        # cells and gates right after BOS, 1st event, ..., N-th event
        # hidden right before 1st event, ..., N-th event, End event (PAD)
        all_cell = torch.stack( all_cell, dim=2)
        all_cell_bar = torch.stack( all_cell_bar, dim=2)
        all_gate_decay = torch.stack( all_gate_decay, dim=2)
        all_gate_output = torch.stack( all_gate_output, dim=2)
        all_hidden = torch.stack( all_hidden, dim=2 )
        all_hidden_after_update = torch.stack( all_hidden_after_update, dim=2)
        #assert all_gate_decay.data.cpu().numpy().all() >= 0.0, "Decay > 0"
        return batch_size, num_groups, T_plus_2, \
        all_cell, all_cell_bar, all_gate_decay, all_gate_output, \
        all_hidden, all_hidden_after_update

    def getTarget(self, event, dtime):
        r"""
        make target variable and masks
        """
        batch_size, num_groups, T_plus_2 = event.size()
        mask_complete = torch.ones_like(dtime[:, :, 1:])
        target_data = event[:, :, 1:].detach().data.clone()

        mask_complete[target_data >= self.event_num] = 0.0
        target_data[target_data >= self.event_num] = 0 # PAD to be 0
        target = target_data
        return target, mask_complete

    def getSampledStates(
        self, dtime_sampling, index_of_hidden_sampling,
        all_cell, all_cell_bar, all_gate_output, all_gate_decay):
        r"""
        we output the sampled hidden states of the left-to-right machine
        states shape : batch_size * num_groups * T+1 * hidden_dim
        dtime_sampling : batch_size * num_groups * max_len_sampling
        index_of_hidden_sampling : batch_size * num_groups * max_len_sampling
        """
        batch_size, num_groups, T_plus_1, _ = all_cell.size()   #256 1 9 16   ##256 1 200 16
        _, _, max_len_sampling = dtime_sampling.size()     #256  1 8  ##256 1 199
        # a=all_cell.view(
        #     batch_size * num_groups * T_plus_1, self.hidden_dim)
        # b=index_of_hidden_sampling.view(-1)
        # c=a[:2048,:].view(
        #             batch_size, num_groups, max_len_sampling, self.hidden_dim)
        #hidden_num=index_of_hidden_sampling.view(-1).size()
        hidden_num=256*max_len_sampling
        '''
        all_cell_sampling = all_cell.view(
            batch_size * num_groups * T_plus_1, self.hidden_dim )[
                index_of_hidden_sampling.view(-1), :].view(
                    batch_size, num_groups, max_len_sampling, self.hidden_dim)
    '''
        all_cell_sampling = all_cell.view(
            batch_size * num_groups * T_plus_1, self.hidden_dim)[
                            :hidden_num, :].view(
            batch_size, num_groups, max_len_sampling, self.hidden_dim)
        all_cell_bar_sampling = all_cell_bar.view(
            batch_size * num_groups * T_plus_1, self.hidden_dim )[
                :hidden_num, :].view(
                    batch_size, num_groups, max_len_sampling, self.hidden_dim)
        all_gate_output_sampling = all_gate_output.view(
            batch_size * num_groups * T_plus_1, self.hidden_dim )[
                :hidden_num, :].view(
                    batch_size, num_groups, max_len_sampling, self.hidden_dim)
        all_gate_decay_sampling = all_gate_decay.view(
            batch_size * num_groups * T_plus_1, self.hidden_dim )[
                :hidden_num, :].view(
                    batch_size, num_groups, max_len_sampling, self.hidden_dim)
        #执行公式7、4b
        cy_sample, hy_sample = self.rnn_cell.decay(
            all_cell_sampling, all_cell_bar_sampling,
            all_gate_decay_sampling, all_gate_output_sampling,
            dtime_sampling
        )

        return hy_sample

    def getLambda(
        self, batch_size, num_groups, T_plus_2,
        target, mask_complete, all_hidden, sampled_hidden ):
        #print("type is {}".format(type(all_hidden) ) )

        all_lambda= F.softplus(self.hidden_lambda(all_hidden), beta=self.beta)
        log_lambda= torch.log(all_lambda+ self.eps)

        #print("batchsize {}, num_groups {}, T_plus_2 {}".format(
        #    batch_size, num_groups, T_plus_2))

        log_lambda_target = log_lambda.view(
            batch_size * num_groups * (T_plus_2 - 1), self.event_num
        )[
            torch.arange(0, batch_size * num_groups * (T_plus_2 - 1),
                         dtype=torch.int64, device=self.device),
            target.view( batch_size * num_groups * (T_plus_2 - 1) )
        ].view(batch_size, num_groups, T_plus_2 - 1)

        log_lambda_target_complete = log_lambda_target * mask_complete

        lambda_sum_complete = torch.sum(all_lambda, dim=3)

        log_lambda_sum_complete = torch.log(lambda_sum_complete + self.eps)
        log_lambda_sum_complete *= mask_complete

        all_lambda_sample = F.softplus(
            self.hidden_lambda(sampled_hidden), beta=self.beta )

        return log_lambda_target_complete, all_lambda_sample

    def getIntegral(
        self, lambda_sample, mask_sampling, duration ):
        r"""
        mask_sampling : batch_size * num_groups * max_len_sampling
        duration : batch_size * num_groups
        """
        lambda_sample_sum = lambda_sample.sum(dim=3)
        lambda_sample_mean = torch.sum(
            lambda_sample_sum * mask_sampling, dim=2 ) / torch.sum(
            mask_sampling, dim=2 )
        integral = lambda_sample_mean * duration

        return integral

    def forward(self, input):

        event, feats, dtime, duration, \
        dtime_sampling, index_of_hidden_sampling, mask_sampling = input

        r"""
        event, dtime : batch_size, M, T+2
        post(erior of incomplete unobserved) : batch_size, M (not used)
        duration : batch_size, M
        dtime_sampling : batch_size, M, T_sample
        """
        #核心LSTM计算部分
        batch_size, num_groups, T_plus_2, \
        all_cell, all_cell_bar, all_gate_decay, all_gate_output, \
        all_hidden, all_hidden_after_update = self.getStates(event, dtime)

        assert num_groups == 1, "more than one group?"
        #遮罩
        target, mask_complete = self.getTarget( event, dtime )
        #公式4b
        sampled_hidden = self.getSampledStates(
            dtime_sampling, index_of_hidden_sampling,
            all_cell, all_cell_bar, all_gate_output, all_gate_decay
        )
        #公式4a
        # <s> \lambda_{k_i}(t_i | H_i) for events
        log_lambda_target_complete, all_lambda_sample = self.getLambda(
            batch_size, num_groups, T_plus_2,
            target, mask_complete, all_hidden, sampled_hidden )
        # batch_size * num_groups * T_plus_2-1
        # </s> \lambda_{k_i}(t_i) for events

        # <s> int_{0}^{T} lambda_sum dt for events
        integral_of_lambda_complete = self.getIntegral(
            all_lambda_sample, mask_sampling, duration )

        # batch_size * num_groups
        # </s> int_{0}^{T} lambda_sum dt for events

        # <s> log likelihood computation                                                                                `
        logP_complete = log_lambda_target_complete.sum(2) - integral_of_lambda_complete
        # log p --- according to (trained) neural Hawkes process

        # </s> log likelihood computation

        # complete log likelihood
        objective = -torch.sum( logP_complete )
        num_events = torch.sum( mask_complete )


        #return objective, num_events
        #计算时序预测值
        return objective,num_events,all_lambda_sample
