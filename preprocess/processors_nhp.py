# -*- coding: utf-8 -*-
"""
@author: hongyuan
"""

import torch
import numpy as np

def sampleForIntegral(input, sampling=1, device=None):
    r"""
    sampling dtimes in each interval given other tensors
    this function only deals with the same seq with possibly diff groups
    此函数仅处理可能带有差异组的相同序列
    so their duration and lens are the same
    """
    device = device or 'cpu'
    device = torch.device(device)
    event, feats, time, duration, lens = input

    num_groups, T_plus_2 = event.size()
    assert lens.max() + 2 == T_plus_2, "max len should match"
    #print("sampling for integral ")
    max_sampling = max( int( lens.max() * sampling ), 1 )

    sampled_times = torch.empty(size=[max_sampling], dtype=torch.float32,
                                device=device)
    sampled_times.uniform_(0.0, float(duration[0]))
    sampled_times = sampled_times.sort()[0].unsqueeze(0).expand(num_groups, max_sampling)

    dtime_sampling = torch.zeros(size=[num_groups, max_sampling],
                                 dtype=torch.float32, device=device)

    index_of_hidden_sampling = torch.zeros(size=[num_groups, max_sampling],
                                    dtype=torch.int64, device=device)
    #当前列之前的和加到当前列上====距离第一个事件的时间间隔
    cum_time = time.cumsum(dim=1)
    indices_mat = torch.arange(0, num_groups, dtype=torch.int64,
                               device=device).unsqueeze(1).expand(num_groups, max_sampling)
    indices_mat = (T_plus_2 - 1) * indices_mat
    # + device.LongTensor(range(T_plus_2-1)).unsqueeze(0)

    current_step = torch.zeros(size=[num_groups, max_sampling],
                               dtype=torch.int64, device=device)

    for j in range( lens.max() + 1 ):

        bench_cum_time = cum_time[:, j].unsqueeze(1).expand(
            num_groups, max_sampling)
        indices_to_edit = sampled_times > bench_cum_time
####如果大，就减，否则，
        dtime_sampling[indices_to_edit] = \
        (sampled_times - bench_cum_time)[indices_to_edit]

        current_step.fill_(j)
        index_of_hidden_sampling[indices_to_edit] = \
        (indices_mat + current_step)[indices_to_edit]

    assert dtime_sampling.min() >= 0.0, "Time >= 0"
##一个用户 对应了一个3维的dtime_sampling
    return event, feats, time, duration, dtime_sampling, index_of_hidden_sampling
    # idx of output :
    # event 0, feats 1, time 2, duration 3,
    # dtime_sampling 4, index_of_hidden_sampling 5

###统一event的大小，为全序列中最长的那个长度
def processBatchParticles(
    batch_of_seqs, idx_BOS, idx_EOS, idx_PAD, device=None):

    device = device or 'cpu'
    device = torch.device(device)

    batch_size = len(batch_of_seqs)
    num_groups = batch_of_seqs[0][2].size(0)
    feature_dim = batch_of_seqs[0][1].size(2)

    max_len = -1  #event最长的
    max_len_sampling = -1   #3

    for i_batch, seq_with_particles in enumerate(batch_of_seqs):
        seq_len = seq_with_particles[0].size(1)
        seq_len_sampling = seq_with_particles[4].size(1)
        max_len = seq_len if seq_len > max_len else max_len
        max_len_sampling = seq_len_sampling if seq_len_sampling > max_len_sampling else max_len_sampling

    duration = torch.zeros(size=[batch_size, num_groups], dtype=torch.float32, device=device)

    event = torch.empty(size=[batch_size, num_groups, max_len],
                        dtype=torch.int64, device=device).fill_(idx_PAD)
    feats = torch.zeros(size=[batch_size, num_groups, max_len, feature_dim],
                        dtype=torch.float32, device=device)
    time = torch.zeros(size=[batch_size, num_groups, max_len],
                       dtype=torch.float32, device=device)

    dtime_sampling = torch.zeros(size=[batch_size, num_groups, max_len_sampling],
                                 dtype=torch.float32, device=device)
    index_of_hidden_sampling = torch.zeros(size=[batch_size, num_groups, max_len_sampling],
                                           dtype=torch.int64, device=device)
    mask_sampling = torch.zeros(size=[batch_size, num_groups, max_len_sampling],
                                dtype=torch.float32, device=device)
    # note we use batch_size as 0-dim here
    # because we need to flatten num_groups and max_len_sampling
    # in forward method of nhp

    for i_batch, seq_with_particles in enumerate(batch_of_seqs):

        seq_len = seq_with_particles[0].size(1)
        seq_len_sampling = seq_with_particles[4].size(1)

        event[i_batch, :, :seq_len] = seq_with_particles[0].clone()
        feats[i_batch, :, :seq_len, :] = seq_with_particles[1].clone()
        time[i_batch, :, :seq_len] = seq_with_particles[2].clone()

        duration[i_batch, :] = seq_with_particles[3].clone()

        dtime_sampling[i_batch, :, :seq_len_sampling] = seq_with_particles[4].clone()
        mask_sampling[i_batch, :, :seq_len_sampling] = 1.0

        r"""
        since we now have an extra dimension i.e. batch_size
        we need to revise the index_of_hidden_sampling, that is,
        it should not be i_particle * (T+1) + j anymore
        what it should be ?
        consider when we flat the states, we make them to
        ( batch_size * num_groups * T+1 ) * hidden_dim
        and when we flatten the index_of_hidden_sampling, it is
        batch_size * num_groups * max_len_sampling
        so each entry should be :
        i_seq * ( num_groups * (T+1) ) + i_particle * (T+1) + j , that is
        for whatever value of element in this current matrix
        we should add it with i_seq * ( num_groups * (T+1) )
        this part is tricky so I should design sanity check
        """
        remainder = seq_with_particles[5] % ( seq_len - 1 )
        multiple = seq_with_particles[5] / ( seq_len - 1 )
        index_of_hidden_sampling[i_batch, :, :seq_len_sampling] = \
        i_batch * num_groups * (max_len - 1) + multiple * (max_len - 1) + remainder
    #import ipdb; ipdb.set_trace()

    return [
        event,
        feats,
        time,
        duration,
        dtime_sampling,
        index_of_hidden_sampling,
        mask_sampling
    ]


class DataProcessorBase(object):
    def __init__(self,
        idx_BOS, idx_EOS, idx_PAD,
        feature_dim, group_num, sampling=1, device=None):
        self.idx_BOS = idx_BOS
        self.idx_EOS = idx_EOS
        self.idx_PAD = idx_PAD
        self.feature_dim = feature_dim
        self.group_num = group_num
        self.sampling = sampling

        device = device or 'cpu'
        self.device = torch.device(device)

        self.funcBatch = processBatchParticles
        self.sampleForIntegral = sampleForIntegral

    def getRange(self, head, tail):
        return torch.arange(
            head, tail, dtype=torch.int64, device=self.device)

    def processSeq(self, seq, fix_group, group_id=None):
        """
        The process seq function is moved to the class.
        :param list seq:
        :param boolean fix_group:
        True -- process with given group ID
        False -- process for all possible groups
        :param group_id: given group ID
        """

        if fix_group:
            # if we only process seq with a fixed group
            if group_id is None:
                # if group id is not given, use ground-truth
                if 'group' in seq:
                    group_id = int(seq['group']-1)
                else:
                    group_id = 0
            n = 1
            assert group_id >=0 and group_id < self.group_num, "range error"
        else:
            n = self.group_num

        # including BOS and EOS
        len_seq = len(seq)+2
        event = torch.zeros(
            size=[n, len_seq], device=self.device, dtype=torch.int64)
        event[:, -1] = self.idx_EOS
        if fix_group:
            event[0, 0] = self.idx_BOS[group_id]
        else:
            for i in range(self.group_num):
                event[i, 0] = self.idx_BOS[i]

        feats = torch.zeros(
            size=[n, len_seq, self.feature_dim],
            device=self.device, dtype=torch.float32)

        dtime = torch.zeros(
            size=[n, len_seq, ], device=self.device, dtype=torch.float32)

        for token_idx, token in enumerate(seq):
            event[:, token_idx+1] = int(token['type_event'])
            
            dtime[:, token_idx+1] = float(token['time_since_last_event'])

        time_stamps = dtime[0, :].cumsum(dim=0)
        duration = torch.empty(
            size=[n], dtype=torch.float32, device=self.device).fill_(time_stamps[-1])
        lens = torch.empty(size=[n], dtype=torch.int64, device=self.device).fill_(len_seq-2)  #序列长度-2 没有开始和结束的

        return [event, # 0: n * len_seq
                feats, # 1: n * len_seq
                dtime, # 2: n * len_seq
                duration, # 3: n
                lens # 4: n # total len without BOS or EOS
                ]

    def processBatchParticles(self, input):
        return self.funcBatch(input,
            idx_BOS=self.idx_BOS, idx_EOS=self.idx_EOS, idx_PAD=self.idx_PAD,
            device=self.device)

    #@profile
    """
    should really change the function name(s)
    """
    def processBatchSeqsWithParticles(self, input):
        """
        batch of seqs, where each seq is many particles (as torch tensors)
        对于输入的所有数据，一条一条（一个用户）的处理，
        """
        batch_of_seqs = []
        for seq in input:
            batch_of_seqs.append(self.sampleForIntegral(
                seq, sampling=self.sampling, device=self.device) )
        return self.processBatchParticles(batch_of_seqs)


class DataProcessorNeuralHawkes(DataProcessorBase):
    def __init__(self, *args, **kwargs):
        super(DataProcessorNeuralHawkes, self).__init__(*args, **kwargs)

class DataProcessorNaive(DataProcessorBase):
    def __init__(self, *args, **kwargs):
        super(DataProcessorNaive, self).__init__('Naive', *args, **kwargs)


class LogWriter(object):

    def __init__(self, path, args):
        self.path = path
        self.args = args
        with open(self.path, 'w') as f:
            f.write("Training Log\n")
            f.write("Hyperparameters\n")
            for argname in self.args:
                f.write("{} : {}\n".format(argname, self.args[argname]))
            f.write("Checkpoints:\n")

    def checkpoint(self, to_write):
        with open(self.path, 'a') as f:
            f.write(to_write+'\n')

class LogReader(object):

    def __init__(self, path):
        self.path = path
        with open(self.path, 'r') as f:
            self.doc = f.read()

    def isfloat(self, str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    def casttype(self, str):
        res = None
        if str.isdigit():
            res = int(str)
        elif self.isfloat(str):
            res = float(str)
        elif str == 'True' or str == 'False':
            res = True if str == 'True' else False
        else:
            res = str
        return res

    def getArgs(self):
        block_args = self.doc.split('Hyperparameters\n')[-1]
        block_args = block_args.split('Checkpoints:\n')[0]
        lines_args = block_args.split('\n')
        res = {}
        for line in lines_args:
            items = line.split(' : ')
            res[items[0]] = self.casttype(items[-1])
        return res

if __name__ == '__main__':
    test_miss()



