# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
@author: hongyuan
"""

import pickle
import time
#from time import time
import numpy
import random
import os
import datetime

import torch
import torch.optim as optim

import nhp
from DataProcess import processors_nhp
from util import *

#
import argparse

Lamda_file='lamda_taobao_0.001'

def run_complete(args):

    random.seed(args['Seed'])
    numpy.random.seed(args['Seed'])
    torch.manual_seed(args['Seed'])

    with open(os.path.join(args['PathData'], 'train.pkl'), 'rb') as f:
        pkl_train = pickle.load(f, encoding ='latin1')
    with open(os.path.join(args['PathData'], 'dev.pkl'), 'rb') as f:
        pkl_dev = pickle.load(f, encoding ='latin1')

    learning_rate = args['LearnRate']

    data = pkl_train['train']
    data_dev = pkl_dev['dev']

    total_event_num = int(pkl_train['dim_process'])   #75

    hidden_dim = args['DimLSTM']   #16

    agent = nhp.NeuralHawkes(
        event_num=total_event_num, group_num=args['NumGroup'],
        hidden_dim=hidden_dim,
        device='cuda' if args['UseGPU'] else 'cpu'
    )

    if args['UseGPU']:
        agent.cuda()

    sampling = 1

    proc = processors_nhp.DataProcessorNeuralHawkes(
        idx_BOS=agent.idx_BOS,  #{0:75}
        idx_EOS=agent.idx_EOS,  #76
        idx_PAD=agent.idx_PAD, #
        feature_dim=1,
        group_num=args['NumGroup'],
        sampling=sampling,
        device = 'cuda' if args['UseGPU'] else 'cpu'
    )
    logger = processors_nhp.LogWriter(args['PathLog'], args)

    optimizer = optim.Adam(
        agent.parameters(), lr=learning_rate
    )
    optimizer.zero_grad()

    print("Start training ... ")
    total_logP_best = -1e6
    avg_dis_best = 1e6
    episode_best = -1
    #time0 = time.time()
    time0 = time()

    episodes = []
    total_rewards = []

    max_episode = args['MaxEpoch'] * len(data)
    report_gap = args['TrackPeriod']

    time_sample = 0.0
    time_train_only = 0.0
    time_dev_only = 0.0
    input = []
    lambda_save = []

    for episode in range(max_episode):

        idx_seq = episode % len(data)
        idx_epoch = episode // len(data)
        one_seq = data[ idx_seq ]
        #[{'time_since_start': 0.0, 'time_since_last_event': 0.0, 'type_event': 1},{},{}]  表示一条序列点

        #time_sample_0 = time.time()
        #给编码
        input.append( proc.processSeq( one_seq, fix_group=False ) )
        #time_sample += (time.time() - time_sample_0)

        if len(input) >= args['SizeBatch']:
            batchdata_seqs = proc.processBatchSeqsWithParticles(input)
            #weight, _ = agent(batchdata_seqs, mode=4)
            agent.train()
            #time_train_only_0 = time.time()
            time_train_only_0 = time()
            #objective, _ = agent(
            #    batchdata_seqs, mode=1,
            #    weight=Variable(weight.data.clone() ) )
            #objective, _ = agent( batchdata_seqs )
            objective, _ ,all_lambda_sample= agent(batchdata_seqs)
            objective.backward()
            #torch.nn.utils.clip_grad_norm(agent.parameters(), 0.25)
            optimizer.step()
            optimizer.zero_grad()
            time_train_only += (time.time() - time_train_only_0)


            # 读取时序值到文件
            all_lambda_sample = all_lambda_sample.detach().numpy()
            lambda_save.append(all_lambda_sample)

            input = []

            if episode % report_gap == report_gap - 1:

                #time1 = time.time()
                time1 = time()
                time_train = time1 - time0
                time0 = time1

                print("Validating at episode {} ({}-th seq of {}-th epoch)".format(
                    episode, idx_seq, idx_epoch))
                total_logP = 0.0
                total_num_token = 0.0

                #total_dis = 0.0
                #total_dis = torch.FloatTensor(args['NumPenalty']).fill_(0.0)
                #editdis.reset()

                input_dev = []
                agent.eval()
                #print("scale now is : {}".format(agent.scale.data))

                for i_dev, one_seq_dev in enumerate(data_dev):

                    input_dev.append(
                        proc.processSeq( one_seq_dev, fix_group=False ) )

                    if (i_dev+1) % args['SizeBatch'] == 0 or \
                            (i_dev == len(data_dev)-1 and (len(input_dev)%args['SizeBatch']) > 0):

                        batchdata_seqs_dev = proc.processBatchSeqsWithParticles(
                            input_dev )

                        time_dev_only_0 = time()
                        objective_dev, num_events_dev,all_lambda_sample= agent(
                            batchdata_seqs_dev )
                        time_dev_only = time() - time_dev_only_0

                        total_logP -= float( objective_dev.data.sum() )

                        total_num_token += float(
                            num_events_dev.data.sum() / ( args['NumGroup'] * 1.0 ) )

                        input_dev = []

                total_logP /= total_num_token

                message = "Episode {} ({}-th seq of {}-th epoch), loglik is {:.4f}".format(
                    episode, idx_seq, idx_epoch, total_logP )
                logger.checkpoint(message)
                print(message)

                updated = None
                if total_logP > total_logP_best:
                    total_logP_best = total_logP
                    updated = True
                    episode_best = episode
                else:
                    updated = False
                message = "Current best loglik is {:.4f} (updated at episode {})".format(
                    total_logP_best, episode_best )

                if updated:
                    message += ", best updated at this episode"
                    torch.save(
                        agent.state_dict(), args['PathSave'])
                logger.checkpoint(message)
                #print("After episode {}, agent with sampling {} reaches log-likelihood of {}, with current best {}".format( episode, sampling, round(total_logP, 4), round(total_logP_best, 4) ) )
                print(message)
                episodes.append(episode)

                #if args['WhatTrack'] == 'lik':
                #    total_rewards.append(round(total_logP, 4) )
                #elif args['WhatTrack'] == 'ed':
                #    total_rewards.append(round(avg_dis, 4) )
                #else:
                #    raise Exception(
                #        "WhatTrack {} not defined".format(args['WhatTrack']))

                time1 = time()
                time_dev = time1 - time0
                time0 = time1
                message = "time to train {} episodes is {:.2f} and time for dev is {:.2f}".format(
                    report_gap, time_train, time_dev )
                #
                #message += ", sampling in training takes {} and training only takes {}".format(
                #    round(time_sample, 2), round(time_train_only, 2) )
                #message += ", sampling in dev takes {} and dev only takes {}".format(
                #    round(time_sample, 2), round(time_dev_only, 2) )
                time_sample, time_train_only = 0.0, 0.0
                time_dev_only = 0.0
                #
                logger.checkpoint(message)
                print(message)
    message = "training finished"
    logger.checkpoint(message)
    saveVariableOnDisk(lambda_save, Lamda_file)
    print(message)
    #return episodes, total_rewards


def main():

    parser = argparse.ArgumentParser(description='Trainning model ...')
    #parser.add_argument(
    #    '-m', '--Model', required=True,
    #    choices=[ 'nh', 'nhpf', 'nhps' ],
    #    help='what model to use?'
    #)
    parser.add_argument(
        '-ds', '--Dataset', type=str,# required=True,
        help='e.g. pilothawkes',
        default='nhp_data'
    )
    parser.add_argument(
        '-rp', '--RootPath', type=str,
        help='Root path of project',
        default='../'
    )
    #parser.add_argument(
    #    '-pm', '--PathModel', default='',
    #    help='Path of saved neural Hawkes? Only used for nhps'
    #)
    parser.add_argument(
        '-ng', '--NumGroup', default=1, type=int,
        help='Number of groups'
    )
    parser.add_argument(
        '-d', '--DimLSTM', default=16, type=int,
        help='Dimension of LSTM?'
    )
    parser.add_argument(
        '-sb', '--SizeBatch', default=50, type=int,
        help='Size of mini-batch'
    )
    parser.add_argument(
        '-tp', '--TrackPeriod', default=5000, type=int,
        help='How many sequences before every checkpoint?'
    )
    parser.add_argument(
        '-me', '--MaxEpoch', default=50, type=int,
        help='Max epoch number of training'
    )
    parser.add_argument(
        '-lr', '--LearnRate', default=1e-3, type=float,
        help='What is the (starting) learning rate?'
    )
    parser.add_argument(
        '-gpu', '--UseGPU', action='store_true', #default=0, type=int, choices=[0,1],
        help='Use GPU?'
    )
    parser.add_argument(
        '-sd', '--Seed', default=12345, type=int,
        help='Random seed. e.g. 12345'
    )

    args = parser.parse_args()
    dict_args = vars(args)
    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()

    r"""
    make tag_model with arguments
    """
    #use_in_foldername = [
    #    'DimLSTM'
    #]
    # think about modiying this
    # we definitely do not want to keep all of these
    #tag_model =
    # for k in dict_args:
    #     v = dict_args[k]
    #     if k in use_in_foldername:
    #         tag_model += '_{}={}'.format(k, str(v))

    root_path = os.path.abspath(dict_args['RootPath'])
    dict_args['PathData'] = os.path.join(root_path, 'data', dict_args['Dataset'])
    dict_args['Version'] = torch.__version__
    dict_args['ID'] = id_process
    dict_args['TIME'] = time_current

    # format: [arg name, name used in path]
    args_used_in_name = [
        ['DimLSTM', 'dim'],
        ['SizeBatch', 'batch'],
        ['Seed', 'seed'],
        ['LearnRate', 'lr'],
    ]
    folder_name = list()
    for arg_name, rename in args_used_in_name:
        folder_name.append('{}-{}'.format(rename, dict_args[arg_name]))
    folder_name = '_'.join(folder_name)
    folder_name = '{}_{}'.format(folder_name, id_process)
    print(folder_name)

    path_log = os.path.join(root_path, 'logs', dict_args['Dataset'], folder_name)
    os.makedirs(path_log)

    file_log = os.path.join(path_log, 'log.txt')
    file_model = os.path.join(path_log, 'saved_model')

    dict_args['PathLog'] = file_log
    dict_args['PathSave'] = file_model
    dict_args['Model'] = 'nhp'

    if '' in dict_args:
        del dict_args['']

    run_complete(dict_args)


if __name__ == "__main__": main()
