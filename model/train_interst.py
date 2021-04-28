# coding:utf-8
import numpy as np
import time
import random
import os
import datetime
import sys
from util import *
from cal_dpp import *
import multiprocessing
from tensorboardX import SummaryWriter
import argparse

import torch
import torch.optim as optim

from data_iterator import DataIterator
from model_interest import *
from model_gru import *



#
parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--model_type', type=str, default='MIMN', help='DIEN | MIMN | ..')  # 训练mimn
parser.add_argument('-dataset', default='book', type=str,help='Max epoch number of training')
parser.add_argument('--memory_size', type=int, default=4)
parser.add_argument('--topN', type=int, default=20, help='item that generate')
parser.add_argument('--cate_file', type=str, default='../data/taobao_data/taobao_item_cate.txt',help='item--cate')
parser.add_argument('-sd', '--Seed', default=12345, type=int,help='Random seed. e.g. 12345')
parser.add_argument('-lr', '--LearnRate', default=1e-3, type=float,help='What is the (starting) learning rate?')  #0.001
parser.add_argument('-gpu', '--UseGPU', action='store_true', help='Use GPU?',default=1) # default=0, type=int, choices=[0,1],
parser.add_argument('-me', '--MaxEpoch', default=1, type=int,help='Max epoch number of training')
parser.add_argument("-sigma", type=float, default=-1, help="init weight -1: range [-sigma, sigma], -2: range [0, sigma]") # weight initialization [-sigma sigma] in literature
# nhp


EMBEDDING_DIM = 32
HIDDEN_SIZE = 32
best_auc = 0.0


def init_model(model):
    for p in model.parameters():
        if args.sigma != -1 and args.sigma != -2:
            sigma = args.sigma
            p.data.uniform_(-sigma, sigma)
        elif len(list(p.size())) > 1:
            sigma = np.sqrt(6.0 / (p.size(0) + p.size(1)))
            if args.sigma == -1:
                p.data.uniform_(-sigma, sigma)
            else:
                p.data.uniform_(0, sigma)


def prepare_data(src, target, train_flag=1):
    nick_id, item_id = src
    # 加入时间信息以及target items
    hist_item_list, hist_cate_list, hist_mask, neg_item_list = target
    return nick_id, item_id, hist_item_list, hist_cate_list, hist_mask, neg_item_list


def train(
        args,
        train_file="../data/taobao_data/taobao_train.txt",
        valid_file="../data/_data/taobao_valid.txt",
        item_count=1,
        batch_size=128,  ##256
        maxlen=200,
        test_iter=100,  ##100
        save_iter=100,
):
    np.random.seed(args['Seed'])
    torch.manual_seed(args['Seed'])
    Memory_Size = args['memory_size']
    #item_cate_map = load_item_cate(args['cate_file'])
    lr = args['LearnRate']
    max_episode = args['MaxEpoch']
    best_metric = 0.0
    writer = SummaryWriter()


    train_data = DataIterator(train_file, batch_size, maxlen, train_flag=1)
    valid_data = DataIterator(valid_file, batch_size, maxlen, train_flag=0)

    BATCH_SIZE = batch_size
    SEQ_LEN = maxlen


    model = GRURec(item_count, EMBEDDING_DIM, HIDDEN_SIZE, batch_size, maxlen, Memory_Size,
                   device='cuda' if args['UseGPU'] else 'cpu')
    # model = GRU(item_count, EMBEDDING_DIM, HIDDEN_SIZE, batch_size, maxlen, Memory_Size,
    #                device='cuda' if args['UseGPU'] else 'cpu')
    init_model(model)

    if args['UseGPU']:
        model.cuda()

    iter=0
    optimizer = optim.Adam(
        model.parameters(), lr=lr
    )
    #print(model.parameters())
    optimizer.zero_grad()
    print('training begin')
    # sys.stdout.flush()
    time_train_only = 0.0

    ##训练MIMN
    for itr in range(max_episode):
        # 利用多线程调出数据，一次是Batchsize大小
        print("epoch" + str(itr))
        loss_sum = 0.0
        accuracy_sum = 0.
        aux_loss_sum = 0.
        hidden=model.init_hidden()
        for src, tgt in train_data:
            ####加入时间戳信息 ,有batch_size条
            data_iter= prepare_data(src, tgt)
            model.train()
           # 训练
            loss= model(list(data_iter))
            loss.backward()
            optimizer.step()
            #检查梯度是否计算
            # for name, parms in model.named_parameters():
            #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad, ' -->grad_value:', parms.grad)
            optimizer.zero_grad()
            loss_sum += loss
            sys.stdout.flush()
            iter=iter+1
            #print('iter: %d ----> train_loss: %.4f' % (iter, loss))
            # # 加载valid集，调用DPP，测试评估
            # if (iter % test_iter) == 0:

            if (iter % test_iter) == 0:
                print('iter: %d ----> train_loss: %.4f' % (iter, loss_sum / test_iter))
                total = 0
                total_recall = 0.0
                total_ndcg = 0.0
                total_hitrate = 0
                total_diversity = 0.0

                print("start eval model")

                for src, tgt in valid_data:
                    nick_id, item_id, hist_item_list, hist_cate_list, hist_mask, neg_item_list = prepare_data(src, tgt)
                    target_item = item_id
                    new_item_id = [i[0] for i in item_id]
                    model.eval()
                    loss_valid = model([nick_id, new_item_id, hist_item_list, hist_cate_list, hist_mask, neg_item_list])
                    dpp_item = cal_dpp(model, EMBEDDING_DIM, BATCH_SIZE, args['topN'],Memory_Size)
                    item_cate_map=[] #不做多样性
                    total_recall, total_ndcg, total_hitrate, total_diversity = evaluate(target_item, dpp_item,item_cate_map, total_recall,total_ndcg, total_hitrate,total_diversity,save=False)
                    total += BATCH_SIZE

                recall = total_recall / total
                ndcg = total_ndcg / total
                hitrate = total_hitrate * 1.0 / total
                diversity = total_diversity * 1.0 / total

                ##打印结果
                metrics = {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}
                log_str = 'iter: %d, train loss: %.4f' % (iter, loss_sum / test_iter)
                log_str += ', ' + ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
                print(log_str)
                # writer.add_scalar('train/loss', loss_sum / test_iter, iter)
                # for key, value in metrics.items():
                #     writer.add_scalar('eval/' + key, value, iter)
                #
                # if recall > best_metric:
                #     best_metric = recall
                #     saveVariableOnDisk(best_metric, 'metric')
                #     # torch.save(model, best_model_path) #保存，path是pkl
                #     # model=torch.load(best_model_path)  #加载

                loss_sum = 0.0
                accuracy_sum = 0.0
                aux_loss_sum = 0.0
            if iter > 400000:
                break
    print("end train")


def reload(model):
    print('===> Try resume from checkpoint')
    if os.path.isdir('checkpoint'):
        try:
            checkpoint = torch.load('./checkpoint/autoencoder.t7')
            model.load_state_dict(checkpoint['state'])  # 从字典中依次读取
            start_epoch = checkpoint['epoch']
            print('===> Load last checkpoint data')
        except FileNotFoundError:
            print('Can\'t found autoencoder.t7')
    else:
        start_epoch = 0
        print('===> Start from scratch')



if __name__ == '__main__':
    args = parser.parse_args()
    # mimn
    SEED = args.random_seed
    Model_Type = args.model_type
    Memory_Size = args.memory_size


    if args.dataset == 'taobao':
        path = '../data/taobao_data/'
        #item_count = 1708531
        batch_size = 256
        maxlen = 50
    elif args.dataset == 'book':
        path = '../data/book_data/'
        #item_count = 367983
        #item_count=18396
        batch_size = 128
        maxlen = 20

    train_file = path + args.dataset + '_train.txt'
    valid_file = path + args.dataset + '_valid.txt'
    test_file = path + args.dataset + '_test.txt'
    feature_file = args.dataset + '_data/' + args.dataset + '_feature'
    item_count,cate_count=restoreVariableFromDisk(feature_file)

    dict_args = vars(args)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train(args=dict_args, train_file=train_file,valid_file=valid_file,item_count=item_count,batch_size=batch_size,maxlen=maxlen)



