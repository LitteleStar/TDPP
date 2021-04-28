# coding:utf-8
import os
import datetime
import sys
from cal_dpp import *
from tensorboardX import SummaryWriter
import argparse

import torch.optim as optim

from model.data_iterator import DataIterator
from model_mimn import *
import nhp
from DataProcess import processors_nhp

#mimn
parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--model_type', type=str, default='MIMN', help='DIEN | MIMN | ..')  # 训练mimn
parser.add_argument('--memory_size', type=int, default=4)
parser.add_argument('--mem_induction', type=int, default=0, help='0:false|1:true')
parser.add_argument('--util_reg', type=int, default=0, help='0:false|1:true')  #没用到
parser.add_argument('--topN', type=int, default=30, help='item that generate')
parser.add_argument('--cate_file', type=str, default='../data/taobao_data/taobao_item_cate.txt', help='item cate')

#nhp
parser.add_argument('-ds', '--Dataset', type=str, help='e.g. pilothawkes',default='nhp_data')
parser.add_argument('-rp', '--RootPath', type=str,help='Root path of project', default='../')
parser.add_argument('-ng', '--NumGroup', default=1, type=int,help='Number of groups')   ####定义len大小为1
parser.add_argument('-d', '--DimLSTM', default=16, type=int,help='Dimension of LSTM?')
parser.add_argument('-sb', '--SizeBatch', default=50, type=int,help='Size of mini-batch')
parser.add_argument('-tp', '--TrackPeriod', default=5000, type=int,help='How many sequences before every checkpoint?')
parser.add_argument('-me', '--MaxEpoch', default=10, type=int,help='Max epoch number of training')
parser.add_argument('-lr', '--LearnRate', default=1e-3, type=float,help='What is the (starting) learning rate?')
parser.add_argument('-gpu', '--UseGPU', action='store_true', help='Use GPU?',default=1) # default=0, type=int, choices=[0,1],
parser.add_argument('-sd', '--Seed', default=12345, type=int,help='Random seed. e.g. 12345')


EMBEDDING_DIM = 16
HIDDEN_SIZE = 16 * 2
best_auc = 0.0

def prepare_data(src, target,train_flag=1):
    nick_id, item_id, cate_id = src
     #加入时间信息以及valid和test增加target
    if train_flag==0:
        label, hist_item, hist_cate, neg_item, neg_cate, hist_mask, time_list_array,target_item = target
        return nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask,time_list_array,target_item
    label, hist_item, hist_cate, neg_item, neg_cate, hist_mask, time_list_array = target
    return nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask,time_list_array


def train(
        args,
        train_file="../data/taobao_data/taobao_train.txt",
        valid_file="../data/taobao_data/taobao_valid.txt",
        feature_file="../data/taobao_data/taobao_feature.pkl",
        batch_size=256,
        maxlen=200,
        test_iter=100,  ##100
        save_iter=100,
        model_type='DNN',
        Memory_Size=4,
        Mem_Induction=0,
        Util_Reg=0
):
    np.random.seed(args['Seed'])
    np.random.seed(args['Seed'])
    torch.manual_seed(args['Seed'])
    Memory_Size=args['memory_size']
    item_cate_map = load_item_cate(args['cate_file'])
    best_metric=0.0
    writer=SummaryWriter()


    if model_type != "MIMN" or model_type != "MIMN_with_aux":
        model_path = "dnn_save_path/book_ckpt_noshuff" + model_type
        best_model_path = "dnn_best_model/book_ckpt_noshuff" + model_type
    else:
        model_path = "dnn_save_path/book_ckpt_noshuff" + model_type + str(Memory_Size) + str(Mem_Induction)
        best_model_path = "dnn_best_model/book_ckpt_noshuff" + model_type + str(Memory_Size) + str(Mem_Induction)
    iter = 0
    lr = args['LearnRate']

    train_data = DataIterator(train_file, batch_size, maxlen)
    valid_data = DataIterator(valid_file, batch_size, maxlen,train_flag=0)

    # with open(feature_file, 'rb') as handle:
    #     feature_num = pickle.load(handle, encoding='iso-8859-1')

    n_iid, n_cid =restoreVariableFromDisk('/taobao_data/taobao_feature')
    #n_iid, n_cid=4000000,4000000
    BATCH_SIZE = batch_size
    SEQ_LEN = maxlen

    model = MIMN(n_iid, n_cid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, Memory_Size,device='cuda' if args['UseGPU'] else 'cpu')
###nhp
    total_event_num = 4  #总事件数----应该是兴趣的总数
    hidden_dim = args['DimLSTM']  # 16
    agent = nhp.NeuralHawkes(event_num=total_event_num, group_num=args['NumGroup'],hidden_dim=hidden_dim,device='cuda' if args['UseGPU'] else 'cpu')
    if args['UseGPU']:
        agent.cuda()
        model.cuda()
    sampling = 1

    proc = processors_nhp.DataProcessorNeuralHawkes(
        idx_BOS=agent.idx_BOS,  #{0:x}
        idx_EOS=agent.idx_EOS,  #
        idx_PAD=agent.idx_PAD, #
        feature_dim=1,
        group_num=args['NumGroup'],
        sampling=sampling,
        device = 'cuda' if args['UseGPU'] else 'cpu'
    )
    logger = processors_nhp.LogWriter(args['PathLog'], args)

    #max_episode = args['MaxEpoch'] * len(data)
    max_episode = args['MaxEpoch']
    report_gap = args['TrackPeriod']

    optimizer = optim.Adam(
        model.parameters(), lr=lr
    )
    optimizer.zero_grad()

    optimizer_nhp = optim.Adam(
        agent.parameters(), lr=lr
    )
    optimizer_nhp.zero_grad()

    print('training begin')
    #sys.stdout.flush()

    time_train_only = 0.0
    nhpinput=[]

    ##训练MIMN
    for itr in range(max_episode):
        #利用多线程调出数据，一次是Batchsize大小
        print("epoch" + str(itr))
        loss_sum = 0.0
        accuracy_sum = 0.
        aux_loss_sum = 0.
        for src,tgt in train_data:
            ####加入时间戳信息 ,有batch_size条
            nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask,time_list= prepare_data(
                src, tgt)

            model.train()
            # 训练
            loss, aux_loss, acc,w_list= model([nick_id, item_id, cate_id, hist_item, hist_cate,
                                         neg_item, neg_cate, hist_mask, label])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #test(model, 50,EMBEDDING_DIM)
            loss_sum += loss
            accuracy_sum += acc  # accuracy
            aux_loss_sum += aux_loss
            iter += 1
            sys.stdout.flush()


            if(iter>=test_iter): ##训练MIMN稳定一段时间后开始训练NHP
                print("start training NHP")
                # 加载数据
                time_inp = load_interst(w_list, time_list)  ##为了方便测试，这里只取了后7个
                for idx_seq in range(len(time_inp)):
                    one_seq = time_inp[idx_seq]
                    nhpinput.append(proc.processSeq(one_seq, fix_group=False))  ##这里都没问题


                # 增加dtime_sampling(t-ti), index_of_hidden_sampling，大小len-2长度-----统一了event的长度，10
                batchdata_seqs = proc.processBatchSeqsWithParticles(nhpinput)
                agent.train()
                objective, _, all_lambda_sample = agent(batchdata_seqs)
                objective.backward()
                # torch.nn.utils.clip_grad_norm(agent.parameters(), 0.25)
                optimizer_nhp.step()
                optimizer_nhp.zero_grad()

            # 加载valid集，调用DPP，测试评估
            if (iter % test_iter) == 0:
                total=0
                total_recall=0.0
                total_ndcg=0.0
                total_hitrate=0
                total_diversity=0.0

                print("start eval model")
                for src, tgt in valid_data:
                    nick_id_valid, item_id_valid, cate_id_valid, label_valid, hist_item_valid, hist_cate_valid, neg_item_valid, neg_cate_valid, hist_mask_valid, time_list_valid,target_item= prepare_data(
                        src, tgt,0)
                    model.eval()
                    loss_valid, aux_loss_valid, acc_valid, w_list_valid = model(
                        [nick_id_valid, item_id_valid, cate_id_valid, hist_item_valid, hist_cate_valid,
                         neg_item_valid, neg_cate_valid, hist_mask_valid, label_valid])

                    nhpinput_valid = []
                    time_inp_valid = load_interst(w_list_valid, time_list_valid)
                    for idx_seq in range(len(time_inp_valid)):
                        one_seq_valid = time_inp_valid[idx_seq]
                        nhpinput_valid.append(proc.processSeq(one_seq_valid, fix_group=False))

                    # 增加dtime_sampling(t-ti), index_of_hidden_sampling，大小len-2长度-----统一了event的长度，10
                    batchdata_seqs_valid = proc.processBatchSeqsWithParticles(nhpinput_valid)
                    agent.eval()
                    objective_test, _, all_lambda_sample_valid = agent(batchdata_seqs_valid)
                    print("end eval")

                    dpp_item=cal_dpp(model, EMBEDDING_DIM, BATCH_SIZE, args['topN'],Memory_Size,all_lambda_sample_valid,dpp_=True,nhp=True)
                    total_recall,total_ndcg,total_hitrate,total_diversity=evaluate(target_item,dpp_item,item_cate_map,total_recall,total_ndcg,total_hitrate,total_diversity)
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
                writer.add_scalar('train/loss', loss_sum / test_iter, iter)
                for key, value in metrics.items():
                    writer.add_scalar('eval/' + key, value, iter)

                if recall > best_metric:
                    best_metric = recall
                    saveVariableOnDisk(best_metric, 'metric')
                    #torch.save(model, best_model_path) #保存，path是pkl
                    #model=torch.load(best_model_path)  #加载
                loss_sum = 0.0
                accuracy_sum = 0.0
                aux_loss_sum = 0.0


    print("end train")


'''
    #训练nhp
    #max_episode=50
    # train nhp 应该在while true循环之外。
    # 加载数据集
    # 是一个list，表示一个用户的时序行为集合，每个行为是一个字典，分别带有这样的信息
    # #{'time_since_start': 0.0, 'time_since_last_event': 0.0, 'type_event': 1}]
    
####开始测试
    nhpinput_test = []
    time_list_test=[]
  ##MIMN
    test_data_pool, _stop, _ = generator_queue(test_data)  # q, _stop, generator_threads
    while True:
        if _stop.is_set() and test_data_pool.empty():
            break
        if not test_data_pool.empty():
            src, tgt = test_data_pool.get()
        else:
            continue
        ####加入时间戳信息 ,有batch_size条
        nick_id_test, item_id_test, cate_id_test, label_test, hist_item_test, hist_cate_test, neg_item_test, neg_cate_test, hist_mask_test, time_list_test = prepare_data(
            src, tgt)
        model.eval()
        loss_test, aux_loss_test, acc_test, Memory_Mat_test, w_list_test = model([nick_id_test, item_id_test, cate_id_test, hist_item_test, hist_cate_test,
                                                         neg_item_test, neg_cate_test, hist_mask_test, label_test])
###NHP
    for idx_seq in range(len(time_list_test)):
        one_seq_test = time_list_test[idx_seq]
        nhpinput_test.append(proc.processSeq(one_seq_test, fix_group=False))

    #增加dtime_sampling(t-ti), index_of_hidden_sampling，大小len-2长度-----统一了event的长度，10
    batchdata_seqs_test = proc.processBatchSeqsWithParticles(nhpinput_test)
    agent.eval()
    objective_test, _, all_lambda_sample_test = agent(batchdata_seqs_test)
'''




if __name__ == '__main__':
    args = parser.parse_args()
    #mimn
    SEED = args.random_seed
    Model_Type = args.model_type
    Memory_Size = args.memory_size
    Mem_Induction = args.mem_induction
    Util_Reg = args.util_reg
    #nhp
    dict_args = vars(args)
    root_path = os.path.abspath(dict_args['RootPath'])
    dict_args['PathData'] = os.path.join(root_path, 'data', dict_args['Dataset'])
    dict_args['Version'] = torch.__version__
    id_process = os.getpid()
    dict_args['ID'] = id_process
    time_current = datetime.datetime.now().isoformat()
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
    #print(folder_name)

    path_log = os.path.join(root_path, 'logs', dict_args['Dataset'], folder_name)
    os.makedirs(path_log)

    file_log = os.path.join(path_log, 'log.txt')
    file_model = os.path.join(path_log, 'saved_model')
    dict_args['PathLog'] = file_log
    dict_args['PathSave'] = file_model
    dict_args['Model'] = 'nhp'

    if '' in dict_args:
        del dict_args['']

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train(args=dict_args,model_type=Model_Type, Memory_Size=Memory_Size, Mem_Induction=Mem_Induction, Util_Reg=Util_Reg)



