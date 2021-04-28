# coding:utf-8
import os
import sys
from tensorboardX import SummaryWriter
import argparse

import torch.optim as optim

from model.data_iterator import DataIterator
from model_mimn import *

# mimn
parser = argparse.ArgumentParser()
parser.add_argument('-p', type=str, default='train', help='train | test')
parser.add_argument('--random_seed', type=int, default=19)
parser.add_argument('--model_type', type=str, default='MIMN', help='DIEN | MIMN | ..')  # 训练mimn
parser.add_argument('--memory_size', type=int, default=4)
parser.add_argument('--mem_induction', type=int, default=0, help='0:false|1:true')
parser.add_argument('--util_reg', type=int, default=0, help='0:false|1:true')  # 没用到
parser.add_argument('--topN', type=int, default=30, help='item that generate')
parser.add_argument('--cate_file', type=str, default='../data/taobao_data/taobao_item_cate.txt',help='item--cate')
parser.add_argument('-sd', '--Seed', default=12345, type=int,help='Random seed. e.g. 12345')
parser.add_argument('-lr', '--LearnRate', default=1e-4, type=float,help='What is the (starting) learning rate?')  #0.001
parser.add_argument('-gpu', '--UseGPU', action='store_true', help='Use GPU?',default=1) # default=0, type=int, choices=[0,1],
parser.add_argument('-me', '--MaxEpoch', default=1, type=int,help='Max epoch number of training')
# nhp


EMBEDDING_DIM = 16
HIDDEN_SIZE = 16 * 2
best_auc = 0.0

#
# def prepare_data(src, target, train_flag=1):
#     nick_id, item_id, cate_id = src
#     # 加入时间信息以及valid和test增加target
#     if train_flag == 0:
#         label, hist_item, hist_cate, neg_item, neg_cate, hist_mask, time_list_array, target_item = target
#         return nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask, time_list_array, target_item
#     label, hist_item, hist_cate, neg_item, neg_cate, hist_mask, time_list_array = target
#     return nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask, time_list_array
def prepare_data(src, target, train_flag=1):
    nick_id, item_id, cate_id = src
    # 加入时间信息以及valid和test增加target
    label, hist_item, hist_cate, neg_item, neg_cate, hist_mask= target
    return nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask


def train(
        args,
        train_file="../data/taobao_data/taobao_train_1024.txt",
        valid_file="../data/taobao_data/taobao_valid.txt",
        feature_file="../data/taobao_data/taobao_feature.pkl",
        batch_size=128,  ##256
        maxlen=200,
        test_iter=100,  ##100
        save_iter=100,
        model_type='DNN',
        Memory_Size=4,
        Mem_Induction=0,
        Util_Reg=0
):
    np.random.seed(args['Seed'])
    torch.manual_seed(args['Seed'])
    Memory_Size = args['memory_size']
    item_cate_map = load_item_cate(args['cate_file'])
    best_metric = 0.0
    writer = SummaryWriter()

    if model_type != "MIMN" or model_type != "MIMN_with_aux":
        model_path = "dnn_save_path/book_ckpt_noshuff" + model_type
        best_model_path = "dnn_best_model/book_ckpt_noshuff" + model_type
    else:
        model_path = "dnn_save_path/book_ckpt_noshuff" + model_type + str(Memory_Size) + str(Mem_Induction)
        best_model_path = "dnn_best_model/book_ckpt_noshuff" + model_type + str(Memory_Size) + str(Mem_Induction)
    lr = args['LearnRate']

    train_data = DataIterator(train_file, batch_size, maxlen)
    #valid_data = DataIterator(valid_file, batch_size, maxlen,train_flag=0)

    with open(feature_file, 'rb') as handle:
        feature_num = pickle.load(handle, encoding='iso-8859-1')
    n_iid,n_cid=feature_num,feature_num

    #n_iid, n_cid =restoreVariableFromDisk('/taobao_data/taobao_feature')
    BATCH_SIZE = batch_size
    SEQ_LEN = maxlen

    model = MIMN(n_iid, n_cid, EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE, SEQ_LEN, Memory_Size,
                 device='cuda' if args['UseGPU'] else 'cpu')
    if args['UseGPU']:
        model.cuda()
    iter=0
    sampling = 1

    max_episode = args['MaxEpoch']


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
        for src, tgt in train_data:
            ####加入时间戳信息 ,有batch_size条
            # nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask, time_list = prepare_data(
            #     src, tgt)
            nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask= prepare_data(
                src, tgt)
            model.train()
           # 训练
            #reload(model)
            loss, aux_loss, acc, w_list = model([nick_id, item_id, cate_id, hist_item, hist_cate,
                                                 neg_item, neg_cate, hist_mask, label])
            w_list = [i[0] for i in w_list]  # len  batchsize interst_num
            w_list_tensor = torch.stack(w_list)
            w_list_max = torch.max(w_list_tensor, dim=-1)[-1]
            # w_to_interst = w_list_max.reshape(256, 200)  ##这样是Z字形提取
            w_to_interst = [w_list_max[:, i] for i in range(128)]

            loss.backward() # retain_graph=True
            optimizer.step()
            optimizer.zero_grad()
            # test(model, 50,EMBEDDING_DIM)
            loss_sum += loss
            accuracy_sum += acc  # accuracy
            aux_loss_sum += aux_loss
            sys.stdout.flush()
            iter=iter+1

            # # 加载valid集，调用DPP，测试评估
            # if (iter % test_iter) == 0:
            #     total = 0
            #     total_recall = 0.0
            #     total_ndcg = 0.0
            #     total_hitrate = 0
            #     total_diversity = 0.0
            #
            #     print("start eval model")
            #
            #     for src, tgt in valid_data:
            #         nick_id_valid, item_id_valid, cate_id_valid, label_valid, hist_item_valid, hist_cate_valid, neg_item_valid, neg_cate_valid, hist_mask_valid, time_list_valid, target_item = prepare_data(
            #             src, tgt, 0)
            #         model.eval()
            #         loss_valid, aux_loss_valid, acc_valid, w_list_valid = model(
            #             [nick_id_valid, item_id_valid, cate_id_valid, hist_item_valid, hist_cate_valid,
            #              neg_item_valid, neg_cate_valid, hist_mask_valid, label_valid])
            #         dpp_item = cal_dpp(model, EMBEDDING_DIM, BATCH_SIZE, args['topN'],Memory_Size)
            #         total_recall, total_ndcg, total_hitrate, total_diversity = evaluate(target_item, dpp_item,
            #                                                                             item_cate_map, total_recall,
            #                                                                             total_ndcg, total_hitrate,
            #                                                                             total_diversity)
            #         total += BATCH_SIZE
            #     recall = total_recall / total
            #     ndcg = total_ndcg / total
            #     hitrate = total_hitrate * 1.0 / total
            #     diversity = total_diversity * 1.0 / total
            #
            #     ##打印结果
            #     metrics = {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}
            #     log_str = 'iter: %d, train loss: %.4f' % (iter, loss_sum / test_iter)
            #     log_str += ', ' + ', '.join(['valid ' + key + ': %.6f' % value for key, value in metrics.items()])
            #     print(log_str)
            #     writer.add_scalar('train/loss', loss_sum / test_iter, iter)
            #     for key, value in metrics.items():
            #         writer.add_scalar('eval/' + key, value, iter)
            #
            #     if recall > best_metric:
            #         best_metric = recall
            #         saveVariableOnDisk(best_metric, 'metric')
            #         # torch.save(model, best_model_path) #保存，path是pkl
            #         # model=torch.load(best_model_path)  #加载
            if (iter % test_iter) == 0:
                print('iter: %d ----> train_loss: %.4f ---- train_accuracy: %.4f ---- train_aux_loss: %.4f' % \
                      (iter, loss_sum / test_iter, accuracy_sum / test_iter, aux_loss_sum / test_iter))
                # w_list = [i[0] for i in w_list]  ##200 256*4
                # ww = w_list[-5:]
                # with open("../data/w.txt", "a") as f:  # 设置文件对象
                #     for i in ww:  # 对于双层列表中的数据
                #         f.writelines(str(i))
                #     f.writelines("\nend 1\n")
                        # ##保存
                # print('===> Saving models...')
                # state = {
                #     'state': model.state_dict(),
                #     'epoch': itr  # 将epoch一并保存
                # }
                # if not os.path.isdir('checkpoint'):
                #     os.mkdir('checkpoint')
                # torch.save(state, './checkpoint/autoencoder.t7')


                loss_sum = 0.0
                accuracy_sum = 0.0
                aux_loss_sum = 0.0
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
    Mem_Induction = args.mem_induction
    Util_Reg = args.util_reg

    dict_args = vars(args)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train(args=dict_args, model_type=Model_Type, Memory_Size=Memory_Size, Mem_Induction=Mem_Induction,
          Util_Reg=Util_Reg)



