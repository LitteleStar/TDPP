import logging
import pickle
from time import time
import math
import faiss
import numpy as np
import torch
from collections import defaultdict

folder_pickles = '../data/'
best_metric=0

#读取和存储数据到pkl
#取数据
def restoreVariableFromDisk(name):
    #logging.info('Recovering variable...')
    #t0 = time()
    val = None
    with open(folder_pickles + name + '.pickle', 'rb') as handle:
        val = pickle.load(handle)
    #t1 = time()
    #logging.info('Variable recovered. Time: {}m'.format((t1-t0)/60))
    return val

#存数据
def saveVariableOnDisk(f,name):
    #logging.info('Saving variable on disk...')
    #t0 = time()
    with open(folder_pickles + name + '.pickle', 'wb') as handle:
        pickle.dump(f, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #t1 = time()
    #logging.info('Variable saved. Time: {}m'.format((t1-t0)/60))
    return
'''
li=[73741,81609]
saveVariableOnDisk(li,'/taobao_data/taobao_feature')
m1,m2=restoreVariableFromDisk('/taobao_data/taobao_feature')

print(m1)
'''
#list=restoreVariableFromDisk('/taobao_data/test_target_taobao')
#print(list[1])

#读取数据
def prepare_data(src, target):
    nick_id, item_id = src
    hist_item, hist_mask = target
    return nick_id, item_id, hist_item, hist_mask

##加载item及其对应的category
def load_item_cate(source):
    item_cate = {}
    with open(source, 'r') as f:
        for line in f:
            conts = line.strip().split(',')
            item_id = int(conts[0])
            cate_id = int(conts[1])
            item_cate[item_id] = cate_id
    return item_cate

#将MIMN中的兴趣提取出来作为nhp的输入
def load_interst(w_list,time_list):
    # 加载兴趣
    w_list = [i[0] for i in w_list]
    w_list_tensor = torch.stack(w_list)
    w_list_max = torch.max(w_list_tensor, dim=-1)[-1]
    #w_to_interst = w_list_max.reshape(256, 200)  ##这样是Z字形提取
    w_to_interst= [w_list_max[:, i] for i in range(256)]
    # 将兴趣放到time_list中
    for i in range(256):
        for j in range(len(time_list[i])):
            time_list[i][j]['type_event'] = int(w_to_interst[i][j])
    # item_ = [time_list[i][-8:] for i in range(len(time_list))]
    # return item_  ##太大，小一点测试速度快
    return time_list

def get_lamda(all_lambda_sample):
    # 读取时序值到文件
    ###all_lambda_sample每个时刻每个类型发生的概率,这里numgroup=1占了一个维度，删掉
    lambda_x = torch.squeeze(all_lambda_sample, dim=1)
    ###取最后一个时刻的每个类型发生的概率
    lambda_y = torch.squeeze(lambda_x[:, -1:, :], dim=1).cpu()
    _,tmp=torch.sort(lambda_y,dim=-1)
    _,rank=torch.sort(tmp)
    lamda_rank=rank+1

    lambda_ = lambda_y.detach().numpy().tolist()

    return lambda_,lamda_rank
#评价指标
def get_item(model,topN,EMBEDDING_DIM):

    item_embs = model.output_item_em()
    '''
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = 0

    try:
        gpu_index = faiss.GpuIndexFlatIP(res, EMBEDDING_DIM, flat_config)
        gpu_index.add(item_embs)
    except Exception as e:
        return {}
'''
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(item_embs)

    user_embs = model.output_user()
    # if model_type=='GRU':
    #     interest_num=len(user_embs)
    #     D_,I_=[],[]
    #     for num in range(interest_num):
    #         d, i = index.search(np.ascontiguousarray(user_embs[num]), topN)  ##这样快一点
    #         D_.append(d)
    #         I_.append(i)
    #     ##找不到一个好方法，不具有
    #     multi_D = list(map(list, zip(D_[0],D_[1], D_[2], D_[3])))
    #     multi_I = list(map(list, zip(I_[0], I_[1], I_[2], I_[3])))
    #     D = sum(multi_D, [])
    #     I = sum(multi_I, [])
    #
    # else:
    D, I = index.search(np.ascontiguousarray(user_embs), topN)  ##这样快一点
    #D, I = index.search(user_embs, topN)
    return I,D


#多样性计算
def compute_diversity(item_list, item_cate_map):
    n = len(item_list)
    diversity = 0.0
    for i in range(n):
        for j in range(i+1, n):
            diversity += item_cate_map[item_list[i]] != item_cate_map[item_list[j]]
    diversity /= ((n-1) * n / 2)
    return diversity

def evaluate(target_item,dpp_item,item_cate_map,total_recall,total_ndcg,total_hitrate,total_diversity,save=True):
    for i, iid_list in enumerate(target_item):
        recall = 0
        dcg = 0.0
        for no, iid in enumerate(iid_list):
            if iid in dpp_item[i]:
                recall += 1
                dcg += 1.0 / math.log(no + 2, 2)
        idcg = 0.0
        for no in range(recall):
            idcg += 1.0 / math.log(no + 2, 2)
        total_recall += recall * 1.0 / len(iid_list)
        if recall > 0:
            total_ndcg += dcg / idcg
            total_hitrate += 1
        if save:
            total_diversity += compute_diversity(dpp_item[i], item_cate_map)
    return total_recall,total_ndcg,total_hitrate,total_diversity


'''
def evaluate_full(test_data, model, topN,EMBEDDING_DIM,memory_size,item_cate_map, save=True, coef=None):
    #item_embs = model.output_item(sess)  #所有item embedding
    item_embs = model.output_item_em()
    
    # res = faiss.StandardGpuResources()
    # flat_config = faiss.GpuIndexFlatConfig()
    # flat_config.device = 0
    # 
    # try:
    #     gpu_index = faiss.GpuIndexFlatIP(res, EMBEDDING_DIM, flat_config)
    #     gpu_index.add(item_embs)
    # except Exception as e:
    #     return {}

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(item_embs)

    total = 0
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    total_map = 0.0
    total_diversity = 0.0
    for src, tgt in test_data:
        nick_id, item_id, hist_item, hist_mask = prepare_data(src, tgt)  #不需要nick_id
        #user_embs = model.output_user(sess, [hist_item, hist_mask])
        user_embs = model.output_user()
        D, I = index.search(user_embs, topN)
        ni = memory_size
        for i, iid_list in enumerate(item_id):
            recall = 0
            dcg = 0.0
            item_list_set = set()
            if coef is None:
                item_list = list(
                    zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                item_list.sort(key=lambda x: x[1], reverse=True)
                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.add(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break
            else:
                origin_item_list = list(
                    zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                origin_item_list.sort(key=lambda x: x[1], reverse=True)
                item_list = []
                tmp_item_set = set()
                for (x, y) in origin_item_list:
                    if x not in tmp_item_set and x in item_cate_map:
                        item_list.append((x, y, item_cate_map[x]))
                        tmp_item_set.add(x)
                cate_dict = defaultdict(int)
                for j in range(topN):
                    max_index = 0
                    max_score = item_list[0][1] - coef * cate_dict[item_list[0][2]]
                    for k in range(1, len(item_list)):
                        if item_list[k][1] - coef * cate_dict[item_list[k][2]] > max_score:
                            max_index = k
                            max_score = item_list[k][1] - coef * cate_dict[item_list[k][2]]
                        elif item_list[k][1] < max_score:
                            break
                    item_list_set.add(item_list[max_index][0])
                    cate_dict[item_list[max_index][2]] += 1
                    item_list.pop(max_index)

            for no, iid in enumerate(iid_list):
                if iid in item_list_set:
                    recall += 1
                    dcg += 1.0 / math.log(no + 2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no + 2, 2)
            total_recall += recall * 1.0 / len(iid_list)
            if recall > 0:
                total_ndcg += dcg / idcg
                total_hitrate += 1
            if not save:
                total_diversity += compute_diversity(list(item_list_set), item_cate_map)

        total += len(item_id)

    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    diversity = total_diversity * 1.0 / total

    if save:
        return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate}
    return {'recall': recall, 'ndcg': ndcg, 'hitrate': hitrate, 'diversity': diversity}


def recall_N(y_true, y_pred, N=10):
    return len(set(y_pred[:N]) & set(y_true)) * 1.0 / len(y_true)


def sampledsoftmaxloss(y_true, y_pred):
    return K.mean(y_pred)

def get_item_embedding(item_embedding, item_input_layer):
    embedding = nn.Embedding(10, 2)  # 10个词，每个词2维
    return embedding



y_pred=[1,2,3]
y_true=[3,4,5]
m=recall_N(y_pred,y_true)  #0.3333
print("good")


def tf_embeddinglook(n_dim,embedding_dim):
    embedding = tf.constant(
        [[0.21,0.41,0.51,0.11],
        [0.22,0.42,0.52,0.12],
        [0.23,0.43,0.53,0.13],
        [0.24,0.44,0.54,0.14]],dtype=tf.float32)
    # 指定的索引，用户找字典
    feature_batch = tf.constant([2, 3, 1, 0])
    feature_batch2=tf.constant([2,3])
    # 在embedding_lookup中，第一个参数相当于一个二维的词表，并根据第二个参数中指定的索引，去词表中寻找并返回对应的行
    get_embedding1 = tf.nn.embedding_lookup(embedding, feature_batch)
    get_embedding2 = tf.nn.embedding_lookup(embedding, feature_batch2)


def embeddinglook(n_dim,embedding_dim,idex):
    
    # 示例
    embeds = t.nn.Embedding(2, 5)
    # 得到word embedding里面关于hello这个词的初始词向量
    idex_lis = [0, 1]
    idex=embeds(t.LongTensor(idex_lis))
    # 全局编码
    embeds = t.nn.Embedding(n_dim, embedding_dim)
    # 得到embedding里面关于idex的初始词向量
    idex_embedding=embeds(t.LongTensor(idex)) #tensor
    idex_embedding=idex_embedding.detach().numpy()  #转为numpy
    return idex_embedding


#embeddinglook(4,4,[3,2,2,2,2])
#py2运行出来的代码需要用这种方式读取
def restoreVariableFromDisk_py2(name):
    logging.info('Recovering variable...')
    t0 = time()
    val = None
    with open(folder_pickles + name + '.pickle', 'rb') as handle:
        val = pickle.load(handle,encoding='iso-8859-1')
    t1 = time()
    logging.info('Variable recovered. Time: {}m'.format((t1-t0)/60))

    return val
import copy
##这里希望把数值分成4:3:2:1的比例，不改变原来的额顺序----考虑优化
Gamma_sort=copy.deepcopy(Gamma)
for list in Gamma_sort:
    list_new=sorted(list)
    for num,li in enumerate(list):
        value=list_new.index(li)+1
        list[num]=value
'''
'''
def generator_queue(generator, max_q_size=20,
                    wait_time=0.1, nb_worker=1):
    generator_threads = []
    q = multiprocessing.Queue(maxsize=max_q_size)
    _stop = multiprocessing.Event()
    try:
        def data_generator_task():
            while not _stop.is_set():
                try:
                    if q.qsize() < max_q_size:
                        # start_time = time.time()
                        # generator_output = next(generator)  # 执行一次Dataiterator里的next函数
                        generator_output = generator.next()
                        # end_time = time.time()
                        # print end_time - start_time
                        q.put(generator_output)
                    else:
                        # time.sleep(wait_time)
                        continue
                except Exception:
                    _stop.set()
                    print("over1")
                    # raise

        for i in range(nb_worker):
            thread = multiprocessing.Process(target=data_generator_task)
            generator_threads.append(thread)
            thread.daemon = True
            thread.start()
    except Exception:
        _stop.set()
        for p in generator_threads:
            if p.is_alive():
                p.terminate()
        q.close()
        print("over")

    return q, _stop, generator_threads
'''