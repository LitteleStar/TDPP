from dpp import dpp
import logging
import numpy as np
#from sklearn.cluster import KMeans
from util import *
from functools import reduce

dicttest={"result":{"code":"110002","msg":"设备设备序列号或验证码错误"}}
K_dpp = 1
K_dtw = 2
max_length = 50
logging.basicConfig(filename='calc_DPP.log',filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')

def generate_embedding(model,item):
    emb=model.get_item_emb()
    item_eb=emb(torch.LongTensor(item).cuda()).cpu()
    return item_eb.detach().numpy().tolist()

def cal_dpp(model,EMBEDDING_DIM,batchsize,topN,Memory_Size,all_lambda_sample_valid=None,dpp_=False,nhp=False):
    if nhp:
        Gamma, Gamma_rank = get_lamda(all_lambda_sample_valid)  # 时序， cate对应---item  可从lam表查 （256,4）
    # Lamda = restoreVariableFromDisk('lamta_{}'.format(names[choice]))  ###user 每个user对应每个兴趣发生的概率
    item, item_D = get_item(model, topN, EMBEDDING_DIM)  # 得到 根据兴趣得到的item list
    ni = Memory_Size
    dpp_item=list()

    for i in range(batchsize):
        interest_dict = dict()
        interest_item = list()
        # try:
        if nhp:
            ##利用兴趣占比和时序选择
            for inter in range(ni):
                #interst_tmp = list(item[i + inter][:Gamma_rank[i][inter] * topN // 10])  ##这里写的有问题
                interst_tmp = list(item[i*ni+ inter][:Gamma_rank[i][inter] * topN // 10])
                interest_dict[Gamma[i][inter]] = interst_tmp
                interest_item.extend(interst_tmp)
            '''
            interval=args['topN']-len(interest)
            if(interval!=0):
                print("Interest Exception!!!")
                liss=[i for i in range(ni)]
                random=np.random.choice(liss)
                interest.append(item[Gamma_rank[i][random]*args['topN']+1:Gamma_rank[i][random]*args['topN']+interval])
'''
        else:
            if len(item)== batchsize:
                item_list_set = set(item[i])
            else:
                item_list_set = set()
                item_list = list(zip(np.reshape(item[i * ni:(i + 1) * ni], -1), np.reshape(item_D[i * ni:(i + 1) * ni], -1)))
                item_list.sort(key=lambda x: x[1], reverse=True)
                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.add(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break
            interest_item = list(item_list_set)
        # except:
        #     print(i)
        if dpp_:
            ##因为dpp出来是编号，所以用一个字典做映射
            B_train = generate_embedding(model, interest_item)  # 返回列表item的embedding 矩阵
            B_train /= np.linalg.norm(B_train, axis=1, keepdims=True)  # 归一化

            if nhp:
                for gamma, item_list in interest_dict.items():  # 对每个user 计算一个dpp
                    W_pram = []
                    g = gamma
                    for i in range(len(item_list)):  # 计算item对应的lamda,
                        W_pram.append(g)
                    # 结果要连乘
                    W_train = reduce(lambda x, y: x * y, W_pram)  # 是一个值
                B_train = W_train * B_train

            kernel_train = np.dot(B_train, B_train.T)
            max_length = topN
            select_indexs = dpp(kernel_train, max_length)
            select_item = [interest_item[i] for i in select_indexs]
            dpp_item.append(select_item)
        else:
            dpp_item.append(interest_item)


    return dpp_item

'''
def generate_embedding(depth_vertex,v1):
    lis = []
    upper = min( len(depth_vertex[v1]),K_dpp+1 )
    for layer in range(0,upper):
        lis.extend(depth_vertex[v1][layer])
    return lis
'''
def dict_get(dict,objkey):
    for k,v in dict.items():
        for kk,vv in v.items():
            if kk==objkey:
                return vv
    return False
def cal_dpp_():
#一个user对应一个item list--target y
#一个user对应一个根据兴趣得到的item list ---y head  给其做embedding得到 n*embedding_dim的B_train
#这几个item对应着时序预测值lam，兴趣占比 gama
    item=restoreVariableFromDisk('predict_item_{}')   #得到 根据兴趣得到的item list  应该是一个字典  user:item
#时序，cate[标号]对应的概率值，cate:概率值   cate对应---item  可从lam表查
    Gamma = restoreVariableFromDisk('gamma_{}')
#兴趣矩阵,全连接后得到lam值，字典对应 user: item:lam
    Lamda = restoreVariableFromDisk('lamta_{}')   ###user 每个user对应每个兴趣发生的概率
#给一个user item type表
    item_type=restoreVariableFromDisk('item_type{}')

    cnt = 0
    train_cnt = 0

    for user,item_list in item.items():  #对每个user 计算一个dpp

        t1 = time()
        W_pram=[]

        for i in item_list: #计算item对应的lamda, gamma
            type=item_type[i]
            #g = dict_get(item,type)
            g = item_list[type]
            l = item_list[i]
            W_pram.append(g*l)

        # 结果要练乘
        W_train=reduce(lambda x, y: x * y, W_pram)  #是一个值

        B_train = generate_embedding(item_list)  # 返回列表item的embedding 矩阵
        B_train /= np.linalg.norm(B_train, axis=1, keepdims=True)   #归一化

        B_train = W_train * B_train

        kernel_train = np.dot(B_train, B_train.T)
        select_indexs_train = dpp(kernel_train, max_length)
        select_vertexs_train = []

        t2 = time()




