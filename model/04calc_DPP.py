from dpp import dpp
import logging
import numpy as np
from sklearn.cluster import KMeans
from util import *

K_dpp = 1
K_dtw = 2
max_length = 50
logging.basicConfig(filename='calc_DPP.log',filemode='w',level=logging.DEBUG,format='%(asctime)s %(message)s')
def generate_vertex_in_K(depth_vertex,v1):
    lis = []
    upper = min( len(depth_vertex[v1]),K_dpp+1 )
    for layer in range(0,upper):
        lis.extend(depth_vertex[v1][layer])
    return lis

names = ['cora','citeseer','pubmed']
for choice in range(0,3):
    print('calculating {}'.format(names[choice]))
    g = restoreVariableFromDisk('graph_{}'.format(names[choice]))
    depth_vertex = restoreVariableFromDisk('depth_vertex_{}'.format(names[choice]))
    feature = restoreVariableFromDisk('feature_{}'.format(names[choice]))
    R = restoreVariableFromDisk('r_{}'.format(names[choice]))
    degreelist = restoreVariableFromDisk('degreelist_{}'.format(names[choice]))
    Lam = restoreVariableFromDisk('lamta_{}'.format(names[choice]))
    train_list = restoreVariableFromDisk('list_{}'.format(names[choice]))
    net_work_input_train = {}
    net_work_input_test = {}
    cnt = 0
    train_cnt = 0
    for v1 in g.keys():

        t1 = time()
        lis = generate_vertex_in_K(depth_vertex,v1)
        cnt = cnt + 1
        #logging.info( 'from dpp-- node{}has{}vertex'.format(v1,len(lis)) )
        flag = False
        for v2 in lis:
            Len = len(feature[v2])
            r = R[min(v1,v2),max(v1,v2)]
            l = Lam[v1,v2]
            if( flag == False ):
                flag = True
                B_test = np.array(feature[v2],dtype='float').reshape(1,Len)
                W_test = np.array( (1),dtype='float').reshape(1,1)
            else:
                B_test = np.append(B_test, np.array(feature[v2],dtype='float').reshape(1,Len),0)
                W_test = np.append(W_test,np.array((1),dtype='float').reshape(1,1),0)

        flag2 = False
        for v2 in lis:
            if train_list[v1] != 'train' or train_list[v2] != 'train':
                continue
            Len = len(feature[v2])
            r = R[min(v1, v2), max(v1, v2)]
            l = Lam[v1, v2]
            if (flag2 == False):
                flag2 = True
                B_train = np.array(feature[v2], dtype='float').reshape(1, Len)
                W_train = np.array((1), dtype='float').reshape(1, 1)
            else:
                B_train = np.append(B_train, np.array(feature[v2], dtype='float').reshape(1, Len), 0)
                W_train = np.append(W_train, np.array((1), dtype='float').reshape(1, 1), 0)



        seed = 9
        if flag2 == True and B_train.shape[0] > 1:
            clf = KMeans(n_clusters=2, random_state=seed)
            clf.fit(B_train)
            clf.labels_ = clf.labels_^clf.labels_[0]
            clf.labels_ = -clf.labels_
            clf.labels_ = clf.labels_.astype(float)
            clf.labels_ = np.exp( clf.labels_ )
        if  flag2 == True and B_train.shape[0] > 1:
            clf.labels_ = clf.labels_.reshape( W_train.shape )
            W_train = W_train*clf.labels_
        if flag2 == True:
            B_train /= np.linalg.norm(B_train, axis=1, keepdims=True)



        if B_test.shape[0] > 1:
            clf = KMeans(n_clusters=2, random_state=seed)
            clf.fit(B_test)
            clf.labels_ = clf.labels_^clf.labels_[0]
            clf.labels_ = -clf.labels_
            clf.labels_ = clf.labels_.astype(float)
            clf.labels_ = np.exp( clf.labels_ )
        B_test /= np.linalg.norm(B_test, axis=1, keepdims=True)
        if B_test.shape[0] > 1:
            clf.labels_ = clf.labels_.reshape( W_test.shape )
            W_test = W_test*clf.labels_


        if flag2 == True:
            B_train = W_train*B_train
        B_test = W_test*B_test

        if flag2 == True:
            kernel_train = np.dot(B_train, B_train.T)
        kernel_test = np.dot(B_test,B_test.T)

        if flag2 == True:
            select_indexs_train = dpp(kernel_train, max_length )
        select_indexs_test = dpp(kernel_test, max_length )


        select_vertexs_train = []
        select_vertexs_test = []


        if flag2 == True:
            for i in select_indexs_train:
                select_vertexs_train.append(lis[i])
        for i in select_indexs_test:
            select_vertexs_test.append(lis[i])

        t2 = time()
        logging.info('cal_lamta{} time:{}'.format(v1, t2 - t1))
        if flag2 == True:
            net_work_input_train[v1] = select_vertexs_train
        net_work_input_test[v1] = select_vertexs_test

        logging.info('train : select {} nodes'.format(len(select_vertexs_train)))
        logging.info('test : select {} nodes'.format(len(select_vertexs_test)))
    # 目标节点v1所选出来的聚合节点id
    # {id(int): list[id(int)]}
    saveVariableOnDisk(net_work_input_train,'net_work_input_{}_train'.format(names[choice]))
    saveVariableOnDisk(net_work_input_test, 'net_work_input_{}_test'.format(names[choice]))


