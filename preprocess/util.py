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

