# coding=utf-8
import tensorflow as tf
import sys
import faiss
import torch
import random
import numpy as np

from data_iterator import DataIterator

# list1=[1,2.3,4,5,0,0]
# list2=[1,2.3,4,5,6,7]
# list3=[1,2.3,4,5,6,7]
#
# list4=[[1,2.3],[4,5,6]]
# multi_list1 = list(map(list, zip(list4[0],list4[1])))
# multi_list=list(map(list,zip(list1,list2,list3)))
# new=sum(multi_list,[])

A=torch.ones(2,3)
B=torch.rand(2,3)
c=torch.rand(2,3)
C=torch.cat((A,B,c),-1)
m=C.reshape(2,3,3)
print("fa")



# m=zip(list,list2)
# history_mask = np.greater(list,0) * 1.0


def prepare_data(src, target, train_flag=1):
    nick_id, item_id = src
    # 加入时间信息以及target items
    hist_item_list,hist_cate_list,hist_mask,neg_item_list,neg_item_list = target
    return nick_id, item_id, hist_item_list,hist_cate_list,hist_mask,neg_item_list,neg_item_list


train_data = DataIterator("../data/book_data/book_test.txt", 128, 20, train_flag=0)
iter = 1
for src, tgt in train_data:
    # nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask = prepare_data(src, tgt)
    nick_id, item_id, hist_item_list,hist_cate_list,hist_mask,neg_item_list,neg_item_list= prepare_data(src, tgt)
    print(len(item_id[0]))
    if iter > 100:
        break
    iter = iter + 1
print("iter")
print(iter)
# a = torch.tensor(1.0)
# a_ = a.clone()
# a_.requires_grad_() #require_grad=True
# y = a_ ** 2
# y.backward()
# print(a.grad) # None
# print(a_.grad)
