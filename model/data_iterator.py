# coding:utf-8
import numpy as np
import ast
import random

# 读文件操作  ,一次读取batchsize行
class DataIterator:

    def __init__(self, source,
                 batch_size=128,
                 maxlen=100,
                 train_flag=1,  ##   1:表示是train     0：test,valid
                 skip_empty=False,
                 sort_by_length=True,
                 minlen=None,
                 ):
        self.source = open(source, 'r')
        self.file=source
        # self
        self.source_dicts = []

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.minlen = minlen
        self.train_flag=train_flag
        self.skip_empty = skip_empty

        self.sort_by_length = sort_by_length
        self.source_buffer = []
        # self.k = batch_size * max_batch_size
        self.k = batch_size

        self.end_of_data = False


    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)  # 回到文件起始位置

    # 将(uid，target item,target cate)，(target label,正负样本[取后maxlen个]) 转化为int型放入数组放回，且读取batch_size个大小

    def next(self):
        return self.__next__()

    def __next__(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        user_id_list = []
        target = []
        hist_item_list = []
        hist_cate_list = []

        neg_item_list = []
        #neg_cate_list = []
        time_list=[]
        target_item_list=[]

        if len(self.source_buffer) == 0:
            if self.train_flag == 1:  # 如果是训练集，随机读batchsize个数据
                self.source_buffer = random.sample(self.source.readlines(), self.batch_size)
                self.source = open(self.file, 'r')
            else:
                for k_ in range(self.batch_size):  # 这里是读batchsize次
                    ss = self.source.readline()  # 每次读一行
                    if ss == "":
                        break
                    self.source_buffer.append(ss.strip("\n").split("\t"))  # 所以source_buffer就有self.k行

        if len(self.source_buffer) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        try:
            # actual work here
            while True:
                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                    if self.train_flag == 1:
                        ss = ss.strip("\n").split("\t")
                except IndexError:
                    break
                # 将读到的数据都转为int型
                uid = int(ss[0])

                hist_item = list(map(int, ss[1].split(",")))  # uid对应的item s
                hist_cate = list(map(int, ss[2].split(",")))
                # time_stamp=ast.literal_eval(ss[6])  #加入时间信息

                if self.train_flag==1:
                    k = random.choice(range(4, len(hist_item)))
                    target_item_list.append(hist_item[k])
                    if k >= self.maxlen:
                        hist_item_list.append(hist_item[k - self.maxlen: k])
                        hist_cate_list.append(hist_cate[k - self.maxlen: k])

                    else:
                        hist_item_list.append(hist_item[:k] + [0] * (self.maxlen - k))
                        hist_cate_list.append(hist_cate[:k] + [0] * (self.maxlen - k))

                    neg_item = list(map(int, ss[3].split(",")))
                    #neg_cate = list(map(int, ss[4].split(",")))


                else:
                    target_item=list(map(int, ss[3].split(",")))
                    hist_item_list.append(hist_item[-self.maxlen:])  # 从后数maxlen个item放入hist_item_list
                    hist_cate_list.append(hist_cate[-self.maxlen:])
                    target_item_list.append(target_item)
                    neg_item = list(map(int, ss[4].split(",")))
                    #neg_item = random.sample(neg_item, 10)
                    #neg_cate = list(map(int, ss[5].split(",")))

                user_id_list.append(uid)
                ran_ = random.choice(range(10, len(neg_item)))
                neg_item_list.append(neg_item[ran_-10:ran_])
                #neg_cate_list.append(neg_cate[ran_-10:ran_])
                #time_list.append(time_stamp)  #加入时间信息

                if len(user_id_list) >= self.batch_size or len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        # all sentence pairs in maxibatch filtered out because of length
        if len(user_id_list) == 0 :
            source,target = self.next()

        # # 将uid，target item,target cate放入数组
        # uid_array = np.array(user_id_list)
        # target_array = np.array(target)  # label
        #
        # # 正负样本放入数组
        #history_item_array = np.array(hist_item_list)
        # history_cate_array = np.array(hist_cate_list)
        #
        # history_neg_item_array = np.array(neg_item_list)
        # history_neg_cate_array = np.array(neg_cate_list)
        #time_list_array=np.array(time_list)  #加入时间信息
        #history_mask_array = np.greater(history_item_array,0) * 1.0  # 这是什么# ？
        history_mask = np.greater(hist_item_list, 0) * 1.0

        return (user_id_list,target_item_list),(hist_item_list,hist_cate_list,history_mask,neg_item_list)


    # def __next__(self):
    #     if self.end_of_data:
    #         self.end_of_data = False
    #         self.reset()
    #         raise StopIteration
    #
    #     source = []
    #     target = []
    #     hist_item_list = []
    #     hist_cate_list = []
    #
    #     neg_item_list = []
    #     neg_cate_list = []
    #
    #     if len(self.source_buffer) == 0:
    #         for k_ in range(self.k):  # 这里是读batchsize次
    #             ss = self.source.readline()  # 每次读一行
    #             #print(ss)
    #             if ss == "":
    #                 break
    #             self.source_buffer.append(ss.strip("\n").split("\t"))  # 所以source_buffer就有self.k行
    #
    #
    #     if len(self.source_buffer) == 0:
    #         self.end_of_data = False
    #         self.reset()
    #         raise StopIteration
    #     try:
    #         # actual work here
    #         while True:
    #             # read from source file and map to word index
    #             try:
    #                 ss = self.source_buffer.pop()
    #             except IndexError:
    #                 break
    #             # 将读到的数据都转为int型
    #             uid = int(ss[0])
    #             # target item
    #             item_id = int(ss[1])
    #             cate_id = int(ss[2])
    #             label = int(ss[3])
    #
    #             hist_item = list(map(int, ss[4].split(",")))  # uid对应的item s
    #             hist_cate = list(map(int, ss[5].split(",")))
    #
    #             neg_item = list(map(int, ss[6].split(",")))
    #             neg_cate = list(map(int, ss[7].split(",")))
    #
    #             source.append([uid, item_id, cate_id])
    #             target.append([label, 1 - label])
    #             hist_item_list.append(hist_item[-self.maxlen:])  # 从后数maxlen个item放入hist_item_list
    #             hist_cate_list.append(hist_cate[-self.maxlen:])
    #
    #             neg_item_list.append(neg_item[-self.maxlen:])
    #             neg_cate_list.append(neg_cate[-self.maxlen:])
    #
    #             if len(source) >= self.batch_size or len(target) >= self.batch_size:
    #                 break
    #     except IOError:
    #         self.end_of_data = True
    #
    #     # all sentence pairs in maxibatch filtered out because of length
    #     if len(source) == 0 or len(target) == 0:
    #         source, target = self.next()
    #
    #     # 将uid，target item,target cate放入数组
    #     uid_array = np.array(source)[:, 0]
    #     item_array = np.array(source)[:, 1]
    #     cate_array = np.array(source)[:, 2]
    #
    #     target_array = np.array(target)  # label
    #
    #     # 正负样本放入数组
    #     history_item_array = np.array(hist_item_list)
    #     history_cate_array = np.array(hist_cate_list)
    #
    #     history_neg_item_array = np.array(neg_item_list)
    #     history_neg_cate_array = np.array(neg_cate_list)
    #     #time_list_array=np.array(time_list)  #加入时间信息
    #
    #     history_mask_array = np.greater(history_item_array,0) * 1.0  # 这是什么# ？
    #
    #     return (uid_array, item_array, cate_array), (
    #     target_array, history_item_array, history_cate_array, history_neg_item_array, history_neg_cate_array,
    #     history_mask_array)


if __name__ == '__main__':
    def prepare_data(src, target, train_flag=1):
        nick_id, item_id = src
        # 加入时间信息以及target items
        hist_item_list, hist_cate_list, hist_mask, neg_item_list = target
        return nick_id, item_id, hist_item_list, hist_cate_list, hist_mask, neg_item_list


    train_data = DataIterator("../data/book_data/book_test.txt", 128, 20, train_flag=1)
    iter = 1
    for src, tgt in train_data:
        # nick_id, item_id, cate_id, label, hist_item, hist_cate, neg_item, neg_cate, hist_mask = prepare_data(src, tgt)
        nick_id, item_id, hist_item_list, hist_cate_list, hist_mask, neg_item_list = prepare_data(src,tgt)
        print(len(item_id[0]))
        if iter > 300:
            break
        iter = iter + 1
    print("iter")
    print(iter)