# coding=utf-8
import pandas as pd
import random
from DataProcess.util_time import inteval
from model.util import *

RAW_DATA_FILE = '../data/taobao_data/UserBehaviorTime.csv'
Train_Input_FILE = '../data/taobao_data/train.pkl'

DATASET_PKL = '../data/taobao_data/dataset.pkl'
Train_File = "../data/taobao_data/taobao_train.txt"
Test_File = "../data/taobao_data/taobao_test.txt"
Valid_File = "../data/taobao_data/taobao_valid.txt"

Train_handle = open(Train_File, 'w')
Test_handle = open(Test_File, 'w')
Valid_handle = open(Valid_File, 'w')
Feature_handle = open("../data/taobao_data/taobao_feature.pkl", 'wb+')

Dpp_Scale = 10
MAX_LEN_ITEM = 200
#FRAC=0.01   #3140  31399 313995 3139947


def to_df(file_name):
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])  #100150807   ###10000000--55555555是可以运行的
    #df = df.sample(frac=FRAC, replace=False, random_state=0)
    return df





# 将数据属性统一编码为0-len
def remap_item_type(df):
    # 处理 item id，编号为0---len(id)
    item_key = sorted(df['iid'].unique().tolist())  # [a,b,c]
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(item_len)))  # {'a': 0, 'b': 1, 'c': 2}
    df['iid'] = df['iid'].map(lambda x: item_map[x])  # 给用户id替换位对应的编号

    # 处理用户id，编号为
    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(item_len, item_len + user_len)))
    df['uid'] = df['uid'].map(lambda x: user_map[x])

    # 处理类别id，编号为
    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(user_len + item_len, user_len + item_len + cate_len)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])

    # 处理btag，编号为0---len(btag)
    btag_key = sorted(df['btag'].unique().tolist())
    btag_len = len(btag_key)
    btag_map = dict(zip(btag_key, range(user_len + item_len + cate_len, user_len + item_len + cate_len + btag_len)))
    df['btag'] = df['btag'].map(lambda x: btag_map[x])

    print("sort finish")
    return df,item_len, user_len + item_len + cate_len +btag_len + 1


# 一个user对应于多条历史记录，存储在group对象中
def gen_user_item_group(df, item_cnt, feature_size):
    # 以uid作为分组group顺序，uid按照位置编号，time也是按照从小到大排序的
    # 将'uid', 'time'、item 、categary、btag信息放入组中，每一组是一个uid和其对应的time、item 、categary、btag们
    user_df = df.sort_values(['uid', 'time']).groupby('uid')
    item_df = df.sort_values(['iid', 'time']).groupby('iid')  #数字太大 内存会溢出
    print("group completed")
    return user_df,item_df

# 生成测试集和数据集，存储方式 user target item-list cate-list
def gen_dataset(user_df,item_df, item_cnt, feature_size, dataset_pkl):
    train_sample_list = []
    test_sample_list = []
    valid_sample_list=[]

    # get each user's last touch point time
    # 存储每个用户最近一次浏览的时间
    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['time'].tolist()[-1])  # 最后一个元素
    print("get user last touch time completed")
    # 将时间在前面的用户作为训练集，后面的验证集和测试集
    user_last_touch_time_sorted = sorted(user_last_touch_time)
    split_time_1 = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.8)]
    split_time_2 = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.9)]

    cnt = 0
    valid_target = []
    test_target = []

    for uid, hist in user_df:
        cnt += 1
        # print(cnt)
        # 将group对象分别放入列表中
        item_hist = hist['iid'].tolist()
        #因为针对小测试集，有的用户的item收集的并不全
        if(len(item_hist)<Dpp_Scale):
            continue
        cate_hist = hist['cid'].tolist()
        btag_hist = hist['btag'].tolist()


        # 转换时间time_since_last_event
        time_hist = hist['time'].tolist()
        time_since_last_event = []
        time_since_start = []
        t_before, time_start = time_hist[0], time_hist[0]
        for t in time_hist:
            #print(t,t_before)
            time_s_last_event = inteval(t_before, t)
            time_s_start = inteval(time_start, t)
            time_since_last_event.append(time_s_last_event)
            time_since_start.append(time_s_start)
            t_before = t


        target_item_time = hist['time'].tolist()[-1]

        target_item = item_hist[-1]
        target_item_cate = cate_hist[-1]
        target_item_btag = feature_size
        label = 1
        test = (target_item_time > split_time_2)
        train = (target_item_time < split_time_1)

        # neg sampling 随机负采样 target item
        neg = random.randint(0, 1)
        if neg == 1:
            label = 0
            while target_item == item_hist[-1]:
                target_item = random.randint(0, item_cnt - 1)
                target_item_cate = item_df.get_group(target_item)['cid'].tolist()[0]  #第几个都一样
                #target_item_cate = item_cate_map[target_item]
                target_item_btag = feature_size

        # the item history part of the sample整合到item_part中
        item_part = []
        for i in range(len(item_hist) - 1):
            item_part.append([uid, item_hist[i], cate_hist[i], btag_hist[i]])
        item_part.append([uid, target_item, target_item_cate, target_item_btag])
        # item_part_len = min(len(item_part), MAX_LEN_ITEM)

        # choose the item side information: which user has clicked the target item  列表长度对齐
        # padding history with 0
        if len(item_part) <= MAX_LEN_ITEM:
            #item_part_pad = [[0] * 4] * (MAX_LEN_ITEM - len(item_part)) + item_part
            if train:
                item_part_pad = item_part+[[0] * 4] * (MAX_LEN_ITEM - len(item_part))
                len_item=len(item_hist)
            else:
                item_part_pad = item_part[:len(item_part)-Dpp_Scale-1] + [[0] * 4] * (MAX_LEN_ITEM - len(item_part)+Dpp_Scale+1)
                len_item = len(item_hist)-Dpp_Scale
        else:
            item_part_pad = item_part[len(item_part) - MAX_LEN_ITEM:len(item_part)]
            len_item=MAX_LEN_ITEM

        # gen sample
        # sample = (label, item_part_pad, item_part_len, user_part_pad, user_part_len)

        # 前70%的数据作为训练数据集，后30%作为测试集 user target item-list cate-list
        test_time_sample_list=[]
        train_time_sample_list=[]
        valid_time_sample_list = []
        if test:
            # test_set.append(sample)
            cat_list = []
            item_list = []
            #加入时间信息
            for i in range(len_item- 1):   ##这里有问题，time_part才是被统一化的，所以要考虑，大于maxlen的item_hist要被裁剪
                test_time_sample_list.append(
                    {'time_since_start': time_since_start[i], 'time_since_last_event': time_since_last_event[i],
                     'type_event': cate_hist[i]})

            #整理item/cate
            for i in range(len(item_part_pad)):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
                # cat_list.append(item_part_pad[i][0])
                # .join()函数: str以.来分割连接
            test_sample_list.append(
                str(uid) + "\t" + str(target_item) + "\t" + str(target_item_cate) + "\t" + str(label) + "\t" + ",".join(
                    map(str, item_list)) + "\t" + ",".join(map(str, cat_list)) + "\t" +str(test_time_sample_list)+"\n")
            #目标DPP item
            target_dpp = item_hist[-Dpp_Scale:]   # 提取dpp大小的taget-item
            test_target.append(target_dpp)

        elif train:
            cat_list = []
            item_list = []
            # btag_list = []
            #加入时间信息
            for i in range(len_item - 1):
                train_time_sample_list.append(
                    {'time_since_start': time_since_start[i], 'time_since_last_event': time_since_last_event[i],
                     'type_event': cate_hist[i]})

            for i in range(len(item_part_pad)):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
            train_sample_list.append(
                str(uid) + "\t" + str(target_item) + "\t" + str(target_item_cate) + "\t" + str(label) + "\t" + ",".join(
                    map(str, item_list)) + "\t" + ",".join(map(str, cat_list)) + "\t" +str(train_time_sample_list)+"\n")
        else:  #valid data
            cat_list = []
            item_list = []
            # btag_list = []
            # 加入时间信息
            for i in range(len_item - 1):
                valid_time_sample_list.append(
                    {'time_since_start': time_since_start[i], 'time_since_last_event': time_since_last_event[i],
                     'type_event': cate_hist[i]})

            for i in range(len(item_part_pad)):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
            valid_sample_list.append(
                str(uid) + "\t" + str(target_item) + "\t" + str(target_item_cate) + "\t" + str(label) + "\t" + ",".join(
                    map(str, item_list)) + "\t" + ",".join(map(str, cat_list)) + "\t" + str(
                    valid_time_sample_list) + "\n")
            # 目标DPP item
            target_dpp = item_hist[-Dpp_Scale:]  # 提取dpp大小的taget-item
            valid_target.append(target_dpp)


    print("cnt", cnt)  # 987994

   #保证数据时256的倍数
    train_sample_length_quant = int(len(train_sample_list) / 256)*256
    test_sample_length_quant = int(len(test_sample_list) / 256)*256
    valid_sample_length_quant =int( len(valid_sample_list) / 256)*256
    
   #print"train_sample_list length", len(train_sample_list)  # length 691603
    train_sample_list = train_sample_list[:train_sample_length_quant]
    test_sample_list = test_sample_list[:test_sample_length_quant]
    test_target=test_target[:test_sample_length_quant]
    valid_sample_list = valid_sample_list[:valid_sample_length_quant]
    valid_target=valid_target[:valid_sample_length_quant]

    random.shuffle(train_sample_list)
    # print "train_sample_list length",len(train_sample_list)  #length 691456

    #存储目标item
    saveVariableOnDisk(valid_target,'/taobao_data/valid_target_taobao')
    saveVariableOnDisk(test_target,'/taobao_data/test_target_taobao')
    # saveVariableOnDisk(train_sample_list,'/taobao_data/train_sample_list')
    # saveVariableOnDisk(test_sample_list, '/taobao_data/test_sample_list')
    # saveVariableOnDisk(valid_sample_list, '/taobao_data/valid_sample_list')

    return train_sample_list, test_sample_list,valid_sample_list


# 给训练集和测试集随机负采样，大小和正样本相同，且保证不与正样本重复，结果存储到文件中
def produce_neg_item_hist_with_cate(train_file, test_file,valid_file):
    item_dict = {}  #item全集
    sample_count = 0
    hist_seq = 0
    # 读取训练集和测试集，将一个用户的item和cate一一对应打包到hist_list， 作为字典item_dict的key，value为0
    for line in train_file:
        units = line.strip().split("\t")
        item_hist_list = units[4].split(",")
        cate_hist_list = units[5].split(",")
        hist_list = list(zip(item_hist_list, cate_hist_list))
        hist_seq = len(hist_list)
        sample_count += 1
        for item in hist_list:
            item_dict.setdefault(str(item), 0)


    for line in test_file:
        units = line.strip().split("\t")
        item_hist_list = units[4].split(",")
        cate_hist_list = units[5].split(",")
        hist_list = list(zip(item_hist_list, cate_hist_list))
        hist_seq = len(hist_list)
        sample_count += 1
        for item in hist_list:
            item_dict.setdefault(str(item), 0)

    for line in valid_file:
        units = line.strip().split("\t")
        item_hist_list = units[4].split(",")
        cate_hist_list = units[5].split(",")
        hist_list = list(zip(item_hist_list, cate_hist_list))
        hist_seq = len(hist_list)
        sample_count += 1
        for item in hist_list:
            item_dict.setdefault(str(item), 0)

    # 随机负采样
    del (item_dict["('0', '0')"])  # 删除填充项
    # 随机组合（item,cate）组成sample行，hist_seq+20列的数组矩阵
    #neg_array = np.random.choice(np.array(item_dict.keys()), (sample_count, hist_seq + 20))
    neg_array = np.random.choice(np.array(list(item_dict.keys())), (sample_count, hist_seq + 20))
    neg_list = neg_array.tolist()
    # 过滤负样本，保证每个用户的负样本不和用户的正样本（历史行为）重合
    # line.strip() + "\t" + ",".join(neg_item_list) + "\t" + ",".join(neg_cate_list) + "\n"写入训练和测试文件
    sample_count = 0
    for line in train_file:
        units = line.strip().split("\t")
        item_hist_list = units[4].split(",")
        cate_hist_list = units[5].split(",")
        hist_list = list(zip(item_hist_list, cate_hist_list))
        hist_seq = len(hist_list)
        neg_hist_list = []
        # 将neg_list的（item,cate）对应放入neg_hist_list，并且保证不在对应hist_list中出现
        for item in neg_list[sample_count]:
            item = eval(item)  # 转化为tuple
            if item not in hist_list:
                neg_hist_list.append(item)
            if len(neg_hist_list) == hist_seq:
                break
        sample_count += 1
        neg_item_list, neg_cate_list = zip(*neg_hist_list)  # 解压缩
        Train_handle.write(line.strip() + "\t" + ",".join(neg_item_list) + "\t" + ",".join(neg_cate_list) + "\n")

    for line in test_file:
        units = line.strip().split("\t")
        item_hist_list = units[4].split(",")
        cate_hist_list = units[5].split(",")
        hist_list = list(zip(item_hist_list, cate_hist_list))
        hist_seq = len(hist_list)
        neg_hist_list = []
        for item in neg_list[sample_count]:
            item = eval(item)
            if item not in hist_list:
                neg_hist_list.append(item)
            if len(neg_hist_list) == hist_seq:
                break
        sample_count += 1
        neg_item_list, neg_cate_list = zip(*neg_hist_list)
        Test_handle.write(line.strip() + "\t" + ",".join(neg_item_list) + "\t" + ",".join(neg_cate_list) + "\n")

    for line in valid_file:
        units = line.strip().split("\t")
        item_hist_list = units[4].split(",")
        cate_hist_list = units[5].split(",")
        hist_list = list(zip(item_hist_list, cate_hist_list))
        hist_seq = len(hist_list)
        neg_hist_list = []
        for item in neg_list[sample_count]:
            item = eval(item)
            if item not in hist_list:
                neg_hist_list.append(item)
            if len(neg_hist_list) == hist_seq:
                break
        sample_count += 1
        neg_item_list, neg_cate_list = zip(*neg_hist_list)
        Valid_handle.write(line.strip() + "\t" + ",".join(neg_item_list) + "\t" + ",".join(neg_cate_list) + "\n")

def main():

    df = to_df(RAW_DATA_FILE)  # 加载数据集到df
    df, item_cnt, feature_size = remap_item_type(df)  # 将数据属性统一编码为0-len到df，item个数，所有特征大小
    print("feature_size", item_cnt, feature_size)  # 4162024 5159462
    feature_total_num = feature_size + 1
    pickle.dump(feature_total_num, Feature_handle)  # 将获得的特征大小放入文件Feature_handle中
    user_df,item_df = gen_user_item_group(df, item_cnt, feature_size)  # 一个user对应于多条历史记录，按时间顺序存储在group对象user_df中，item也是
    train_sample_list, test_sample_list,valid_sample_list = gen_dataset(user_df, item_df,item_cnt, feature_size,DATASET_PKL)  # 生成测试集和数据集，存储方式 user target（id cate label） item-list cate-list 负样本item-list  负样本cate-list
    # train_sample_list=restoreVariableFromDisk('/taobao_data/train_sample_list')
    # test_sample_list=[]
    # valid_sample_list=[]
    # test_sample_list=restoreVariableFromDisk('/taobao_data/test_sample_list')
    # valid_sample_list=restoreVariableFromDisk('/taobao_data/valid_sample_list')
    produce_neg_item_hist_with_cate(train_sample_list, test_sample_list,valid_sample_list)  # 给训练集和测试集随机负采样，大小和正样本相同，且保证不与正样本重复，结果存储到文件中

    #to_csv(RAW_DATA_FILE)
if __name__ == '__main__':
    main()
