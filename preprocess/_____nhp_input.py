# coding=utf-8
##没用
import pandas as pd
from DataProcess.util_time import inteval
from model.util import *

RAW_DATA_FILE = '../data/taobao_data/UserBehavior.csv'
Train_Input_FILE = '../data/taobao_data/time.pkl'

DATASET_PKL = '../data/taobao_data/dataset.pkl'
'''
Train_File = "./MIMN-master/data/taobao_data/taobao_train.txt"
Test_File = "./MIMN-master/data/taobao_data/taobao_test.txt"

Train_handle = open(Train_File, 'w')
Test_handle = open(Test_File, 'w')
Feature_handle = open("../data/taobao_data/taobao_feature.pkl", 'w')
'''
Dpp_Scale = 1
MAX_LEN_ITEM = 200
FRAC=0.0001   #100,150,807


def to_df(file_name):
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'],nrows=1000)
    #df = df.sample(frac=FRAC, replace=False, random_state=0)
    return df


def to_csv(file_name):
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
    #df = df.sample(frac=FRAC, replace=False, random_state=0)
    time_df = df.sort_values(['time'])

    #saveVariableOnDisk(time_df[0:800], 'taobaodata/time')
    time = time_df[783:df.len - 1500]
    #time.to_csv('../data/taobao_data/UserBehaviorTime.csv')


# 将数据属性统一编码为0-len
def remap(df):
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

    # 处理btag，编号为
    btag_key = sorted(df['btag'].unique().tolist())
    btag_len = len(btag_key)
    btag_map = dict(zip(btag_key, range(user_len + item_len + cate_len, user_len + item_len + cate_len + btag_len)))
    df['btag'] = df['btag'].map(lambda x: btag_map[x])


    #将item_type对应放到pkl
    item_type = df.loc[:, ['iid', 'cid']]
    # print(len(item_type))
    item_type.drop_duplicates(keep='first', inplace=True)
    # print(len(item_type))
    dict_item_type = dict(zip(item_type['iid'], item_type['cid']))
    saveVariableOnDisk(dict_item_type, 'item_type_{}'.format(FRAC))

    print(item_len, user_len, cate_len, btag_len)
    return df, item_len, user_len + item_len + cate_len + btag_len + 1  # +1 is for unknown target btag


# 一个user对应于多条历史记录，存储在group对象中
def gen_user_item_group(df, item_cnt, feature_size):
    # 以uid作为分组group顺序，uid按照位置编号，time也是按照从小到大排序的
    # 将'uid', 'time'、item 、categary、btag信息放入组中，每一组是一个uid和其对应的time、item 、categary、btag们
    user_df = df.sort_values(['uid', 'time']).groupby('uid')
    item_df = df.sort_values(['iid', 'time']).groupby('iid')
    print("group completed")
    return user_df, item_df


def gen_dataset(user_df, item_df, item_cnt, feature_size, dataset_pkl):
    train_sample_list = []
    test_sample_list = []

    # 存储每个用户最近一次浏览的时间
    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['time'].tolist()[-1])  # 最后一个元素
    # 将时间从远到近，远的占70%作为训练集
    user_last_touch_time_sorted = sorted(user_last_touch_time)
    split_time = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.7)]

    cnt = 0
    for uid, hist in user_df:
        cnt += 1
        # print(cnt)
        # 将group对象分别放入列表中
        item_hist = hist['iid'].tolist()
        cate_hist = hist['cid'].tolist()
        btag_hist = hist['btag'].tolist()

        # 转换时间time_since_last_event
        time_hist = hist['time'].tolist()
        time_since_last_event = []
        time_since_start = []
        t_before, time_start = time_hist[0], time_hist[0]
        for t in time_hist:
            time_s_last_event = inteval(t_before, t)
            time_s_start = inteval(time_start, t)
            time_since_last_event.append(time_s_last_event)
            time_since_start.append(time_s_start)
            t_before = t

        target_item_time = hist['time'].tolist()[-1]

        # target_item = item_hist[-Dpp_Scale:]
        target_item_cate = cate_hist[-Dpp_Scale:]
        # target_item_btag = feature_size
        label = 1
        test = (target_item_time > split_time)

        item_part = []
        if test:
            for i in range(len(item_hist) - 1):
                test_sample_list.append(
                    {'time_since_start': time_since_start[i], 'time_since_last_event': time_since_last_event[i],
                     'type_event': cate_hist[i]})
            for j in range(Dpp_Scale):
                item_part.append([target_item_cate[j]])
        else:
            for i in range(len(item_hist) - 1):
                train_sample_list.append(
                    {'time_since_start': time_since_start[i], 'time_since_last_event': time_since_last_event[i],
                     'type_event': cate_hist[i]})
            for j in range(Dpp_Scale):
                item_part.append([target_item_cate[j]])

        train_data = {'dim_process': 75, 'devtest': [], 'args': None, 'dev': [], 'train': [], 'test': []}
        train_data['train'] = train_sample_list
        with open(Train_Input_FILE, 'wb') as fo:  # 将数据写入pkl文件
            pickle.dump(train_data, fo)

        return train_sample_list, test_sample_list


def main():
    '''
    df = to_df(RAW_DATA_FILE)  # 加载数据集到df
    df, item_cnt, feature_size = remap(df)  # 将数据属性统一编码为0-len到df，item个数，所有特征大小
    user_df, item_df = gen_user_item_group(df, item_cnt, feature_size)
    train, test = gen_dataset(user_df, item_df, item_cnt, feature_size, DATASET_PKL)
    # print("feature_size", item_cnt, feature_size)
'''
    to_csv(RAW_DATA_FILE)

if __name__ == '__main__':
    main()