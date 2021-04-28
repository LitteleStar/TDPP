# coding=utf-8
import pandas as pd
import random
from util_time import inteval
from util import *

'''
功能：划分train/valid/test data
train data: target item 1个   dpp_target_items 总item的20%个  item hist:200个， 不够除去target item补0够取前200
test/valid data : target item 1个  dpp_target_items 总item的20%个  item hist:200个，不够除去dpp_target_items补0  够取后200
划分依据： 用户最后一个时间大小排序  8:1:1========> 用户按序排列 8:1:1
'''
##这里target item要改一下

name='book'
if name=='book':
    RAW_DATA_FILE = '../data/book_data/UserBehavior_80.csv'
    path='../data/book_data/'
    # 60 60
    # item=27190
    # user=6331
    ##80 10
    # item 18396
    # user 25717
    batch_size = 128
    MAX_LEN_ITEM = 20
    frac=0.2  ##0.05
    scale=539740800
elif name=='taobao':
    RAW_DATA_FILE = '../data/taobao_data/UserBehavior_5.csv'
    path='../data/taobao_data/'
    batch_size = 256
    MAX_LEN_ITEM = 50
    frac=0.2
    scale=22179163

hist_seq=100 ##负样本个数
#
Train_File = path + name+ '_train.txt'
Test_File = path + name + '_test.txt'
Valid_File = path + name+ '_valid.txt'
cate_file=path + name+'_item_cate.txt'
feature_file=name+'_data/'+name+'_feature'

Train_handle = open(Train_File, 'w')
Test_handle = open(Test_File, 'w')
Valid_handle = open(Valid_File, 'w')



def write_(map):
    with open(cate_file, 'w') as f:
        for key, value in map.items():
            f.write('%s,%s\n' % (key, value))

def to_df(file_name):
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid',  'time'])  # 100150807   ###10000000--55555555是可以运行的
    #df = df.sample(frac=0.1, replace=False, random_state=0)  ##这样不是顺序取样，不可取
    return df


# 将数据属性统一编码为0-len
def remap_item_type(df):
    # 处理 item id，编号为1---len(id)+1
    a=len(df)
    item_key = sorted(df['iid'].unique().tolist())  # [a,b,c]
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(1,item_len+1)))  # (0:243),(1,233)  考虑负样本
    df['iid'] = df['iid'].map(lambda x: item_map[x])  # 给用户id替换位对应的编号

    # 处理用户id，编号为
    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(user_len)))
    df['uid'] = df['uid'].map(lambda x: user_map[x])

    # 处理类别id，编号为0--len(id)
    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(1,cate_len+1)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])

    print("sort finish")
    return df, item_len+1, cate_len

def item_cate(df):
    #item_ca=df.groupby('iid')['cid'].apply(set).to_dict()
    item_ca = {}
    for index,row in df.iterrows():
        cid=row[2]
        iid=row[1]
        item_ca[iid]=cid
    write_(item_ca)

# 一个user对应于多条历史记录，存储在group对象中
def gen_user_item_group(df, item_cnt, feature_size):
    # 以uid作为分组group顺序，uid按照位置编号，time也是按照从小到大排序的
    # 将'uid', 'time'、item 、categary、btag信息放入组中，每一组是一个uid和其对应的time、item 、categary、btag们
    user_df = df.sort_values(['uid', 'time']).groupby('uid')
    item_df = df.sort_values(['iid', 'time']).groupby('iid')  # 数字太大 内存会溢出
    print("group completed")
    return user_df, item_df


# 生成train,valid,test data，存储方式 user  target-item   target-cate   item-list   cate-list
def gen_dataset(user_df, item_df, hist_seq):
    train_sample_list = []
    test_sample_list = []
    valid_sample_list = []
    cnt = 0
    items=[]
    for item, hist in item_df:
        items.append(item)
    '''
    这样划分的一个问题是用户的item长度都是从小到大，导致test和valid的target item最多
    user_len=len(user_df)
    train=user_len*0.8
    test=user_len*0.9
    '''
    user_last_touch_time = []
    for uid, hist in user_df:
        user_last_touch_time.append(hist['time'].tolist()[-1])
    print("get user last touch time completed")

    user_last_touch_time_sorted = sorted(user_last_touch_time)
    split_time_train = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.8)]
    split_time_test = user_last_touch_time_sorted[int(len(user_last_touch_time_sorted) * 0.9)]

    for uid, hist in user_df:
        cnt += 1
        # 将group对象分别放入列表中
        item_hist = hist['iid'].tolist()
        cate_hist = hist['cid'].tolist()
        target_item_time = hist['time'].tolist()[-1]
        train = (target_item_time < split_time_train)
        test=(target_item_time > split_time_test)
        # 转换时间time_since_last_event，time_since_start
        time_hist = hist['time'].tolist()
        #scale=time_hist[-1]-time_hist[0]
        time_since_last_event = []
        time_since_start = []
        t_before, time_start = time_hist[0], time_hist[0]
        for t in time_hist:
            # print(t,t_before)
            time_s_last_event = inteval(t_before, t,scale)
            time_s_start = inteval(time_start, t,scale)
            time_since_last_event.append(time_s_last_event)
            time_since_start.append(time_s_start)
            t_before=t


        # 8:1:1  user target item-list cate-list
        test_time_sample_list = []
        train_time_sample_list = []
        valid_time_sample_list = []
        item_part = []
        if  train:  #train数据集
            # 加入时间信息
            for i in range(len(item_hist)):
                train_time_sample_list.append(
                    {'time_since_start': time_since_start[i], 'time_since_last_event': time_since_last_event[i],
                     'type_event': cate_hist[i]})
            ##生成负样本
            neg_hist_list = []
            for item in items:
                if item not in item_hist:
                    neg_hist_list.append(item)
                    #cate= hist['cid']
                if len(neg_hist_list) == hist_seq:
                    break
            train_sample_list.append(str(uid) + "\t" + ",".join(map(str, item_hist)) + "\t" + ",".join(map(str, cate_hist))+ "\t" + ",".join(map(str, neg_hist_list))+"\n")

        elif test:   ##test数据集
            Dpp_Scale=int(len(item_hist)*frac)
            if Dpp_Scale>50:
                Dpp_Scale=10
            target_dpp = item_hist[-Dpp_Scale:]  # 提取dpp大小的target-item
            ##除去target，剩下的都是item hist
            for i in range(len(item_hist)-Dpp_Scale):
                item_part.append([uid, item_hist[i], cate_hist[i]])
            #确保item hist长度为MAX_LEN_ITEM
            if len(item_part) <= MAX_LEN_ITEM:
                item_part_pad = item_part[:len(item_part)] + [[0] * 4] * ( MAX_LEN_ITEM - len(item_part))
            else:
                item_part_pad = item_part[len(item_part) - MAX_LEN_ITEM:len(item_part)]

            cat_list = []
            item_list = []
            # # 加入时间信息
            # for i in range(len(item_part)):  ##这里有问题，time_part才是被统一化的，所以要考虑，大于maxlen的item_hist要被裁剪
            #     test_time_sample_list.append(
            #         {'time_since_start': time_since_start[i], 'time_since_last_event': time_since_last_event[i],
            #          'type_event': cate_hist[i]})

            # 整理item/cate
            for i in range(MAX_LEN_ITEM):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])
            ##生成负样本
            neg_hist_list = []
            for item in items:
                if item not in item_hist:
                    neg_hist_list.append(item)
                    # cate= hist['cid']
                if len(neg_hist_list) == hist_seq:
                    break

            test_sample_list.append(
                str(uid) + "\t" + ",".join(map(str, item_list)) + "\t" + ",".join(map(str, cat_list)) + "\t" + ",".join(map(str, target_dpp)) + "\t" + ",".join(map(str, neg_hist_list))+ "\n")

        else:  # valid data
            Dpp_Scale=int(len(item_hist)*frac)
            if Dpp_Scale > 50:
                Dpp_Scale = 10
            target_dpp = item_hist[-Dpp_Scale:]  # 提取dpp大小的taget-item
            ##除去target，剩下的都是item hist
            for i in range(len(item_hist) - Dpp_Scale):
                item_part.append([uid, item_hist[i], cate_hist[i]])
            # 确保item hist长度为MAX_LEN_ITEM
            if len(item_part) <= MAX_LEN_ITEM:
                item_part_pad = item_part[:len(item_part)] + [[0] * 4] * (MAX_LEN_ITEM - len(item_part))
            else:
                item_part_pad = item_part[len(item_part) - MAX_LEN_ITEM:len(item_part)]

            cat_list = []
            item_list = []
            # btag_list = []
            # 加入时间信息
            # for i in range(len(item_part)):
            #     valid_time_sample_list.append(
            #         {'time_since_start': time_since_start[i], 'time_since_last_event': time_since_last_event[i],
            #          'type_event': cate_hist[i]})

            for i in range(MAX_LEN_ITEM):
                item_list.append(item_part_pad[i][1])
                cat_list.append(item_part_pad[i][2])

            # 生成负样本
            neg_hist_list = []
            for item in items:
                if item not in item_hist:
                    neg_hist_list.append(item)
                    # cate= hist['cid']
                if len(neg_hist_list) == hist_seq:
                    break

            valid_sample_list.append(str(uid) + "\t" + ",".join(map(str, item_list)) + "\t" + ",".join(map(str, cat_list)) + "\t" + ",".join(map(str, target_dpp)) + "\t" + ",".join(
                    map(str, neg_hist_list)) + "\n")




    print("cnt", cnt)  # 987994

    # 保证数据是batch_size的倍数
    #train_sample_length_quant = int(len(train_sample_list) / batch_size) * batch_size
    test_sample_length_quant = int(len(test_sample_list) / batch_size) * batch_size
    valid_sample_length_quant = int(len(valid_sample_list) / batch_size) * batch_size

    # print"train_sample_list length", len(train_sample_list)  # length 691603
    train_sample_list = train_sample_list
    test_sample_list = test_sample_list[:test_sample_length_quant]
    valid_sample_list = valid_sample_list[:valid_sample_length_quant]


    random.shuffle(train_sample_list)
    # print "train_sample_list length",len(train_sample_list)  #length 691456

    return train_sample_list, test_sample_list, valid_sample_list

def to_file(train_file, test_file, valid_file):
    for line in train_file:
        Train_handle.write(line.strip() + "\n")

    for line in test_file:
        Test_handle.write(line.strip() +"\n")

    for line in valid_file:
        Valid_handle.write(line.strip() + "\n")


def main():
    df = to_df(RAW_DATA_FILE)  # 加载数据集到df
    df, item_cnt, feature_size = remap_item_type(df)  # 将数据属性统一编码为0-len到df，item个数，所有特征大小
    #item_cate(df)
    print("feature_size", item_cnt, feature_size)  #
    saveVariableOnDisk([item_cnt, feature_size], feature_file)
    user_df, item_df = gen_user_item_group(df, item_cnt, feature_size)  # 一个user对应于多条历史记录，按时间顺序存储在group对象user_df中，item也是
    train_sample_list, test_sample_list, valid_sample_list = gen_dataset(user_df, item_df, hist_seq)  # 生成测试集和数据集，存储方式 user target（id cate label） item-list cate-list 负样本item-list  负样本cate-list
    to_file(train_sample_list, test_sample_list, valid_sample_list)

if __name__ == '__main__':
    main()
