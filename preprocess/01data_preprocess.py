import os
import pandas as pd
import sys
import json
import random
from collections import defaultdict
import csv
import pickle

# 22179163 taobao单个用户最大scale
random.seed(1230)
name = 'book'
filter_size = 80 # 过滤item  18396
filter_len = 40 # 过滤item history
#max_time_interval 572659200   539740800
def read_from_amazon(source):
    """
    :param source:  ../data/book_data/merged_book_data.json
    内含UserID ItemID CateID/cate_string timestamp
    :return:  item_count, users
    """
    users = defaultdict(list)
    item_count = defaultdict(int)
    print("amazon data reading")
    cate_map = {}
    cate_index = 0
    with open(source, 'r') as f:
        for line in f:
            # 空格划分
            r = json.loads(line.strip())
            uid = r['user_id']
            iid = r['item_id']
            cate = r['cate']
            if cate not in cate_map:
                cate_map[cate] = cate_index
                cate_index = cate_index + 1
            cid = cate_map[cate]
            ts = float(r['timestamp'])
            item_count[iid] += 1
            users[uid].append((iid, cid, ts))
    return item_count, users


def read_from_taobao(source):
    print("reading from taobao")
    users = defaultdict(list)
    item_count = defaultdict(int)
    i = 0
    with open(source, 'r') as f:
        for line in f:
            '''
            if i>100000:
                break
            i = i + 1
            '''
            conts = line.strip().split(',')
            uid = int(conts[0])
            iid = int(conts[1])
            cate = int(conts[2])
            if conts[3] != 'pv':
                continue
            item_count[iid] += 1  # 计算每个item出现的次数
            ts = int(conts[4])
            if ts > 1512571193 or ts < 1490053988:
                continue
            users[uid].append((iid, cate, ts))  # user映射
    return item_count, users

def read_from_tmall(source):  ##meixiewan
    users = defaultdict(list)
    item_count = defaultdict(int)
    with open(source, 'r') as f:
        next(f)  ##跳过第一行
        for line in f:
            conts = line.strip().split(',')
            uid = int(conts[0])
            iid = int(conts[1])
            if conts[6] != '2':
                continue
            item_count[iid] += 1
            ts = int(conts[5])
            users[uid].append((iid, ts))

def export_map(name, map_dict):
    with open(name, 'w') as f:
        for key, value in map_dict.items():
            f.write('%s,%d\n' % (key, value))

def func(item_count, users, path):
    items = list(item_count.items())  # 罗列出了所有item
    items.sort(key=lambda x: x[1], reverse=True)  # 按照item出现的次数给item排序

    item_total = 0
    for index, (iid, num) in enumerate(items):  # 计算有多少个item的出现次数大于filter_size 5
        if num >= filter_size:
            item_total = index + 1
        else:
            break
    # 只留下次数大于filter_size 5 的item
    item_map = dict(zip([items[i][0] for i in range(item_total)], list(range(1, item_total+1))))

    user_ids = list(users.keys())  # 所有用户
    filter_user_ids = []
    for user in user_ids:
        item_list = users[user]
        index = 0
        item_l = []
        for item, cate, timestamp in item_list:  # books 中只有item 和 timestamp?
            # 但这也包含次数没有filter_size的item，后面写入的时候没录入
            if item in item_map:
                index += 1
                item_l.append((item, cate, timestamp))  # 改动：  只记录大于filter_size的item
        if index >= filter_len:
            filter_user_ids.append(user)
            users[user] = item_l

    user_ids = filter_user_ids  # 把user的浏览的item hist中存在的次数多于5次的item的 个数少于filter_size 5 的用户给去除掉
    random.shuffle(user_ids)
    num_users = len(user_ids)
    user_map = dict(zip(user_ids, list(range(num_users))))
    print("item" + str(len(item_map)))  # 1708530
    print("user" + str(len(user_map)))  # 976777
    total_train = export_data(path+'UserBehavior_'+str(filter_size)+'.csv', user_ids, users, user_map, item_map)
    #time_maxInterval=counter_item(user_ids, users, user_map)  ##数了一下最大时间间隔
    # 把item-cate文件写入：
    # with open('../data/taobao_data/item_map.pickle', 'wb') as handle:
    #     pickle.dump(item_map, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # load_item_cate(item_map,name='taobao')


# 把user_map里面的user都写入文档
def export_data(name, user_list, users, user_map, item_map):
    total_data = 0
    csvfile = open(name, "w")
    writer = csv.writer(csvfile)
    for user in user_list:
        if user not in user_map:
            continue
        item_list = users[user]
        # if len(item_list)<filter_len:
        #     continue
        item_list.sort(key=lambda x: x[2])  # 按时间排序
        for item, cate, timestamp in item_list:
            writer.writerow([user_map[user], item_map[item], cate, timestamp])
            total_data += 1
    return total_data


def counter_item(user_list, users,user_map):
    print("calculate_scale")
    time_maxInterval=0
    time_list=[]
    for user in user_list:
        if user not in user_map:
            continue
        item_list = users[user]
        item_list.sort(key=lambda x: x[2])  # 按时间排序
        time_interval=item_list[-1][-1]-item_list[0][-1]   ##计算user的时间间隔
        time_list.append(time_interval)

        if time_interval > time_maxInterval:
            time_maxInterval = time_interval
            start=item_list[0][-1]
            end=item_list[-1][-1]
    print("the time_maxInterval: " + str(time_maxInterval))
    print(start)
    print(end)

    return time_maxInterval





def main():
    global filter_size
    path = '../data/' + name + '_data/'
    if not os.path.exists(path):
        os.mkdir(path)

    if name == 'book':
        item_count, users = read_from_amazon('../data/book_data/merged_book_data.json')
    elif name == 'taobao':
        item_count, users = read_from_taobao('/home/chentianzeng/jx/TDPP/data/taobao_data/UserBehavior.csv')
        print("filter_size" + str(filter_size))

    func(item_count, users, path)

# def counter_taobao(item_count, users, path):
#     print("calculate_scale for taobao")
#     user_map = {}
#     csv_file = csv.reader(open("../data/taobao_data/UserBehavior_5_50.csv", 'r'))
#     for line in csv_file:
#         # line ['1', '2266567', '4145813', 'pv', '1511741471']
#         # cur_dict = {'uid': line[0], 'iid': line[1], 'cid': line[2], 'btag': str(line[3]), 'timestamp': int(line[4])}
#
#         if int(line[-1]) > 1512571193 or int(line[-1]) < 1490053988:
#             continue
#         if line[0] not in user_map:
#             user_map[line[0]] = []
#         cur_list = []
#         cur_list.append(line[0])  # user id
#         # cur_list.append(line[1])  # item id
#         user_map[line[0]].append(int(line[-1]))  # 按user建立访问序列
#         # item_list.append(line[1])  # 出现过的item
#
#     item_cnt = []
#     max_scale = 0
#     print("merged_book_data handling")
#     print(len(user_map))
#     for key in user_map:
#         sorted_user_bh = sorted(user_map[key])  # 同一个user的不同item按出现时间排序
#         if len(sorted_user_bh) == 0:
#             continue
#         time_interval = sorted_user_bh[len(sorted_user_bh) - 1] - sorted_user_bh[0]
#         if time_interval > max_scale:
#             max_scale = time_interval
#             max_id = key
#             start = sorted_user_bh[0]
#             end = sorted_user_bh[len(sorted_user_bh) - 1]
#     print(max_scale)  # 572659200 tao bao:22179163
#     print(user_map[max_id])
#     print(start)  # 1996-05-20 08:00:00  tao bao:1490053988
#     print(end)  # 2014-07-13 08:00:00  tao bao:1512233151


# 把不对的时间筛掉
# def to_csv(RAW_DATA_FILE):
#     df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid', 'btag', 'time'])
#     # df = df.sample(frac=FRAC, replace=False, random_state=0)
#     time_df = df.sort_values(['time'])
#     # 时间位数不是13的会报错
#     time = time_df[783:df.shape[0] - 1500]
#     time.to_csv('../data/taobao_data/UserBehaviorTime.csv',header=None)



if __name__ == '__main__':
    main()