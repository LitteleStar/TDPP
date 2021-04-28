import sys
import random
import time
import os
import json
import csv
"""
输出jointed-new-split-info
date 正负样本0/1标识 userid itemid rate time 字符串格式cate
"""


# max book scale 572659200 in seconds
# max book scale 572659200/86400 in days
def calculate_scale_book():
    print("calculate_scale for Books")
    f_rev = open("reviews-info", "r")  # UserID + ItemID + Timestamp
    user_map = {}
    item_list = []

    print("reviews-info  handling")
    for line in f_rev:  # 读入评论 找出其中的
        line = line.strip()
        items = line.split("\t")
        if items[0] not in user_map:
            user_map[items[0]] = []
        user_map[items[0]].append(("\t".join(items), float(items[-1])))  # 按user建立访问序列
        item_list.append(items[1])  # 出现过的item

    # 生成item - cate 的map

    print("item-info handling")
    f_meta = open("item-info", "r")  # ItemID + 最小分类
    meta_map = {}
    for line in f_meta:
        arr = line.strip().split("\t")
        if arr[0] not in meta_map:
            meta_map[arr[0]] = arr[1]
    max_scale = 0

    item_cnt = []
    print("merged_book_data handling")
    print(len(user_map))
    for key in user_map:
        sorted_user_bh = sorted(user_map[key], key=lambda x: x[1])  # 同一个user的不同item按出现时间排序
        time_interval = sorted_user_bh[len(sorted_user_bh)-1][1] - sorted_user_bh[0][1]
        item_cnt.append(len(sorted_user_bh)-1)
        if time_interval > max_scale:
            max_scale = time_interval
            max_id = key
            start = sorted_user_bh[0][1]
            end = sorted_user_bh[len(sorted_user_bh)-1][1]
    print(max_scale)  # 572659200
    print(user_map[max_id])
    print(len(sorted_user_bh)-1)
    print(start)  # 1996-05-20 08:00:00
    print(end)    # 2014-07-13 08:00:00


def data_merge():
    file = "../data/book_data/meta_Books.json"
    process_meta(file)

    file = "../data/book_data/reviews_Books_5.json"
    process_reviews(file)

    manual_join()


def process_meta(file):
    """
    :param file: 读入meta文件
    :return: ItemID + 最小分类
    """
    print("process_meta")
    fi = open(file, "r")
    fo = open("item-info", "w")
    for line in fi:
        obj = eval(line)  # 以一行记录建立一个对象
        cat = obj["categories"][0][-1]  # 读取最小分类
        print(obj["asin"] + "\t" + cat, file = fo)


def process_reviews(file):
    """
    :param file: 读入review文件
    :return: UserID + ItemID + Timestamp
    """
    print("process_reviews")
    fi = open(file, "r")
    user_map = {}
    fo = open("reviews-info", "w")
    for line in fi:
        obj = eval(line) # 以一行记录建立一个对象
        userID = obj["reviewerID"]
        itemID = obj["asin"]
        # rating = obj["overall"]
        time = obj["unixReviewTime"]
        # print(userID + "\t" + itemID + "\t" + str(rating) + "\t" + str(time), file = fo)
        print(userID + "\t" + itemID  + "\t" + str(time), file = fo)


def manual_join():
    """
    :return:  UserID ItemID CateID/cate_string timestamp
    in ../data/book_data/merged_book_data.json
    """
    print("manual_join")
    f_rev = open("reviews-info", "r")  # UserID + ItemID + Timestamp
    user_map = {}
    item_list = []
    for line in f_rev:  # 读入评论 找出其中的
        line = line.strip()
        items = line.split("\t")
        if items[0] not in user_map:
            user_map[items[0]] = []
        user_map[items[0]].append(("\t".join(items), float(items[-1])))  # 按user建立访问序列
        item_list.append(items[1])  # 出现过的item

    # 生成item - cate 的map
    print("generating item2cate map")
    f_meta = open("item-info", "r")  # ItemID + 最小分类
    meta_map = {}
    for line in f_meta:
        arr = line.strip().split("\t")
        if arr[0] not in meta_map:
            meta_map[arr[0]] = arr[1]

    print("data merging")
    with open("../data/book_data/merged_book_data.json", "w") as f:
        for key in user_map:
            sorted_user_bh = sorted(user_map[key], key=lambda x:x[1])  # 同一个user的不同item按出现时间排序
            for line, t in sorted_user_bh:  # line : user item time
                items = line.split("\t")
                asin = items[1]
                if asin in meta_map:
                    cate = meta_map[asin]
                else:
                    cate = None
                cur_dict = {'user_id': items[0], 'item_id': items[1], 'cate': cate, 'timestamp': int(items[2])}
                json.dump(cur_dict, f)
                f.write('\n')
# tic 21:34 21:36

# done
# process_meta("../data/book_data/meta_Books.json")
# process_reviews("../data/book_data/reviews_Books_5.json")
manual_join()
#calculate_scale_book()