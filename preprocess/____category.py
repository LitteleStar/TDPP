##item 对应category
import pandas as pd

RAW_DATA_FILE = '../data/taobao_data/UserBehavior_120_200.csv'
Dpp_Scale = 10
MAX_LEN_ITEM = 200


# FRAC=0.01   #3140  31399 313995 3139947
def to_df(file_name):
    df = pd.read_csv(RAW_DATA_FILE, header=None, names=['uid', 'iid', 'cid',  'time'])  # 100150807   ###10000000--55555555是可以运行的
    #df = df.sample(frac=0.1, replace=False, random_state=0)  ##这样不是顺序取样，不可取
    return df


# 将数据属性统一编码为0-len
def remap_item_type(df):
    # 处理 item id，编号为0---len(id)
    item_key = sorted(df['iid'].unique().tolist())  # [a,b,c]
    item_len = len(item_key)
    item_map = dict(zip(item_key, range(item_len)))  # (0:243),(1,233)
    df['iid'] = df['iid'].map(lambda x: item_map[x])  # 给用户id替换位对应的编号

    # 处理用户id，编号为
    user_key = sorted(df['uid'].unique().tolist())
    user_len = len(user_key)
    user_map = dict(zip(user_key, range(user_len)))
    df['uid'] = df['uid'].map(lambda x: user_map[x])

    # 处理类别id，编号为
    cate_key = sorted(df['cid'].unique().tolist())
    cate_len = len(cate_key)
    cate_map = dict(zip(cate_key, range(cate_len)))
    df['cid'] = df['cid'].map(lambda x: cate_map[x])

    print("sort finish")
    return df, item_len, cate_len

def item_cate(df):
    #item_ca=df.groupby('iid')['cid'].apply(set).to_dict()
    item_ca = {}
    for index,row in df.iterrows():
        cid=row[2]
        iid=row[1]
        item_ca[iid]=cid
    write_('taobao',item_ca)

'''
name = 'taobao'
if len(sys.argv) > 1:
    name = sys.argv[1]
with open('../data/%s_data/%s_item_map.txt' % (name, name), 'r') as f:
    for line in f:
        conts = line.strip().split(',')
        item_map[conts[0]] = conts[1]
'''

def write_(name,map):
    with open('../data/%s_data/%s_item_cate.txt' % (name, name), 'w') as f:
        for key, value in map.items():
            f.write('%s,%s\n' % (key, value))


def load_item_cate(item_map,name='taobao'):
    item_cate = {}
    cate_map = {}
    if name == 'taobao':
        with open('../data/taobao_data/UserBehavior_120_200.csv', 'r') as f:
            for line in f:
                conts = line.strip().split(',')
                iid = int(conts[1]) ##这是str类型
                # if conts[3] != 'pv':
                #     continue
                cid = int(conts[2])
                if iid in item_map:
                    if cid not in cate_map:
                        cate_map[cid] = len(cate_map) + 1
                    #item_cate[item_map[iid]] = cate_map[cid]  ##编号对应
                    item_cate[iid]=cid  ##未编号的对应
    elif name == 'book':
        with open('meta_Books.json', 'r') as f:
            for line in f:
                r = eval(line.strip())
                iid = r['asin']
                cates = r['categories']
                if iid not in item_map:
                    continue
                cate = cates[0][-1]
                if cate not in cate_map:
                    cate_map[cate] = len(cate_map) + 1
                item_cate[item_map[iid]] = cate_map[cate]
    #write_(name,cate_map)
    write_(name,item_cate)

if __name__ == '__main__':
    # with open('../data/taobao_data/item_map.pickle', 'rb') as handle:
    #     item_map = pickle.load(handle)
    # load_item_cate(item_map,name='taobao')
    df = to_df(RAW_DATA_FILE)  # 加载数据集到df
    df, item_cnt, feature_size = remap_item_type(df)  # 将数据属性统一编码为0-len到df，item个数，所有特征大小
    item_cate(df)