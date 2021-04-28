# encoding: utf-8
'''
时间戳和日期之间的转换
'''
from datetime import datetime
import datetime as dt
import time
import os

scale= 572659200
taobao_scale= 22179163
amazon_scale= 539740800


def timestamp_datetime(ts):
    if isinstance(ts, (int, float, str)):
        try:
            ts = int(ts)
        except ValueError:
            raise

        if len(str(ts)) == 13:
            ts = int(ts / 1000)
        if len(str(ts)) != 10:
            raise ValueError
    else:
        raise ValueError()

    return datetime.fromtimestamp(ts)
##有两种策略，一种是利用全部user的最大时间间隔，另一种是一个用户一个scale
def inteval(t,s,scale):  #s比t大
    d=s-t
    if d==0:
        d=1
    d=d/scale
    return d




def datetime_timestamp(dt, type='ms'):
    if isinstance(dt, str):
        try:
            if len(dt) == 10:
                dt = datetime.strptime(dt.replace('/', '-'), '%Y-%m-%d')
            elif len(dt) == 19:
                dt = datetime.strptime(dt.replace('/', '-'), '%Y-%m-%d %H:%M:%S')
            else:
                raise ValueError()
        except ValueError as e:
            raise ValueError(
                "{0} is not supported datetime format." \
                "dt Format example: 'yyyy-mm-dd' or yyyy-mm-dd HH:MM:SS".format(dt)
            )

    if isinstance(dt, time.struct_time):
        dt = datetime.strptime(time.stftime('%Y-%m-%d %H:%M:%S', dt), '%Y-%m-%d %H:%M:%S')

    if isinstance(dt, datetime):
        if type == 'ms':
            ts = int(dt.timestamp()) * 1000
        else:
            ts = int(dt.timestamp())
    else:
        raise ValueError(
            "dt type not supported. dt Format example: 'yyyy-mm-dd' or yyyy-mm-dd HH:MM:SS"
        )
    return ts


if __name__ == '__main__':
    try:
        '''
        test=datetime_timestamp('2015-01-01 20:20:00', 's')
        a=timestamp_datetime(1483574401)  #2017-01-05 08:00:01
        b=timestamp_datetime(1420114800123)
        c = "2020-12-08 11:30:00"
        #实现两个时间的相减
        c2 = datetime.strptime(str(c), "%Y-%m-%d %H:%M:%S")
        a2=datetime.strptime(str(a),"%Y-%m-%d %H:%M:%S")

        d1=c2-a2
        d2 = c2 - dt.timedelta(seconds=900)
        d2.ctime()  #转换为年月日
        '''
#2017年11月25日至12月3日
        m=timestamp_datetime(1512571193)   #2017-12-06 22:39:53  第len-1500
        n=timestamp_datetime(1490053988)  #2017-03-21 07:53:08   第783个
        a=inteval(1511544070,1512281847)  #11285205
        #d=inteval(1512571193,1490053988)
        #8位：1157天超过三天
        print('over')
    except Exception as e:
        print(e.args[0])
    
  