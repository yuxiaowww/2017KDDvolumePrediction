# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:18:18 2017

@author: yuwei
"""

import pandas as pd
import datetime as dt
import numpy as np

train_file1 = 'training_20min_avg_volume.csv'
train_file2 = 'training2_20min_avg_volume.csv'
test_file = 'test2_20min_avg_volume.csv'
weather1 = 'weather (table 7)_training_update.csv'
weather2 = 'weather (table 7)_test1.csv'
weather3 = 'weather (table 7)_2.csv'

'''
按tollgate_id和direction五个类型，划分为5个数据集
'''
def loadData():
    # 读取csv文件
    data1 = pd.read_csv(train_file1)
    data2 = pd.read_csv(train_file2)
    data3 = pd.read_csv(test_file)
    data = pd.concat([data1,data2,data3],ignore_index = True)
    # 分解time_window
    data['start_time'] = 0;data['end_time'] = 0
    data.start_time = data.time_window.map(lambda x : dt.datetime.strptime(x.replace('[','').replace(')','').split(',')[0],"%Y-%m-%d %H:%M:%S"))
    data.end_time = data.time_window.map(lambda x : dt.datetime.strptime(x.replace('[','').replace(')','').split(',')[1],"%Y-%m-%d %H:%M:%S"))
    # 按收费站和进出分为5个数据集       
    data1 = data[(data.tollgate_id == 1) & (data.direction == 0)];data1.index = range(0,len(data1))
    data2 = data[(data.tollgate_id == 1) & (data.direction == 1)];data2.index = range(0,len(data2))
    data3 = data[(data.tollgate_id == 2) & (data.direction == 0)];data3.index = range(0,len(data3))
    data4 = data[(data.tollgate_id == 3) & (data.direction == 0)];data4.index = range(0,len(data4))
    data5 = data[(data.tollgate_id == 3) & (data.direction == 1)];data5.index = range(0,len(data5))
    # 返回5个数据集
    return data1,data2,data3,data4,data5

'''
一共补齐的窗口组成是：9月19日到10月30日，以及10月31日到19点的窗口，(35*24*3+19*3)=2557个窗口
'''    
def completeData(data):
    '''对于训练集'''
    # 训练部分窗口个数
    train_total_days = dt.datetime(2016,10,24) - dt.datetime(2016,9,19) + dt.timedelta(days = 1) #总计42天
    train_total_minutes = train_total_days.total_seconds()/60 #总计的分钟数
    train_total_window_cnt = int(train_total_minutes/20) #每20分钟一个窗口，总共的窗口
    # 补全训练部分数据集
    for i in range(train_total_window_cnt):
        start_time = dt.datetime(2016,9,19) + dt.timedelta(minutes = i * 20)
        end_time = start_time + dt.timedelta(minutes = 20)
        #转换起始时间为字符串，写入标准的time_window格式
        time_window = '['+ start_time.strftime("%Y-%m-%d %H:%M:%S") + ',' + end_time.strftime("%Y-%m-%d %H:%M:%S") + ')'
        #假如时间窗口不存在，则挨个补充
        if time_window not in list(data.time_window):
            #取得哪个收费站哪个方向，流量置0
            tollgate_id = data.loc[0,'tollgate_id'];direction = data.loc[0,'direction'];volume = 0
            add = pd.DataFrame({'1':[tollgate_id],'2':[time_window],'3':[direction],'4':[volume],'5':[start_time],'6':[end_time]})
            add.columns = ['tollgate_id','time_window','direction','volume','start_time','end_time']
            data = pd.concat([data,add],ignore_index = True)
            
    '''对于测试集'''        
    # 测试部分窗口个数
    test_total_days = dt.datetime(2016,10,31) - dt.datetime(2016,10,25) + dt.timedelta(hours = 19) #取到10月18日0点到10月24日19点
    test_total_minutes = test_total_days.total_seconds()/60
    test_total_window_cnt = int(test_total_minutes/20)
    # 测试训练部分数据集
    for i in range(train_total_window_cnt + test_total_window_cnt):
        start_time = dt.datetime(2016,9,19) + dt.timedelta(minutes = i * 20)
        end_time = start_time + dt.timedelta(minutes = 20)
        time_window = '['+ start_time.strftime("%Y-%m-%d %H:%M:%S") + ',' + end_time.strftime("%Y-%m-%d %H:%M:%S") + ')'
        if time_window not in list(data.time_window):
            tollgate_id = data.loc[0,'tollgate_id'];direction = data.loc[0,'direction'];volume = np.nan
            add = pd.DataFrame({'1':[tollgate_id],'2':[time_window],'3':[direction],'4':[volume],'5':[start_time],'6':[end_time]})
            add.columns = ['tollgate_id','time_window','direction','volume','start_time','end_time']
            data = pd.concat([data,add],ignore_index = True)

    '''定义测试集'''
    # 为了方便分割训练集和测试集，取了日期，小时打标记
    data['start_month_day'] = 0;data['keep_volume'] = np.nan
    data.start_month_day = data.start_time.map(lambda x : str(x.month) + '-' + str(x.day))
    train = data[(data.start_month_day == '9-19') | (data.start_month_day == '9-20') |
                (data.start_month_day == '9-21') | (data.start_month_day == '9-22')|
                (data.start_month_day == '9-23') | (data.start_month_day == '9-24')|
                (data.start_month_day == '9-25') | (data.start_month_day == '9-26')|
                (data.start_month_day == '9-27') | (data.start_month_day == '9-28')|
                (data.start_month_day == '9-29') | (data.start_month_day == '9-30')|
                (data.start_month_day == '10-1') | (data.start_month_day == '10-2')|
                (data.start_month_day == '10-3') | (data.start_month_day == '10-4')|
                (data.start_month_day == '10-5') | (data.start_month_day == '10-6')|
                (data.start_month_day == '10-7') | (data.start_month_day == '10-8')|
                (data.start_month_day == '10-9') | (data.start_month_day == '10-10')|
                (data.start_month_day == '10-11') | (data.start_month_day == '10-12')|
                (data.start_month_day == '10-13') | (data.start_month_day == '10-14')|
                (data.start_month_day == '10-15') | (data.start_month_day == '10-16')|
                (data.start_month_day == '10-17') | (data.start_month_day == '10-18')|
                (data.start_month_day == '10-19') | (data.start_month_day == '10-20')|
                (data.start_month_day == '10-21') | (data.start_month_day == '10-22')|
                (data.start_month_day == '10-23') | (data.start_month_day == '10-24')]
    validation = data[(data.start_month_day == '10-25') | (data.start_month_day == '10-26')|
                (data.start_month_day == '10-27') | (data.start_month_day == '10-28')|
                (data.start_month_day == '10-29') | (data.start_month_day == '10-30')|
                (data.start_month_day == '10-31')]
    validation.keep_volume = validation.start_time.map(lambda x : '1' if ((x.hour == 6) | (x.hour == 7) | (x.hour == 15) | (x.hour == 16)) else '0')
    # 合并
    data = pd.concat([train,validation])
    # 排序并重设index
    data.sort_values('time_window',ascending = True,inplace = True)
    data.index = range(0,len(data))
    # 处理验证集的volume
    volume = map(lambda x,y : np.nan if y == '0' else x,data.volume,data.keep_volume)
    volume = list(volume)
    volume = pd.DataFrame(volume,columns = ['volume'])
    data.volume = volume.volume.map(lambda x : x)
    # 删去无用信息
    del data['start_month_day'];del data['keep_volume']
    # 返回
    return data

#删除9月30日到10月7日
def removeNationalDay(data):
    # 删去9.30到10.7的数据
    data['date_start'] = 0;data['date_end'] = 0;
    data.date_start = data.start_time.map(lambda x : str(x.month) + '-' + str(x.day))
    data.date_end = data.end_time.map(lambda x : str(x.month) + '-' + str(x.day))
    data = data[(data.date_start != '9-30') & (data.date_start != '10-1') & (data.date_start != '10-2') & (data.date_start != '10-3') & (data.date_start != '10-4') & (data.date_start != '10-5') & (data.date_start != '10-6') & (data.date_start != '10-7')]
    data = data[(data.date_end != '10-1') & (data.date_end != '10-2') & (data.date_end != '10-3') & (data.date_end != '10-4') & (data.date_end != '10-5') & (data.date_end != '10-6') & (data.date_end != '10-7')]
    del data['date_start'];del data['date_end']
    # 排序并重设index
    data.sort_values('time_window',ascending = True,inplace = True)
    data.index = range(0,len(data))
    # 返回
    return data

'''
#计算前七天中，每天的总流量
def count_7days_volumes(data):
    for i in range(1,8):
        data['days_'+str(i)] = 0
    for j in range(1,8):
        for i in range(0,72):
            data['days_'+str(j)][(72*i):(72*(i+1))] = sum(data.shift(72*j)['volume'][(72*i):(72*(i+1))])
    return data
'''
    
def getLabelFeature(data):
    data['order_in_weekday'] = 0
    data.order_in_weekday = data.start_time.map(lambda x : x.weekday())
    # 星期几逆序
    data['invert_order_in_weekday'] = 0
    data.invert_order_in_weekday = data.order_in_weekday.map(lambda x : 6 - x)

    # 是否周末
    data['weekends'] = 0
    data.weekends = data.order_in_weekday.map(lambda x : 1 if (x == 5 or x == 6) else 0)
    # 第几小时
    data['hour'] = 0
    data.hour = data.start_time.map(lambda x : x.hour)
    # 第几个窗口
    data['order_in_window'] = 0
    data.order_in_window = data.start_time.map(lambda x : x.minute/20 + 1)
    # 窗口的工作日、休息日累计(上班/休息的第几天)
    data['accumulate'] = 1
    for i in data.index[:-1]:
        if (data.loc[i,'order_in_weekday'] in (0,1,2,3,4)) and (data.loc[i+1,'order_in_weekday'] in (0,1,2,3,4)):# 工作日
            if data.loc[i,'order_in_weekday'] != data.loc[i+1,'order_in_weekday']:
                data.loc[i+1,'accumulate'] += data.loc[i,'accumulate']
            else:
                data.loc[i+1,'accumulate'] = data.loc[i,'accumulate']
        elif (data.loc[i,'order_in_weekday'] in (5,6)) and (data.loc[i+1,'order_in_weekday'] in (5,6)):
            if data.loc[i,'order_in_weekday'] != data.loc[i+1,'order_in_weekday']:
                data.loc[i+1,'accumulate'] += data.loc[i,'accumulate']
            else:
                data.loc[i+1,'accumulate'] = data.loc[i,'accumulate']
    # 累积特征逆序（还有几天上班/休息）
    data['invert_accumulate'] = 0
    for i in data.index:
        if (data.loc[i,'order_in_weekday'] in (0,1,2,3,4)):
            data.loc[i,'invert_accumulate'] = 6 - data.loc[i,'accumulate']
        elif (data.loc[i,'order_in_weekday'] in (5,6)):
            data.loc[i,'invert_accumulate'] = 3 - data.loc[i,'accumulate']
    # 6-10，15-19，标1。其他时间标0.
    data['is6101519'] = 0
    data.is6101519 = data.start_time.map(lambda x : 1 if ((x.hour >= 6) & (x.hour < 10)) | ((x.hour >= 15) & (x.hour < 19)) else 0)
    # 6-10，15-19，标0。其他时间标1.
    data['invert_is6101519'] = 1
    data.invert_is6101519 = data.start_time.map(lambda x : 0 if ((x.hour >= 6) & (x.hour < 10)) | ((x.hour >= 15) & (x.hour < 19)) else 1)
    
    #一天72个窗口的计算，以及取反
    data['window_1days_count'] = 0
    data['window_1days_count_reverse'] = 0
    for i in range(0,2505):
        data.loc[i,'window_1days_count'] = i%72
        data.loc[i,'window_1days_count_reverse'] = 71 - data.loc[i,'window_1days_count']
    #计算距离18号有多少天，以及倒序
    data['interval_918'] = 0
    data['interval_918_reverse'] = 0
    data = data.reset_index()
    del data['index']
    time = dt.datetime(2016,9,18,0,0,0)
    for i in range(0,2505):
        data.loc[i,'interval_918'] = (data.loc[i,'start_time'] - time).days
    for i in range(792,2505):
        data.loc[i,'interval_918'] = data.loc[i,'interval_918'] - 8
    for i in range(0,2505):
        data.loc[i,'interval_918_reverse'] = 36 - data.loc[i,'interval_918']
    # 返回
    return data

'''    
def getWindowFeature(data):
    # 当天特征窗口
    for i in (1,2,3,4,5,6):
        data['volume_%d' % i] = np.nan
        data['volume_%d' % i] = data.shift(i).volume
    # 均值
    for i in range(2,7):
        data['volume_avg_%d' %i] = np.nan
        data['volume_avg_%d' %i] = data[list(range(19,19+i))].mean(axis = 1)
    # 方差
    for i in range(2,7):
        data['volume_var_%d' %i] = np.nan
        data['volume_var_%d' %i] = data[list(range(19,19+i))].var(axis = 1)
    # 中位数
    for i in range(2,7):
        data['volume_median_%d' %i] = np.nan
        data['volume_median_%d' %i] = data[list(range(19,19+i))].median(axis = 1)
    # 返回
    return data
'''
    

def disperse_feature(data):
    #先做离散化
    tollgate_df = pd.get_dummies(data['tollgate_id'],prefix = 'tollgate_id')
    direction_df = pd.get_dummies(data['direction'],prefix = 'direction')
    weekday_df = pd.get_dummies(data['order_in_weekday'],prefix = 'order_in_weekday')
    hour_df = pd.get_dummies(data['hour'],prefix = 'hour')
    window_df = pd.get_dummies(data['order_in_window'],prefix = 'order_in_window')
    #合并
    data = pd.concat([data,tollgate_df,direction_df,weekday_df,hour_df,window_df],axis=1)
    return data

'''
def weather_feature(data):
    # 读取csv文件
    data1 = pd.read_csv(weather1)
    data2 = pd.read_csv(weather2)
    data3 = pd.read_csv(weather3)
    data1 = data1[['date','hour','precipitation']]
    data2 = data2[['date','hour','precipitation']]
    data3 = data3[['date','hour','precipitation']]
    weather = pd.concat([data1,data2,data3],ignore_index = True)
    #补齐时间
    weather['start_time']=0
    weather.start_time = weather.date.map(lambda x :dt.datetime.strptime(x,"%Y-%m-%d"))
    for i in range(0,8):
        print(i)
        weather.loc[i,'start_time'] = weather.loc[i,'start_time']+dt.timedelta(hours = 3*i)
    for i in range(8,974):
        print(i)
        weather.loc[i,'start_time'] = weather.loc[i,'start_time']+dt.timedelta(hours = 3*((i)%8))
    del weather['date'];del weather['hour'];
    data_weather = pd.merge(data,weather,on=['start_time'],how='left')
    #天气线性插值
    data_weather = data_weather.apply(pd.Series.interpolate)
    data_weather = data_weather[['start_time','precipitation']]
    data_weather.drop_duplicates(subset=['start_time'], keep = 'first', inplace = True)
    data = pd.merge(data,data_weather,on=['start_time'],how='left')
    return data
'''
    
    
def getTrianTestSet(data):
    #data = data.fillna(method='bfill')
    #data = data.fillna(method='ffill')
    train = data[data.start_time < '2016-10-25 00:00:00']
    test = data[((data.start_time >= '2016-10-25 08:00:00') & (data.start_time <= '2016-10-25 09:40:00'))  |
                 ((data.start_time >= '2016-10-25 17:00:00') & (data.start_time <= '2016-10-25 18:40:00')) |
                 ((data.start_time >= '2016-10-26 08:00:00') & (data.start_time <= '2016-10-26 09:40:00')) |
                 ((data.start_time >= '2016-10-26 17:00:00') & (data.start_time <= '2016-10-26 18:40:00')) |
                 ((data.start_time >= '2016-10-27 08:00:00') & (data.start_time <= '2016-10-27 09:40:00')) |
                 ((data.start_time >= '2016-10-27 17:00:00') & (data.start_time <= '2016-10-27 18:40:00')) |
                 ((data.start_time >= '2016-10-28 08:00:00') & (data.start_time <= '2016-10-28 09:40:00')) |
                 ((data.start_time >= '2016-10-28 17:00:00') & (data.start_time <= '2016-10-28 18:40:00')) |
                 ((data.start_time >= '2016-10-29 08:00:00') & (data.start_time <= '2016-10-29 09:40:00')) |
                 ((data.start_time >= '2016-10-29 17:00:00') & (data.start_time <= '2016-10-29 18:40:00')) |
                 ((data.start_time >= '2016-10-30 08:00:00') & (data.start_time <= '2016-10-30 09:40:00')) |
                 ((data.start_time >= '2016-10-30 17:00:00') & (data.start_time <= '2016-10-30 18:40:00')) |
                 ((data.start_time >= '2016-10-31 08:00:00') & (data.start_time <= '2016-10-31 09:40:00')) |
                 ((data.start_time >= '2016-10-31 17:00:00') & (data.start_time <= '2016-10-31 18:40:00')) ]
    return train,test
   
if __name__ == '__main__':
    # data原始数据
    data1,data2,data3,data4,data5 = loadData()
#    #补全数据
#    data1 = completeData(data1);data2 = completeData(data2);data3 = completeData(data3);data4 = completeData(data4);data5 = completeData(data5)
#    # 去除国庆假期数据
#    data1 = removeNationalDay(data1);data2 = removeNationalDay(data2);data3 = removeNationalDay(data3);data4 = removeNationalDay(data4);data5 = removeNationalDay(data5)
#    # 标签的特征
#    data1 = getLabelFeature(data1);data2 = getLabelFeature(data2);data3 = getLabelFeature(data3);data4 = getLabelFeature(data4);data5 = getLabelFeature(data5);
#    # 滑窗特征
#    #data1 = getWindowFeature(data1);data2 = getWindowFeature(data2);data3 = getWindowFeature(data3);data4 = getWindowFeature(data4);data5 = getWindowFeature(data5)
#    # 计算前七天，每一天的总流量
#    #data1 = count_7days_volumes(data1);data2 = count_7days_volumes(data2);data3 = count_7days_volumes(data3);data4 = count_7days_volumes(data4);data5 = count_7days_volumes(data5)
#    #合并数据集
#    data = pd.concat([data1,data2,data3,data4,data5],ignore_index = True)
#    #降雨量线性插值
#    #data = weather_feature(data)
#    #离散化特征
#    data = disperse_feature(data)
#    train,test = getTrianTestSet(data)
#    train.to_csv('dataSetTrainTest/train.csv',index=False)
#    test.to_csv('dataSetTrainTest/test.csv',index=False)


    
    