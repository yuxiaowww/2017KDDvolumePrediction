# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:15:17 2017

@author: yuwei
"""

import pandas as pd
import xgboost as xgb



def model_xgb(file_train,file_test):
    train = pd.read_csv(file_train)
    test = pd.read_csv(file_test)
    train_y = train['volume'].values
    
    train_x = train.drop([
    'tollgate_id','direction','time_window','volume','start_time','end_time'],axis=1).values
    test_x = test.drop([
    'tollgate_id','direction','time_window','volume','start_time','end_time'],axis=1).values
    
    dtrain = xgb.DMatrix(train_x, label=train_y)
    dtest = xgb.DMatrix(test_x)
    
    # 模型参数
    params = {'booster': 'gbtree',
              'objective': 'reg:linear',
              #'objective':'count:poisson',
              'eta': 0.03,
              'max_depth': 5,  # 6
              'colsample_bytree': 0.8,#0.8
              'subsample': 0.8,
              #'lambda':300,
              #'scale_pos_weight': 1,
              'min_child_weight': 18  # 2
              }
    # 训练
    bst = xgb.train(params, dtrain, num_boost_round=1500)
    # 预测
    predict = bst.predict(dtest)
    test_xy = test[['tollgate_id','time_window','direction']]
    test_xy['prob'] = predict
    predict = pd.DataFrame(predict)
    predicted = pd.concat([test_xy[['tollgate_id','time_window','direction']], predict], axis=1)
    return predicted
    
    
if __name__ == '__main__':
        file_train = 'train.csv'
        file_test = 'test.csv'
        predict = model_xgb(file_train,file_test)
        predict.rename(columns={0:'volume'},inplace=True)
        print(predict['volume'].mean())
        predict.to_csv('predict.csv',index=False)

        

        
        