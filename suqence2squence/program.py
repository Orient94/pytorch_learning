# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 18:38:55 2021

@author: fanxu
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


def read_file(filename):
    
    data = pd.read_excel(filename)
    data = data.set_index('日期')
    return data
    

def in_out_creat(Arr, windowSize, preStep):
    
    inpArr = []
    outArr = []
    for i in range(len(Arr)-preStep-windowSize):
        
        iinp = Arr[i:i+windowSize]
        iout = Arr[i+windowSize+preStep]
        inpArr.append(iinp)
        outArr.append(iout)
    return np.array(inpArr), np.array(outArr)
    
    
def model_define():
    model = lgb.LGBMRegressor(objective='regression',
                        num_leaves=10,
                        learning_rate=0.05,
                        n_estimators=150)
    return model

def model_train(model, inpArr, outArr):
    model.fit(inpArr, outArr)
    return model
    



if __name__ == '__main__':
    
    filename = './power.xlsx'
    dat = read_file(filename)
    data = dat
    arr = data.iloc[:-1,:].values.flatten()
    inpArr, outArr = in_out_creat(arr, windowSize=48, preStep=16)
    model = model_define()
    model = model_train(model, inpArr, outArr)
    testArr = dat.iloc[-2:,:].values.flatten()
    inptesArr, outtesArr = in_out_creat(testArr, windowSize=48, preStep=16)
    outPre = model.predict(inptesArr)
    plt.plot(outtesArr)
    plt.plot(outPre)
    
    
    
    