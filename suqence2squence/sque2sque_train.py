# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 14:03:26 2021

@author: fanxu
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from sque2sque import Encoder, Decoder, sque2sque, Attention

def train(model, dataLoader, optimizer, criterion):
    model.train()    
    epoch_loss = 0
    for i, (x, y) in enumerate(dataLoader):
        tx, ty = x, y

        # pred = [trg_len, batch_size, pred_dim]
        pred = model(x, y)
        
        pred_dim = pred.shape[-1]
        
        trg = trg[1:].view(-1)
        pred = pred[1:].view(-1, pred_dim)
        loss = criterion(pred, trg)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def read_file(filename):
    
    data = pd.read_excel(filename)
    data = data.set_index('日期')
    return data.astype(np.float32)
    
def in_out_creat(Arr, windowSize, preStep):
    
    inpArr = []
    outArr = []
    for i in range(len(Arr)-preStep-windowSize):
        
        iinp = Arr[i:i+windowSize]
        iout = Arr[i+windowSize+1: i+windowSize+preStep+1]
        inpArr.append(iinp)
        outArr.append(iout)
    return np.array(inpArr), np.array(outArr)

def dataLoader_creat(x, y):
    
    xTen = torch.from_numpy(x).to(torch.float32)
    yTen = torch.from_numpy(y).to(torch.float32)
    dataSet = Data.TensorDataset(
                xTen.unsqueeze(2), 
                yTen.unsqueeze(2))
    return Data.DataLoader(
                dataset=dataSet, batch_size=64)

def model_define(inpDim, outDim, endHidDim, decHidDim):
    attention = Attention(endHidDim, decHidDim)
    enc = Encoder(inpDim, endHidDim, decHidDim)
    dec = Decoder(outDim, inpDim, endHidDim, decHidDim, attention)
    model = sque2sque(enc, dec)
    return model


def save_torch(model, filename):
    ''' 保存模型 '''
    torch.save(model.state_dict(), filename)

if __name__ == "__main__":
    
    
    filename = './power.xlsx'
    dat = read_file(filename)
    data = dat
    arr = data.iloc[:-1,:].values.flatten()
    inpArr, outArr = in_out_creat(arr, windowSize=48, preStep=16)
    trainLoader = dataLoader_creat(inpArr, outArr)
    import pdb;pdb.set_trace()
    inpDim = 1
    outDim = 1
    endHidDim = 5
    decHidDim = 5
    model = model_define(inpDim, outDim, endHidDim, decHidDim)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    lossfunction = nn.MSELoss()
    train(model, trainLoader, optimizer, lossfunction)
    filename = './model_torch.mdl'
    save_torch(model, filename)
    
    
    
    

    
    
    
    
