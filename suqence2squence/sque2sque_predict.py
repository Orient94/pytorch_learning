# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 11:21:25 2021

@author: fanxu
"""
import torch
import matplotlib.pyplot as plt
from sque2sque import sque2sque, Encoder, Decoder
from sque2sque_train import read_file,in_out_creat, model_define

def get_torh_modl(modelOri, filename):
    modelOri.load_state_dict(torch.load(filename))
    return modelOri



def model_predict(model, inputs, outLen = 16):
    
    encOut, encSta = model.encoder(inputs)
    
    sample = inputs.shape[0]
    decInput = torch.zeros(sample, outLen, outDim)
    
    for t in range(1, outLen):
            
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            dec_input = decInput[:, t-1, :]
            decOut, decSta = model.decoder(dec_input, encOut, encSta)
            decInput[:,t,:] = decOut 
    return decInput

def tensor_creat(x, y):

    xTen = torch.from_numpy(x).to(torch.float32)
    yTen = torch.from_numpy(y).to(torch.float32)
    return xTen.unsqueeze(2),yTen.unsqueeze(2)



if __name__ == '__main__':

    filename = './power.csv'
    dat = read_file(filename)
    data = dat
    arr = data.iloc[-2:,:].values.flatten()
    inpArr, outArr = in_out_creat(arr, windowSize=48, preStep=16)
    xTen, yTen = tensor_creat(inpArr, outArr)
    inpDim = 1
    outDim = 1
    endHidDim = 5
    decHidDim = 5
    model = model_define(inpDim, outDim, endHidDim, decHidDim)
    filename = './model_torch.mdl'
    model = get_torh_modl(model, filename)
    outputs = model_predict(model, xTen)
    out16 = outputs[:,-1,:].detach().numpy()
    plt.plot(out16[-96:], label='predict')
    plt.plot(outArr[-96:,-1], label='Origin')
    plt.legend()
    plt.savefig('./torch_compare.png') 
     
    
    
