# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 11:21:25 2021

@author: fanxu
"""
import troch
from sque2sque import sque2sque

def get_torh_modl(modelOri, filename):
    modelOri.load_state_dict(torch.load(filename))
    return modelOri



def model_predict(model, inputs):
    
    encOut, encSta = model.encoder(inputs)
    
    
    decInput = torch.zeros(1, 1, outDim)
    
    for t in range(1, outLen):
            
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            dec_input = decInput[:, t-1, :]
            decOut, decSta = model.decoder(dec_input, encOut, encSta)
            
            decInput = torch.cat(decInput, decOut)
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
    model = sque2sque()
    filename = './model_torch.mdl'
    model = get_torh_modl(model, filename)
    outputs = model_predict(model, inputs)
    
    
    
    