# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 12:02:15 2021

@author: fanxu
"""
import torch 
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    
    def __init__(self, inpDim, hidDim, decDim):
        super(Encoder, self).__init__()
        
        self.rnn = nn.GRU(inpDim, hidDim, bidirectional = True)
        self.fc = nn.Linear(hidDim * 2, decDim)

    def forward(self, inputs):
        """前向传播过程
        
        function：
            output, hidSta = self.rnn(inputs)
            args：
                inputs : [sque_len, batch_size, feature_num]
            return:
                output: [sque_len, batch_size, hidDim * 2]
                hidSta: [sque_len, barch_size, hidDim * 2]
             encSta = self.fc(**args)
             args:
                 hidSta 最后一个隐含层输出，双向所以为torch.cat(-1,-2)
             return:
                 encSta: [batch_size, decDim]
        """
        inputs = inputs.transpose(0,1)
        encOut, hidSta = self.rnn(inputs)
        
        encSta = self.fc(torch.cat((hidSta[-2,:,:], hidSta[-1,:,:]), dim = 1))
        ### torch.cat 将两个tensor合并，dim 为合并的维度 ###

        return encOut, encSta

class Attention(nn.Module):
    
    def __init__(self, hidDim, decDim):
        super(Attention, self).__init__()
        
        self.att_fc = nn.Linear(hidDim*2 + decDim, decDim)
        self.eng_fc = nn.Linear(decDim, 1, bias = False)
        
    def forward(self, encOut, encSta):
        """Attention 前向传播过程
        args：
            输入数据为 encoder部分的输出
        return：
            输出的为 每一个时次的贡献
        """
        
        batSiz = encOut.shape[0]
        squLen = encOut.shape[1]
        
        encSta = encSta.unsqueeze(1).repeat(1, squLen, 1)
        ### tensor.unsqueeze 取消掉encSta的第二个维度 ###
        ### tensor.repeat 在第二维度上将encSta重复squLen次 ###
        energy = torch.tanh(self.att_fc(
                    torch.cat((encSta, encOut), dim = 2)))
        att = self.eng_fc(energy).squeeze(2)
        
        return F.softmax(att, dim=1)
    

class Decoder(nn.Module):
    def __init__(self, outDim, inpDim, endHidDim, decHidDim, attention):
        super(Decoder, self).__init__()
        self.outDim = outDim
        self.attention = attention
        self.rnn = nn.GRU((endHidDim * 2) + inpDim, decHidDim)
        self.fc_out = nn.Linear((endHidDim * 2) + decHidDim + inpDim, outDim)
        
    def forward(self, decInp, encOut, encSta):
        """Decoder 前向传播过程
        args：
            decInp: 解码层的输入
                [batch_size, sque_len, feature_num]
            encOut, encSta: 与attenion输入一致
                encOut : [batch_size, src_len, enc_hid_dim * 2]
                encSta : [batch_size, dec_hid_dim]
        function：
            self.attention(encOut, encSta) 调用的为上方定义的Attention
            return：
                att: [batch_size, sque_len]   
        """
        decInp = decInp.unsqueeze(1)
        # encOut = [batch_size, src_len, enc_hid_dim * 2]
        encOut = encOut.transpose(0, 1)

        att = self.attention(encOut, encSta).unsqueeze(1)
        
        # cm = [1, batch_size, enc_hid_dim * 2]
        cm = torch.bmm(att, encOut)
        ### toech.bmm 为tensor的对应相乘 ###
        ### 利用attention对输入的每一个样本赋予权重 ###

        rnnInp = torch.cat((decInp, cm), dim = 2).transpose(0,1)
        decOut, decHid = self.rnn(rnnInp,encSta.unsqueeze(0))
        decOut = decOut.squeeze(0)
        cm = cm.transpose(0, 1).squeeze(0)
        decInp = decInp.squeeze(1)
        pred = self.fc_out(torch.cat((decOut, cm, decInp), dim = 1))
        return pred, decHid.squeeze(0)
        
class sque2sque(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(sque2sque, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, inputs, outputs):
        
        encOut, encSta = self.encoder(inputs)
        batSiz = inputs.shape[0]
        outLen = outputs.shape[1]
        outDim = self.decoder.outDim

        decInput = torch.zeros(batSiz, outLen, outDim)
        dec_input = decInput[:,0,:]
        for t in range(1, outLen):
            
            # insert dec_input token embedding, previous hidden state and all encoder hidden states
            # receive output tensor (predictions) and new hidden state
            decOut, decSta = self.decoder(dec_input, encOut, encSta)
            outputs[:,t] = decOut
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = dec_output.argmax(1)
            
            dec_input = outputs[t-1] if teacher_force else top1
        
        return outputs



        
        
        