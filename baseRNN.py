# 1. 利用RNNCell，需要手动循环
import torch

# seq_len = 3#x1,x2,x3
# batch_size = 2
# input_size = 4#每个x都是4个元素的向量
# hidden_size = 2#hidden都是有2个元素的向量
#
# cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
# #input.shape=(batchsize,inputsize)
# #output.shape=(batchsoze,hiddensize)
#
# dataset = torch.randn(seq_len, batch_size, input_size)
# hidden = torch.zeros(batch_size, hidden_size)
#
# for idx, inputs in enumerate(dataset):  # idx==seq_len
#     print('=' * 10, idx, '=' * 10)
#     hidden = cell(inputs, hidden)
#     print(inputs)
#     print(inputs.shape)
#
#     print(hidden.shape)


batch_size = 1#
seq_len = 3#样本长度（样本个数）
input_size = 4#一个样本的特征个数
hidden_size = 2
num_layers = 1

cell = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

inputs = torch.randn(seq_len,batch_size,input_size)
hidden = torch.zeros(num_layers,batch_size,hidden_size)

out, hidden =cell(inputs,hidden)
print('Output size',out.shape)
print('Output',out)
print('Hidden size',hidden.shape)
print('Hidden',hidden)