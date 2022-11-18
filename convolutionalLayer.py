import torch
# in_channels,out_channels=5,10
# width,height=100,100 #图像的长宽
# kernel_size = 3#j卷积核的大小
# batch_size = 1
#
# input = torch.randn(batch_size,in_channels,width,height)
# conv_layer = torch.nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size)
# #
# output = conv_layer(input)
#
# print(input.shape)
# print(output.shape)
# print(conv_layer.weight.shape)

input = [3,4,6,5,7,
         2,4,6,8,2,
         1,6,7,8,4,
         9,7,4,6,2,
         3,7,5,4,1]
input = torch.Tensor(input).view(1,1,5,5)#Batch,C,H,W
#conv_play =torch.nn.Conv2d(1,1,kernel_size=3,padding=1,bias=False)
#第一个1：输入一个通道，第二个1：输出1个通道
conv_play =torch.nn.Conv2d(1,1,kernel_size=3,stride=2,bias=False)
kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3)
#输出的通道数，输入的通道数，宽度，高度
conv_play.weight.data = kernel.data
#卷积层的权重
output = conv_play(input)
print(output)