import torch.nn
import matplotlib.pyplot as plt
x_data =torch.Tensor([[1.0], [2.0], [3.0]])
y_data =torch.Tensor([[2.0], [4.0], [6.0]])
#使用class设计模型
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()#继承父类的构造函数
        self.linear = torch.nn.Linear(1,1)#构造对象，包含权重我，偏置b
        #（1，1）表示输如和输出的维度
    #Linear 也继承自torch

    def forward(self, x):
        y_pred=self.linear(x)  #实现一个可调用的对象    __call__
        return y_pred

model =LinearModel()#实例化，model可以直接调用model（x）
#构建损失函数和优化器的选择
criterion = torch.nn.MSELoss(reduction='mean')#参数y_hat,y
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)#优化器
#torch.optim.SGD是一个类

epoch_list=[]
loss_list=[]

for epoch in range(1100):
    y_pred = model(x_data)
    #通过模型器算y_hat
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())#loss标量，自动调用__str__

    optimizer.zero_grad()#梯度归零
    loss.backward()
    optimizer.step()#更新
    epoch_list.append(epoch)
    loss_list.append(loss.item())

#输出权重和偏置
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

#测试
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred=',y_test.data)

#画图
plt.plot(epoch_list,loss_list)
plt.show()