import torch.nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
#准备数据集
x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])

#构造模型

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred=torch.sigmoid(self.linear(x))
        return y_pred
#实例化
model = LogisticRegressionModel()
#损失
criterion = torch.nn.BCELoss(reduction='mean')#BCEloss只用于二分类问题
#优化器
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
#循环训练
for epoch in range(2000):
    y_pred = model(x_data)
    loss =criterion(y_pred,y_data)
    print(epoch,loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#输出权重和偏置
print('w=',model.linear.weight.item())
print('b=',model.linear.bias.item())

#测试
x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred=',y_test.data)

x =np.linspace(0,10,200)#采点
x_t =torch.Tensor(x).view(200,1)#组成数组
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x,y)
plt.plot([0,10],[0.5,0.5],c='r')
plt.xlabel('Hours')
plt.ylabel('Probability of class')
plt.grid()
plt.show()
