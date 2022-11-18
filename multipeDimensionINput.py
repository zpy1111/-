import torch
import numpy as np


#数据准备
xy = np.loadtxt('diabetes.csv.gz',delimiter=',',dtype=np.float32)
#delimiter相应的分割符   dtype指定数据类型
x_data = torch.from_numpy(xy[:,:-1])
#第一个“：”是指读取所有行，第二个‘：’是指从第一列开始，最后一个列不要
y_data = torch.from_numpy(xy[:, [-1]])
#[-1]  最后拿出的数据是数组

#构建模型
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8,6)#输入数据为8维
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()#继承module

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x
model = Model()
#构造损失和优化器
#损失
criterion = torch.nn.BCELoss(reduction='mean')#BCEloss只用于二分类问题

#优化器
#optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
#用SGD损失函数稳定在0.6
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)
#循环训练
for epoch in range(10000):
    #forward
    y_pred = model(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())
    #backward
    optimizer.zero_grad()
    loss.backward()
    #update
    optimizer.step()