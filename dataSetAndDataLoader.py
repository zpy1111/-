import numpy as np
import torch
from torch.utils.data import Dataset
#DataSet是一个抽象类，只能被继承
from torch.utils.data import DataLoader
#帮助加载数据，可实例化
import matplotlib.pyplot as plt

#定义一个类继承自Dateset
class DiabetesDataset(Dataset):
    def __init__(self,filepath):
        xy=np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        #第一个“：”是指读取所有行，第二个‘：’是指从第一列开始，最后一个列不要
        self.y_data = torch.from_numpy(xy[:, [-1]])
        #[-1]  最后拿出的数据是数组
    #实例化后能够支持下标操作
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
        #(x,y)
    #数据集的条数进行返回
    def __len__(self):
        return self.len

dataset =DiabetesDataset('diabetes.csv.gz')
train_loader =DataLoader(dataset=dataset,batch_size=80,shuffle=True,num_workers=0)
#num_worker并行数量

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
criterion = torch.nn.BCELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=0.0001)

loss_list=[]
epoch_list=[]
if __name__=='__main__':
    for epoch in range(100):
        for i,data in enumerate(train_loader,0):
            #1.准备数据
            inputs,labels = data
            #2.forward
            y_pred = model(inputs)
            loss = criterion(y_pred,labels)
            print(epoch,i,loss.item())
            #3.backward
            optimizer.zero_grad()
            loss.backward()
            #4.update
            optimizer.step()
            epoch_list.append(epoch)
            loss_list.append(loss.item())

plt.plot(epoch_list,loss_list)
plt.show()