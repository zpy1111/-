import torch
from torchvision import transforms#对图像进行原始处理
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim


batch_size =64
transform =transforms.Compose([
    transforms.ToTensor(),#把图像进行转变成张量
    transforms.Normalize((0.1307,),(0.3081,))#标准化：均值  标准差
])
train_dataset = datasets.MNIST(root='../dataset/mnist/',train=True,download=True,transform=transform)
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
test_dataset = datasets.MNIST(root='../dataset/mnist/',train=False,download=True,transform=transform)
test_loader = DataLoader(test_dataset,shuffle=False,batch_size=batch_size)

#model
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)
    def forward(self,x):
        x = x.view(-1,784)#改变形状,-1自动获取mini—batch
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)#最后一层不激活

model = Net()
#softmax   log   one-hot
criterion = torch.nn.CrossEntropyLoss()
#带冲量的
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

#一轮循环封装成函数
def train(epoch):
    running_loss=0.0
    for batch_idx,data in enumerate(train_loader,0):
        #获取一个批次的数据和标签
        inputs,target = data
        optimizer.zero_grad()
        #forward+backward + update
        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()
        running_loss +=loss.item()
        #输出,每300次迭代输出一次
        if batch_idx %300 == 299:
            print('[%d,%5d] loss:%.3f' % (epoch + 1,batch_idx+1,running_loss/300))
            running_loss = 0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():#不会计算梯度
        for data in test_loader:
            images,labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim = 1)#沿着第一维度找最大值的下标（最大值，下标）
            total += labels.size(0)
            correct += (predicted == labels).sum().item() #真为1，假为0
        print('Accuracy on test set: %d %%' % (100 * correct/ total))



if __name__=='__main__':
        for epoch in range(10):
            train(epoch)
            test()
