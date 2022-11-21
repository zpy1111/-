import torch

input_size = 4
hidden_size = 4
batch_size = 1
#准备数据
idx2char = ['e','h','l','o']#字典
x_data = [1,0,2,2,3]
y_data = [3,1,2,3,2]

#独热向量的查询
one_hot_lookup = [[1,0,0,0],
                  [0,1,0,0],
                  [0,0,1,0],
                  [0,0,0,1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]
inputs = torch.Tensor(x_one_hot). view(-1,batch_size,input_size)
labels = torch.LongTensor(y_data).view(-1,1)

class Model(torch.nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell = torch.nn.RNNCell(input_size=self.input_size,hidden_size=self.hidden_size)

    def forward(self,input,hidden):
        hidden = self.rnncell(input,hidden)
        return hidden
    #初始隐层
    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)

net = Model(input_size,hidden_size,batch_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.1)

for epoch in range(20):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()
    print('Predicted string:', end='')
    for input, label in zip(inputs, labels):  # inputs:seq_len*batch_size*input_size-->input:batch_size*input_size; labels:seq_len*1-->label:1
        hidden = net(input, hidden)  # 输出和隐层命明必须一致
        loss += criterion(hidden, label)  # hidden：batch_size*hidden_size,即hidden_size;label:1   构建计算图，loss.item()-->loss
        _, idx = hidden.max(dim=1)
        print(idx2char[idx.item()], end='')
    loss.backward()  # loss.backward()要放for input，……层外面-->否则报错：计算图已被释放。因为loss+=，需要将五个序列的loss相加才是整个的loss，之后再进行后向传播backward
    optimizer.step()
    print(',Epoch [%d/15]  loss =%.4f' % (epoch, loss.item()))
