import torch

input_size = 4
hidden_size = 4
batch_size = 1
num_layers =1
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
labels = torch.LongTensor(y_data).squeeze(-1)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.rnn = torch.nn.RNN(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers)

    def forward(self,input):
        hidden = torch.zeros(num_layers,batch_size,hidden_size)
        out, _ = self.rnn(input,hidden)# 输出为两个，out:seq_len*batch_size*input_size; hidden:num_layers*batch_size*hidden_size
        return out.view(-1,hidden_size)

net = Model()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.1)
for epoch in range(20):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs,labels)
    loss.backward()
    optimizer.step()

    _,idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted:',''.join([idx2char[x] for x in idx]),end='')
    print(',Epoch [%d/15]  loss =%.4f' % (epoch+1, loss.item()))
