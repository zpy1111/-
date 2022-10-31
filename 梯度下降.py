import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

#初始权重
w=1.0

#前馈计算，  y^（y_hat)
def forward(x):
    return x*w

#损失
def cost(xs,ys):
    cost = 0
    for x,y in zip(xs,ys):
        y_pred =forward(x)      #求y_hat
        cost +=(y_pred-y)**2   #（y^-y)**2
        return cost/len(xs)

#梯度函数
def gradient(xs,ys):
    grad =0
    for x,y in zip(xs,ys):
        grad +=2 * x * (x * w -y)
        return grad/len(xs)
epoch_list=[]
cost_list=[]
#训练函数   权重-学习率*梯度
print('Predict (before training)',4,forward(4))
for epoch in range (500):#轮数
    cost_val =cost(x_data,y_data)
    grad_val = gradient(x_data,y_data)#梯度
    w-= 0.01*grad_val#更新
    print("Epoch:",epoch,'w=',w,'loss=',cost_val)
    epoch_list.append(epoch)
    cost_list.append(cost_val)
print('Predict(after training)',4,forward(4))


#画图

plt.plot(epoch_list,cost_list)
plt.ylabel('cost')
plt.xlabel('epoch')
plt.show()
