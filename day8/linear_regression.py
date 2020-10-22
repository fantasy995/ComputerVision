import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

x = np.linspace(0, 5, 256)
# 在0和5间产生256个线性向量，每个向量之间的距离是相同的
# print(x.shape,x[0],x[1],x[2]) # (256,) 0.0 0.0196078431372549 0.0392156862745098

noise = np.random.randn(256) * 2
# 生成256个符合正态分布的随机数
# print(noise.shape, noise[0], noise[1], noise[2]) # (256,) -2.6409764724624236 2.030337787313112 1.7641477196152713

y = x * 5 + 7 + noise
# 要拟合的函数
# print(y.shape) # (256,)

df = pd.DataFrame()
df['x'] = x
df['y'] = y
sns.lmplot(x='x', y='y', data=df, height=4)
plt.show()

train_x = x.reshape(-1, 1).astype('float32')
train_y = y.reshape(-1, 1).astype('float32')
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)

model = nn.Linear(1, 1)

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 3000

for i in range(1, epochs+1):
    optimizer.zero_grad()
    out = model(train_x)
    loss = loss_fn(out, train_y)
    loss.backward()
    optimizer.step()
    if(i % 300 == 0):
        print('epoch {}  loss {:.4f}'.format(i, loss.item()))

w, b = model.parameters()  # parameters()返回的是一个迭代器指向的对象
print(w.item(), b.item())

#model返回的是总tensor，包含grad_fn，用data提取出的tensor是纯tensor
pred = model.forward(train_x).data.numpy().squeeze()
plt.plot(x, y, 'go', label='Truth', alpha=0.3)
plt.plot(x, pred, label='Predicted')
plt.legend()
plt.show()
