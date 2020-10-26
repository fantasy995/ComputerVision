import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


def fun(x: torch.Tensor):
    return 1.3 * x + 4


# 只有叶子节点有梯度
def test0():
    a = torch.tensor(1, requires_grad=True, dtype=float)
    b = fun(a)
    # 不是叶子节点的向量要使用retain_grad()设置属性
    # b.retain_grad()
    y = fun(b)
    y.backward()
    print(y)
    print(a.grad)
    print(b.grad)


# 测试神经网络中的情况
class ModelTest(nn.Module):
    def __init__(self):
        super(ModelTest, self).__init__()

        # 假设输入 5 * 5 输出 3 * 3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
        # 输入3*3 输出 2*2
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1, padding=0)

        self.fc1 = nn.Linear(4, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


def testInModel():
    my_model = ModelTest()
    optimizer = optim.SGD(my_model.parameters(), lr=0.01, momentum=0.5)
    loss_fun = F.mse_loss

    train_data = torch.rand((100, 1, 5, 5))
    target = torch.as_tensor([torch.sum(train_data[i]) for i in range(100)]).reshape(100, 1)

    pre = my_model(train_data)

    loss = loss_fun(pre, target)

    loss.backward()
    for parms in my_model.parameters():
        print('-->grad_requirs:', parms.requires_grad, \
              ' -->grad_value:', parms.grad)

if __name__ == '__main__':
    test0()
    # testInModel()

