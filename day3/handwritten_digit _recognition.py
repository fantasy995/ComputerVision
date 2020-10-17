import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = False


class HDR_M(nn.Module):
    def __init__(self):
        super(HDR_M, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        # out_channels = 5意味着有6个过滤器
        # in: 32 * 32 * 3 out: 28 *25 *6
        # (32 - 5 + 2 * 0) / s + 1 = 8

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # pooling后 高度、宽带减小一半
        # out: 14 * 14 * 6

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        # out: 10 * 10 * 16

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # out: 5 * 5 * 16

        # 全连接层 线性连接
        self.fc3 = nn.Linear(5 * 5 * 16, 120)
        self.fc4 = nn.Linear(120, 84)
        self.fc5 = nn.Linear(84, 10)
        self.softmax6 = nn.Softmax(1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 5 * 5 * 16)  # 表示将x进行reshape，为后面做为全连接层的输入
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.softmax6(x)
        return x


def train():
    model = HDR_M()
    params = [p for p in model.parameters() if p.requires_grad]
    print(params)
    optimizer = torch.optim.Adam(params, lr=0.005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)



if __name__ == '__main__':
    train()

