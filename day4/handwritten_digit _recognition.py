import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import sys
import torchvision.transforms as T
import math

USE_CUDA = False


class HDR_M(nn.Module):
    def __init__(self):
        super(HDR_M, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=0)
        # out_channels = 6意味着有6个过滤器
        # in: 32 * 32 * 3 out: 28 *28 *6
        # (32 - 5 + 2 * 0) / s + 1 = 28

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
        # x = Variable(torch.unsqueeze(x, dim=0).float(), requires_grad=False)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 5 * 5 * 16)  # 表示将x进行reshape，为后面做为全连接层的输入
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        predictions = self.softmax6(x)
        return predictions


class HandwrittenDigitDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.imgs = list(sorted(os.listdir(os.path.join(root, "JPGImages"))))
        self.ids = []
        with open(os.path.join(root, "ImageNum", "ImageNum.txt")) as f:
            for line in f.readlines():
                self.ids.append(int(line))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "JPGImages", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        img = img.resize((32, 32))

        obj_id = self.ids[idx]

        if self.transforms is not None:
            img = self.transforms(img)
        # target： 第几类（数字几）
        target = torch.as_tensor(obj_id)

        return img, target

    def __len__(self):
        return len(self.imgs)



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    # 此语句只针对特定的模型有用
    # 我们可以定义不同模式下模型的不同行为
    # 在这个模型中，此语句没有实际作用
    model.train()
    # 交叉熵损失函数
    loss_fnc = nn.CrossEntropyLoss()
    for images, targets in data_loader:
        predictions = model(images)
        losses = loss_fnc(predictions, targets)
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print('epoch{} | losses:{}'.format(epoch, losses))
    if epoch % 100 == 0:
        torch.save({
            'model_state': model.state_dict(),
            'loss': losses
        }, os.path.join(os.getcwd(),'HDR_M.pth'))

def main():
    model = HDR_M()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=1. / 1000)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    dataset = HandwrittenDigitDataset(os.getcwd(), get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=5, shuffle=True, num_workers=4
        # ,
        # collate_fn=collate_fn
    )

    device = torch.device('cuda') if USE_CUDA else torch.device("cpu")
    model = model.to(device)

    num_epochs = 1000000

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()


if __name__ == '__main__':
    # dataset = HandwrittenDigitDataset(os.getcwd(), get_transform(train=True))
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=5, shuffle=True, num_workers=4,
    #     collate_fn=collate_fn)
    # for i in range(1):
    #     for images, targets in data_loader:
    #         for k in range(len(images)):
    #             print(images[k])
    #             print(targets[k])

    main()

    # loss_fnc = nn.CrossEntropyLoss()
    # pre = torch.tensor([[0.3, 0.4, 0.5, 0.5],[0.3, 0.4, 0.5, 0.5]], dtype=float)
    # target = torch.tensor([2,2])
    # print(loss_fnc(pre, target))

    # lay1 = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=2, padding=0, stride=1)
    # data = torch.tensor([[
    #     [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
    #     , [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]
    #     , [[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]]
    # ]], dtype=float)
    # data = torch.randn(20,3,32,32)
    # out = lay1(data)
    # print(out[0].shape)
