from day1.tv_training_data import *

'''
10.16日
导入预训练模型
并微调模型
进行模型训练
'''
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from references.detection.engine import train_one_epoch, evaluate
from references.detection import utils



def get_model_instance_segmentation(num_classes):
    # 加载预训练的模型
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def main():
    # gpu内存不够
    device = torch.device('cpu')
    # 已经把masks更改为0 1
    # 0-背景 1-目标
    num_classes = 2

    # 数据集
    dataset = PennFudanDataset('..\\data\\PennFudanPed', get_transform(train=True))

    # img, t = dataset.__getitem__(0)
    # print(img)
    # exit(50)

    dataset_test = PennFudanDataset('..\\data\\PennFudanPed', get_transform(train=False))

    # 划分训练数据和测试数据
    indices = torch.randperm(len(dataset)).tolist()
    # 由于内存不够，选取少量数据进行训练和测试
    dataset = torch.utils.data.Subset(dataset, indices[:10])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[10:20])

    # 定义data_loader
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # 加载预训练的模型
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # 定义优化函数
    # 添加模型内需要更新梯度的参数
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # 只训练一次 一次 10张图片
    num_epochs = 1

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()

        # 保存模型
        utils.save_on_master({
            'model': model.state_dict()
        },
            '..\\models\\model_{}.pth'.format(epoch))

        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")


if __name__ == "__main__":
    main()





