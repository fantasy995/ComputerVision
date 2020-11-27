from torch import optim
from torchvision import models
import torch.nn as nn
from tools import device, train_model, visualize_model
import torch
import os

if __name__ == '__main__':
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    if os.path.exists(os.path.join(os.getcwd(),'model_ft.pth')):
        # model_ft = model_ft.to(torch.device('cpu'))
        model_ft.load_state_dict(torch.load(os.path.join(os.getcwd(),'model_ft.pth'))['mode_ft_state'])
        visualize_model(model_ft)
    else:
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=25)
        torch.save({
            'mode_ft_state': model_ft.state_dict(),
        }, os.path.join(os.getcwd(), 'model_ft.pth'))
        visualize_model(model_ft)