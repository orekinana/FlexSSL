import torch
import torch.nn
import torch.nn.functional as F
import torchvision
import torchvision.models
from torchvision.models.resnet import resnet18

from models.cnn13 import cnn13


class Discriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # self.net_x = resnet18(num_classes=1)
        self.net_x = cnn13(num_classes=1)

        # self.net_x = torch.nn.Sequential()
        # # 64 X 1 X 28 X 28
        # self.net_x.add_module('conv1', torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False))
        # self.net_x.add_module('relu1', torch.nn.ReLU(inplace=True))
        # # 64 X 32 X 28 X 28
        # self.net_x.add_module('conv2', torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False))
        # self.net_x.add_module('relu2', torch.nn.ReLU(inplace=True))

        # # 64 X 32 X 28 X 28
        # self.net_x.add_module('gap', torch.nn.AdaptiveAvgPool2d(1))

        # # 64 X 32 X 1 X 1
        # self.net_x.add_module('flat', torch.nn.Flatten(1))
        # # 64 X 32
        # self.net_x.add_module('fc1', torch.nn.Linear(32, 8))
        # self.net_x.add_module('relu3', torch.nn.ReLU(inplace=True))
        # # 64 X 8
        # self.net_x.add_module('fc2', torch.nn.Linear(8, 8))
        # self.net_x.add_module('relu4', torch.nn.ReLU(inplace=True))
        # # 64 X 8
        # self.net_x.add_module('fc3', torch.nn.Linear(8, 1))

        self.net_y = torch.nn.Sequential()
        self.net_y.add_module('fc4', torch.nn.Linear(10, 5))
        self.net_y.add_module('relu5', torch.nn.ReLU(inplace=True))
        self.net_y.add_module('fc5', torch.nn.Linear(5, 5))
        self.net_y.add_module('relu6', torch.nn.ReLU(inplace=True))
        self.net_y.add_module('fc6', torch.nn.Linear(5, 1))

        self.fusion = torch.nn.Linear(2, 1)


        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def forward(self, x: torch.Tensor, y_hat: torch.Tensor, loss: torch.Tensor):
        # x.size():     b X 1 X 28 X 28
        # y_hat.size(): b X 10
        # loss.size():  b X 1

        y_hat = F.one_hot(y_hat.argmax(dim=1), num_classes=10)
        x = self.net_x(x)
        xy = self.net_y(x * y_hat)

        loss = loss.unsqueeze(dim=-1)
        output = torch.cat([xy, loss], dim=1)
        output = self.fusion(output)

        return F.sigmoid(output)