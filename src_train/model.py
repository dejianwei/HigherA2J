import torch.nn as nn
from torch.nn import init
import resnet
import math


class ResNetBackBone(nn.Module):
    def __init__(self):
        super(ResNetBackBone, self).__init__()
        self.model = resnet.resnet50(pretrained=True)

    def forward(self, x):
        n, c, h, w = x.size()  # x: [B, 1, H ,W]

        x = x[:,0:1,:,:]  # depth
        x = x.expand(n,3,h,w)

        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        return x

class A2J_model(nn.Module):
    def __init__(self, num_classes):
        super(A2J_model, self).__init__()
        self.Backbone = ResNetBackBone() # 1 channel depth only, resnet50
        self.offset = nn.Conv2d(2048, num_classes*3, kernel_size=3, padding=1)
        self.weight = nn.Conv2d(2048, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.Backbone(x)
        regression = self.offset(x)
        classification = self.weight(x)
        return (classification, regression)
