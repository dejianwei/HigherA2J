import torch.nn as nn
from resnet import resnet18
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models import resnet

class ResNetBackBone(nn.Module):
    def __init__(self, layer='50'):
        super(ResNetBackBone, self).__init__()
        if layer == '18':
            self.model = resnet18(pretrained=True)
        elif layer == '50':
            self.model = resnet.resnet50(pretrained=True, replace_stride_with_dilation=[False, False, True])
        elif layer == '101':
            self.model = resnet.resnet101(pretrained=True, replace_stride_with_dilation=[False, False, True])

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


class MobileNetV2BackBone(nn.Module):
    def __init__(self):
        super(MobileNetV2BackBone, self).__init__()
        setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ]
        self.model = mobilenet_v2(pretrained=True, inverted_residual_setting=setting)
        self.model.features = self.model.features[:-1]
    
    def forward(self, x):
        n, c, h, w = x.size()
        x = x[:,0:1,:,:]
        x = x.expand(n,3,h,w)

        x = self.model.features(x)
        return x


class A2J_model(nn.Module):
    def __init__(self, num_classes, backbone='resnet50'):
        super(A2J_model, self).__init__()
        if backbone == 'resnet18':
            self.Backbone = ResNetBackBone(layer='18')
            input_channel = 512
        elif backbone == 'resnet50':
            self.Backbone = ResNetBackBone(layer='50')
            input_channel = 2048
        elif backbone == 'mobilenet_v2':
            self.Backbone = MobileNetV2BackBone()
            input_channel = 320

        self.offset = nn.Conv2d(input_channel, num_classes*3, kernel_size=3, padding=1)
        self.weight = nn.Conv2d(input_channel, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.Backbone(x)
        regression = self.offset(x)
        classification = self.weight(x)
        return (classification, regression)
