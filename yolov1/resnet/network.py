# -*- coding: utf-8 -*-
# Author: Changxing DENG
# @Time: 2021/10/10 0:18

import torch.nn as nn
import torchvision
from collections import OrderedDict

class ResNet50Det(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=pretrain)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.detect_head = nn.Sequential(
            OrderedDict([
                ("conv1", nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)),
                ("bn1", nn.BatchNorm2d(256)),
                ("relu1", nn.ReLU()),
                ("conv2", nn.Conv2d(in_channels=256, out_channels=30, kernel_size=3, stride=2, padding=1, bias=False)),
                ("bn2", nn.BatchNorm2d(30)),
                ("sigmoid", nn.Sigmoid())
            ])
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean=0, std=0.02)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, data):
        x = self.backbone(data)
        x = self.detect_head(x)
        return x
