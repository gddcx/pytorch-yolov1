# -*- coding: utf-8 -*-
# Author: Changxing DENG
# @Time: 2021/10/10 0:18

import torch.nn as nn
import torchvision

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu3 = nn.ReLU()
        self.downsample = nn.Sequential()
        if in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, data):
        out = self.relu1(self.bn1(self.conv1(data)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # if self.downsample: # 这样写只有在需要下采样的时候才有跳层连接
        #     out += self.downsample(data)
        out += self.downsample(data)
        out = self.relu3(out)
        return out

class ResNet50Det(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=pretrain)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        # self.detect_head = nn.Sequential(
        #     nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=256, out_channels=30, kernel_size=3, stride=2, padding=1, bias=False),
        #     nn.BatchNorm2d(30),
        #     nn.Sigmoid()
        # )

        # self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU()
        )
        self.detect_block1 = BasicBlock(2048, 256)
        self.detect_block2 = BasicBlock(256, 256)
        self.detect_block3 = BasicBlock(256, 30)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=30, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(30),
            nn.Sigmoid()
        )
        # BUG
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         m.weight.data.normal_(mean=0, std=0.02)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def forward(self, data):
        x = self.backbone(data)
        # x = self.detect_head(x)
        # x = self.avgpool(x)
        x = self.conv(x)  #296, 0.51
        x = self.detect_block1(x)
        x = self.detect_block2(x)
        x = self.detect_block3(x)
        x = self.downsample(x)
        return x


