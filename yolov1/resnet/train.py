# -*- coding: utf-8 -*-
# Author: Changxing DENG
# @Time: 2021/10/8 11:27

import glob
import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import xml.dom.minidom as xdm

from .dataset import VOCDataset
from .network import ResNet50Det

def load_data(data_path):
    voc2007_trainval_annotations = os.path.join(data_path, "VOC2007", "trainval", "Annotations", "*xml")
    voc2007_test_annotations = os.path.join(data_path, "VOC2007", "test", "Annotations", "*xml")
    voc2012_trainval_annotations = os.path.join(data_path, "VOC2012", "trainval", "Annotations", "*xml")
    annotation_path = glob.glob(voc2007_trainval_annotations) + glob.glob(voc2007_test_annotations) + glob.glob(voc2012_trainval_annotations)

    all_category = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                     "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                     "train", "tvmonitor"]

    random.shuffle(annotation_path)
    res = []
    for i, path in enumerate(annotation_path):
        path_split = os.path.dirname(path).split('/')
        folder = [path_split[3], path_split[4]] # [VOC2007/VOC2012, trainval/test]
        DOMTree = xdm.parse(path)
        collection = DOMTree.documentElement
        filename = collection.getElementsByTagName("filename")[0].childNodes[0].data
        category_list = []
        bndbox_list = []
        object_ = collection.getElementsByTagName("object")
        for obj in object_:
            category = obj.getElementsByTagName("name")[0].childNodes[0].data
            bndbox = obj.getElementsByTagName("bndbox")[0]
            xmin = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
            ymin = int(bndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
            xmax = int(bndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
            ymax = int(bndbox.getElementsByTagName("ymax")[0].childNodes[0].data)
            category_list.append(all_category.index(category))
            bndbox_list.append([xmin, ymin, xmax, ymax])
        res.append({"folder": folder, "filename": filename, "category": np.array(category_list), "bndbox": np.array(bndbox_list)})
    return res

def main():
    DATA_PATH = "../../dataset"
    BATCH_SIZE = 64
    LEARNING_RATE=1e-3
    EPOCH=135
    #-----数据加载-----#
    res = load_data(DATA_PATH)
    split = int(len(res)*0.8)
    train_res = res[:split]
    eval_res = res[split:]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_set = VOCDataset(data=train_res, image_root=DATA_PATH, transform=transform, train=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=8, drop_last=True, shuffle=True)
    eval_set = VOCDataset(data=eval_res, image_root=DATA_PATH, transform=transform, train=False)
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, num_workers=8, drop_last=False, shuffle=False)
    #-----模型加载-----#
    net = ResNet50Det(pretrain=True)
    net = nn.DataParallel(net)
    net = net.cuda()
    #-----优化器-----#
    Adam = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=0.0005)
    #-----迭代训练/验证-----#
    for epoch in range(EPOCH):
        for iter, (img, target) in enumerate(train_loader):
            img = img.cuda()
            target = target.cuda()
            net(img)


# if __name__ == '__main__':
#     main()
