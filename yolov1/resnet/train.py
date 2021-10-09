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

import xml.dom.minidom as xdm

from .dataset import VOCDataset


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
    #-----数据加载-----#
    res = load_data(DATA_PATH)
    split = int(len(res)*0.8)
    train_res = res[:split]
    eval_res = res[split:]
    train_set = VOCDataset(data=train_res, image_root=DATA_PATH, transform=None, train=True)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=8, drop_last=True, shuffle=True)
    eval_set = VOCDataset(data=eval_res, image_root=DATA_PATH, transform=None, train=False)
    eval_loader = DataLoader(eval_set, batch_size=BATCH_SIZE, num_workers=8, drop_last=False, shuffle=False)
    #-----模型加载-----#
    #-----优化器-----#
    #-----迭代训练/验证-----#


# if __name__ == '__main__':
#     main()
