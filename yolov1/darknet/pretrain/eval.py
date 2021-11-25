# -*- coding: utf-8 -*-
# Author: Changxing DENG
# @Time: 2021/10/28 22:03

import os
import random
import numpy as np
import glob
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2 as cv
from network import YOLOBackbone

class PreTrainDataset(Dataset):
    def __init__(self, jpg_files, labels, transform=None):
        self.jpg_files = jpg_files
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.jpg_files[index]
        img = cv.imread(file_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # img = self.resize(img) # single-crop
        img = cv.resize(img, (224, 224))
        if self.transform:
            img = self.transform(img)
        label = self.labels[index] - 1
        return img, label

    def __len__(self):
        return len(self.jpg_files)

    def resize(self, img):
        h, w, _ = img.shape
        if h < w:
            factor = 224 / h
            w = int(factor * w) + 1 # 如果不+1， 本身就是正方形的图片，其中一边resize到224后，另外一边有可能会223
            img = cv.resize(img, (w, 224))
        else:
            factor = 224 / w
            h = int(factor * h) + 1
            img = cv.resize(img, (224, h))
        h, w, _ = img.shape
        top = int((h - 224) / 2)
        left = int((w - 224) / 2)
        if w - 2 * left > 224:
            w -= 1
        if h - 2 * top > 224:
            h -= 1
        img = img[top: h-top, left:w-left, :]
        return img



def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def load_data_val(DATA_PATH):
    data_path = os.path.join(DATA_PATH, "ILSVRC2012", "val", "*.JPEG")
    jpg_files = glob.glob(data_path)
    jpg_files = sorted(jpg_files)  # 因为ground truth文件里每一行对应一个文件label
    with open(os.path.join(DATA_PATH, "ILSVRC2012", "ILSVRC2012_validation_ground_truth.txt"), 'r') as f:
        lines = f.readlines()
    labels = [int(line.strip("\n")) for line in lines]
    return jpg_files, labels

def main():
    DATA_PATH = "D:\\dataset\\"
    BATCH_SIZE = 256
    #-----数据加载-----#
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    eval_files, eval_labels = load_data_val(DATA_PATH)
    eval_dataset = PreTrainDataset(eval_files, eval_labels, transform)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, num_workers=1, shuffle=False, drop_last=False)
    #-----模型加载-----#
    net = YOLOBackbone()
    # temp_state_dict = torch.load("45_920828.pth", map_location="cpu")
    temp_state_dict = torch.load("50_1020918.pth", map_location="cpu")
    state_dict = {k.replace("module.", ""): v for k, v in temp_state_dict.items()}
    net.load_state_dict(state_dict)
    net = net.cuda()
    #-----迭代训练/验证-----#
    net.eval()
    total_correct = 0
    with torch.no_grad():
        for iter, (img, label) in enumerate(eval_loader):
            print(f"{iter}/{len(eval_loader)}")
            img = img.cuda()
            label = label.cuda()
            out = net(img)
            _, idx = torch.topk(out, dim=-1, k=5)
            label = label.unsqueeze(-1)
            total_correct += torch.eq(idx, label).sum().float().item()
        accuracy = total_correct / len(eval_dataset)
    print("single crop top-5 accuracy: {}".format(accuracy))

if __name__ == "__main__":
    main()
