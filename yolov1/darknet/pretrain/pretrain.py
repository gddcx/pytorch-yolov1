# -*- coding: utf-8 -*-
# Author: Changxing DENG
# @Time: 2021/10/28 22:03

import os
import random
import numpy as np
import glob
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
# from sacred import Experiment
# from sacred.observers import MongoObserver

from network import YOLOBackbone
from dataset import PreTrainDataset

EXPERIMENT_NAME = "YOLO_PretrainDarkNet"
# ex = Experiment(EXPERIMENT_NAME)
# ex.observers.append(MongoObserver(url="11.11.11.100:11220", db_name="sacred"))

def set_random_seeds(random_seed=0):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def load_data_train(DATA_PATH):
    mapping_dict = {}
    with open(os.path.join(DATA_PATH, "ILSVRC2012", "ILSVRC2012_category_mapping.txt"), 'r') as f:
        lines = f.readlines()
    for line in lines:
        split_line = line.split(" ")  # ["n02119789", "1", "kit_fox"]
        str_id = split_line[0].strip()
        number_id = split_line[1].strip()
        mapping_dict[str_id] = int(number_id)
    data_path = os.path.join(DATA_PATH, "ILSVRC2012", "train", "*", "*.JPEG")
    jpg_files = glob.glob(data_path)
    labels = []
    for file in jpg_files:
        str_id = os.path.dirname(file).split("/")[-1]
        labels.append(mapping_dict[str_id])
    return jpg_files, labels

def load_data_val(DATA_PATH):
    data_path = os.path.join(DATA_PATH, "ILSVRC2012", "val", "*.JPEG")
    jpg_files = glob.glob(data_path)
    jpg_files = sorted(jpg_files)  # 因为ground truth文件里每一行对应一个文件label
    with open(os.path.join(DATA_PATH, "ILSVRC2012", "ILSVRC2012_validation_ground_truth.txt"), 'r') as f:
        lines = f.readlines()
    labels = [int(line.strip("\n")) for line in lines]
    return jpg_files, labels

# @ex.automain
def main(log_name):
    print(torch.cuda.get_device_name(0), ":", torch.cuda.device_count())
    set_random_seeds(42)
    DATA_PATH = "../../../../dataset/"
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCH = 1000
    PRINT_INTERVAL = 2000
    SAVE_ROOT = os.path.join("models", EXPERIMENT_NAME, log_name)
    os.makedirs(SAVE_ROOT, exist_ok=True)
    #-----数据加载-----#
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_files, train_labels = load_data_train(DATA_PATH)
    eval_files, eval_labels = load_data_val(DATA_PATH)
    train_dataset = PreTrainDataset(train_files, train_labels, transform, train=True)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=True, drop_last=True)
    eval_dataset = PreTrainDataset(eval_files, eval_labels, transform, train=False)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, num_workers=8, shuffle=False, drop_last=False)
    #-----模型加载-----#
    net = YOLOBackbone()
    net = nn.DataParallel(net)
    net = net.cuda()
    #-----优化-----#
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    #-----迭代训练/验证-----#
    step = 0
    best_loss = 1e6
    for epoch in range(EPOCH):
        net.train()
        if epoch == 1:
            optimizer.param_groups[0]['lr'] = 1e-2
        elif epoch == 20:
            optimizer.param_groups[0]['lr'] = 1e-3
        elif epoch == 25:
            optimizer.param_groups[0]['lr'] = 1e-4
        for iter, (img, label) in enumerate(train_loader):
            img = img.cuda()
            label = label.cuda()
            out = net(img)
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iter % PRINT_INTERVAL == 0:
                print("Time: {} Epoch:[{}/{}] Iter:[{}/{}] Loss: {}"
                      .format(time.strftime("%m-%d %H:%M:%S", time.localtime()),epoch, EPOCH, iter, len(train_loader), loss.item()), flush=True)
            step += 1
        net.eval()
        total_loss = 0
        total_correct = 0
        with torch.no_grad():
            for _, (img, label) in enumerate(eval_loader):
                img = img.cuda()
                label = label.cuda()
                out = net(img)
                loss = criterion(out, label)
                total_loss += loss.item() * img.shape[0]
                _, idx = torch.topk(out, dim=-1, k=5)
                label = label.unsqueeze(-1)
                total_correct += torch.eq(idx, label).sum().float().item()
            avg_loss = total_loss / len(eval_dataset)
            accuracy = total_correct / len(eval_dataset)
            print("Time: {} Epoch:[{}/{}] Loss: {}, Acc: {}"
                  .format(time.strftime("%m-%d %H:%M:%S", time.localtime()), epoch, EPOCH, avg_loss, accuracy))
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(net.state_dict(), f"{SAVE_ROOT}/{epoch}_{step}.pth")

if __name__ == "__main__":
    log_name = sys.argv[1]
    main(log_name)
