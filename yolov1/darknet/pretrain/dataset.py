# -*- coding: utf-8 -*-
# Author: Changxing DENG
# @Time: 2021/10/29 2:03

import random
import numpy as np
import cv2 as cv
from torch.utils.data import Dataset

import time

class PreTrainDataset(Dataset):
    def __init__(self, jpg_files, labels, transform=None, train=True):
        self.jpg_files = jpg_files
        self.labels = labels
        self.transform = transform
        self.train = train

    def __getitem__(self, index):
        file_path = self.jpg_files[index]
        img = cv.imread(file_path)
        if self.train:
            hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            hsv_img = self.hue(hsv_img)
            hsv_img = self.saturation(hsv_img)
            hsv_img = self.value(hsv_img)
            img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
            img = self.flip(img)
            img = self.blur(img)
            img = self.scale(img)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (224, 224))
        if self.transform:
            img = self.transform(img)
        label = self.labels[index] - 1
        return img, label

    def __len__(self):
        return len(self.jpg_files)

    def hue(self, hsv_img):
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            enhanced_hue = hsv_img[:, :, 0] * factor
            enhanced_hue = np.clip(enhanced_hue, 0, 180).astype(hsv_img.dtype)
            hsv_img[:, :, 0] = enhanced_hue
        return hsv_img

    def saturation(self, hsv_img):
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            enhanced_saturation = hsv_img[:, :, 1] * factor
            enhanced_saturation = np.clip(enhanced_saturation, 0, 255).astype(hsv_img.dtype)
            hsv_img[:, :, 1] = enhanced_saturation
        return hsv_img

    def value(self, hsv_img):
        if random.random() > 0.5:
            factor = random.uniform(0.8, 1.2)
            enhanced_value = hsv_img[:, :, 2] * factor
            enhanced_value = np.clip(enhanced_value, 0, 255).astype(hsv_img.dtype)
            hsv_img[:, :, 2] = enhanced_value
        return hsv_img

    def flip(self, img):
        if random.random() > 0.5:
            flip_code = random.choice([-1, 0, 1])
            img = cv.flip(img, flip_code)
        return img

    def blur(self, img):
        if random.random() > 0.5:
            img = cv.blur(img, (3, 3))
        return img

    def scale(self, img):
        prob = random.random()
        if prob > 0.7:
            factor = random.uniform(0.8, 1.2)
            h, w, _ = img.shape
            h = int(h * factor)
            img = cv.resize(img, (w, h))
        elif prob < 0.3:
            factor = random.uniform(0.8, 1.2)
            h, w, _ = img.shape
            w = int(w * factor)
            img = cv.resize(img, (w, h))
        return img
