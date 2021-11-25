# -*- coding: utf-8 -*-
# Author: Changxing DENG
# @Time: 2021/10/8 11:27

import os
import cv2 as cv
import random
import numpy as np
import torch
from torch.utils.data import Dataset

class VOCDataset(Dataset):
    def __init__(self, data, image_root="", transform=None, train=True):
        super().__init__()
        self.data = data
        self.transform = transform
        self.image_root = image_root
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_dict = self.data[index]
        folder = image_dict["folder"]
        name = image_dict["filename"]
        image_path = os.path.join(self.image_root, *folder, "JPEGImages", name)
        img = cv.imread(image_path)
        category = image_dict["category"]
        bbox = image_dict["bndbox"] # shape: nbox, 4
        if self.train: # 数据增强
            hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
            hsv_img = self.hue(hsv_img)
            hsv_img = self.saturation(hsv_img)
            hsv_img = self.value(hsv_img)
            img = cv.cvtColor(hsv_img, cv.COLOR_HSV2BGR)
            img = self.average_blur(img)
            img, bbox = self.horizontal_flip(img, bbox)
            img, bbox, category = self.crop(img, bbox, category)
            img, bbox = self.scale(img, bbox)
            img, bbox, category = self.translation(img, bbox, category)
        # Caused by data augmentation, such as crop, translation and so on.
        if len(bbox) == 0 or len(category) == 0:
            return [], []
        target = self.encoder(img, bbox, category)
        img = cv.resize(img, (448, 448))
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        target = torch.from_numpy(target)
        return img, target

    def hue(self, hsv_img):
        if random.random() > 0.5:
            factor = random.uniform(0.5, 1.5)
            enhanced_hue = hsv_img[:, :, 0] * factor
            enhanced_hue = np.clip(enhanced_hue, 0, 180).astype(hsv_img.dtype)  # H的范围是0-180(360/2)
            hsv_img[:, :, 0] = enhanced_hue
        return hsv_img

    def saturation(self, hsv_img):
        if random.random() > 0.5:
            factor = random.uniform(0.5, 1.5)
            enhanced_saturation = hsv_img[:, :, 1] * factor
            enhanced_saturation = np.clip(enhanced_saturation, 0, 255).astype(hsv_img.dtype)
            hsv_img[:, :, 1] = enhanced_saturation
        return hsv_img

    def value(self, hsv_img):
        if random.random() > 0.5:
            factor = random.uniform(0.5, 1.5)
            enhanced_value = hsv_img[:, :, 2] * factor
            enhanced_value = np.clip(enhanced_value, 0, 255).astype(hsv_img.dtype)
            hsv_img[:, :, 2] = enhanced_value
        return hsv_img

    def average_blur(self, img):
        if random.random()>0.5:
            img = cv.blur(img, (3, 3))
        return img

    def horizontal_flip(self, img, bbox):
        if random.random() > 0.5:
            img = cv.flip(img, 1)
            h, w, _ = img.shape
            temp = w - bbox[:, 0]
            bbox[:, 0] = w - bbox[:, 2]
            bbox[:, 2] = temp
        return img, bbox

    def crop(self, img, bbox, category):
        if random.random() > 0.5:
            factor_horizontal = random.uniform(0, 0.2)
            factor_vertical = random.uniform(0, 0.2)
            h, w, _ = img.shape
            start_horizontal = int(w * factor_horizontal)
            end_horizontal = start_horizontal + int(0.8 * w)
            start_vertical = int(h * factor_vertical)
            end_vertical = start_vertical + int(0.8 * h)
            img = img[start_vertical: end_vertical, start_horizontal:end_horizontal, :]
            center_x = (bbox[:, 0] + bbox[:, 2]) / 2
            center_y = (bbox[:, 1] + bbox[:, 3]) / 2
            inImage = (center_x > start_horizontal) & (center_x < end_horizontal) \
                      & (center_y > start_vertical) & (center_y < end_vertical)
            bbox = bbox[inImage, :]
            bbox[:, [0, 2]] = bbox[:, [0, 2]] - start_horizontal
            bbox[:, [1, 3]] = bbox[:, [1, 3]] - start_vertical
            bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], 0, int(0.8 * w))
            bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], 0, int(0.8 * h))
            category = category[inImage]
        return img, bbox, category

    def scale(self, img, bbox):
        probility = random.random()
        if probility > 0.7:
            factor = random.uniform(0.5, 1.5)
            h, w, _ = img.shape
            h = int(h * factor)
            img = cv.resize(img, (w, h))  # size的顺序是w,h
            bbox[:, [1, 3]] = bbox[:, [1, 3]] * factor
        elif probility < 0.3:
            factor = random.uniform(0.5, 1.5)
            h, w, _ = img.shape
            w = int(w * factor)
            img = cv.resize(img, (w, h))
            bbox[:, [0, 2]] = bbox[:, [0, 2]] * factor
        bbox = bbox.astype(np.int)
        return img, bbox

    def translation(self, img, bbox, category):
        if random.random() > 0.5:
            factor_horizontal = random.uniform(-0.2, 0.2)
            factor_vertical = random.uniform(-0.2, 0.2)
            h, w, _ = img.shape
            w_tran = int(w * factor_horizontal)
            h_tran = int(h * factor_vertical)
            canvas = np.zeros_like(img)
            if w_tran < 0 and h_tran < 0:  # 向右下移动
                canvas[-h_tran:, -w_tran:, :] = img[:h + h_tran, :w + w_tran, :]
            elif w_tran < 0 and h_tran >= 0:  # 向右上移动
                canvas[:h - h_tran, -w_tran:, :] = img[h_tran:, :w + w_tran, :]
            elif w_tran >= 0 and h_tran < 0:  # 向左下移动
                canvas[-h_tran:, :w - w_tran, :] = img[:h + h_tran, w_tran:, :]
            elif w_tran >= 0 and h_tran >= 0:  # 向左上移动
                canvas[:h - h_tran, :w - w_tran, :] = img[h_tran:, w_tran:, :]
            bbox[:, [0, 2]] = bbox[:, [0, 2]] - w_tran
            bbox[:, [1, 3]] = bbox[:, [1, 3]] - h_tran
            # 确保bbox中心点在图像内，因为中心点所在的格负责预测
            center_x = (bbox[:, 0] + bbox[:, 2]) / 2  # shape: nbox
            center_y = (bbox[:, 1] + bbox[:, 3]) / 2  # shape: nbox
            inImage = ((center_x > 0) & (center_x < w)) & ((center_y > 0) & (center_y < h))
            bbox = bbox[inImage, :]
            bbox[:, [0, 2]] = np.clip(bbox[:, [0, 2]], 0, w)  # 中心虽然还在图片内，但是边框可能会超过边界，要限制范围
            bbox[:, [1, 3]] = np.clip(bbox[:, [1, 3]], 0, h)
            category = category[inImage]
            return canvas, bbox, category
        return img, bbox, category

    def encoder(self, img, bbox, category):
        h, w, _ = img.shape
        w_scale_factor = 448 / w
        h_scale_factor = 448 / h
        bbox[:, [0, 2]] = bbox[:, [0, 2]] * w_scale_factor
        bbox[:, [1, 3]] = bbox[:, [1, 3]] * h_scale_factor
        h=w=448
        grid = 7
        target = np.zeros((7, 7, 30), dtype=np.float32)
        center_x = (bbox[:, 0] + bbox[:, 2]) / 2
        center_y = (bbox[:, 1] + bbox[:, 3]) / 2
        width_each_cell = w / grid
        height_each_cell = h / grid
        location_x = (center_x / width_each_cell).astype(np.int)
        location_y = (center_y / height_each_cell).astype(np.int)
        offset_x = center_x % width_each_cell
        norm_offset_x = offset_x / width_each_cell #相对坐标归一化
        offset_y = center_y % height_each_cell
        norm_offset_y = offset_y / height_each_cell
        target[location_y, location_x, 0] = norm_offset_x
        target[location_y, location_x, 1] = norm_offset_y
        obj_width = bbox[:, 2] - bbox[:, 0]
        obj_height = bbox[:, 3] - bbox[:, 1]
        target[location_y, location_x, 2] = obj_width / w
        target[location_y, location_x, 3] = obj_height / h
        target[location_y, location_x, 4] = 1
        target[location_y, location_x, 5:10] = target[location_y, location_x, :5]
        one_hot = self.one_hot(category)
        target[location_y, location_x, 10:] = one_hot
        return target

    def one_hot(self, label):
        one_hot_array = np.zeros((len(label), 20))
        row_index = np.arange(len(label))
        one_hot_array[row_index, label] = 1
        return one_hot_array
