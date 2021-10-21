# -*- coding: utf-8 -*-
# Author: Changxing DENG
# @Time: 2021/10/11 15:06

import torch
import torch.nn as nn

class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord, lambda_noobj):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, input_, target):
        bs = input_.shape[0]
        input_ = input_.permute(0, 2, 3, 1).contiguous()  # bs, 7, 7, 30
        #TODO: to be optimized

        # eliminate the redundant dimension
        target = target.reshape(-1, 30)
        selected_target = torch.zeros_like(target[:, :25])
        selected_target[:, :5] = target[:, :5]
        selected_target[:, 5:] = target[:, 10:]
        hasObj = (selected_target[:, 4] == 1)

        mask, iou = self.selectBox(input_, target)  # mask shape:bsx49, when false, select the first box, otherwise select the second box
        # Select one box between the predicted two boxes.
        input_ = input_.reshape(-1, 30)
        selected_input = torch.zeros_like(input_[:, :25]) + 1e-6
        selected_input[~mask, :5] = input_[~mask, :5]
        selected_input[mask, :5] = input_[mask, 5:10]
        selected_input[:, 5:] = input_[:, 10:]

        # the gird cells contain object
        # no_obj_input = selected_input[~hasObj, :]
        selected_input = selected_input[hasObj, :] # nx25
        selected_target = selected_target[hasObj, :]  # nx25
        iou = iou[hasObj]

        noobj_confidence_error = torch.sum((input_[~hasObj][:, [4, 9]])**2) # 没有物体的cell中所有confidence都要被惩罚

        coordinate_error = torch.sum((selected_input[:, 0] - selected_target[:, 0])**2 + (selected_input[:, 1] - selected_target[:, 1])**2)
        size_error = torch.sum((torch.sqrt(selected_input[:, 2]) - torch.sqrt(selected_target[:, 2]))**2 + (torch.sqrt(selected_input[:, 3]) - torch.sqrt(selected_target[:, 3]))**2)
        confidence_error = torch.sum((selected_input[:, 4] - iou)**2)
        # noobj_confidence_error = torch.sum((no_obj_input[:, 4])**2) # 只惩罚根据iou选的cell，但是对没有物体的cell来说，不应该存在iou这一说法
        classification_error = torch.sum((selected_input[:, 5:] - selected_target[:, 5:])**2)

        loss = self.lambda_coord * coordinate_error + self.lambda_coord * size_error + confidence_error \
               + self.lambda_noobj * noobj_confidence_error + classification_error
        return loss / bs

    def selectBox(self, input_, target):
        input_clone = input_.clone() # bs, 7, 7, 30
        target_clone = target.clone() # bs, 7, 7, 30
        each_cell = 448 / 7
        # the top-left coordinate of grid cells on original images
        horizontal_lt = torch.arange(7).unsqueeze(-1).unsqueeze(0).unsqueeze(0).to(input_.device) # 1, 1, 7, 1
        horizontal_lt = horizontal_lt.repeat(input_.shape[0], 7, 1, 2) # bs, 7, 7, 2
        horizontal_lt = horizontal_lt * each_cell
        vertical_lt = torch.arange(7).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(input_.device) # 1, 7, 1, 1
        vertical_lt = vertical_lt.repeat(input_.shape[0], 1, 7, 2) # bs, 7, 7, 2
        vertical_lt = vertical_lt * each_cell

        input_clone = input_clone.reshape(-1, 30)
        target_clone = target_clone.reshape(-1, 30)
        horizontal_lt = horizontal_lt.reshape(-1, 2)
        vertical_lt = vertical_lt.reshape(-1, 2)
        # remap coordinate of center point to original image
        input_clone[:, [0, 1, 5, 6]] = input_clone[:, [0, 1, 5, 6]] * each_cell # due to being normalized in dataset.py
        input_clone[:, [0, 5]] = input_clone[:, [0, 5]] + horizontal_lt
        input_clone[:, [1, 6]] = input_clone[:, [1, 6]] + vertical_lt
        target_clone[:, [0, 1, 5, 6]] = target_clone[:, [0, 1, 5, 6]] * each_cell
        target_clone[:, [0, 5]] = target_clone[:, [0, 5]] + horizontal_lt
        target_clone[:, [1, 6]] = target_clone[:, [1, 6]] + vertical_lt
        # remap height and width to original image
        input_clone[:, [2, 3, 7, 8]] = input_clone[:, [2, 3, 7, 8]] * 448
        target_clone[:, [2, 3, 7, 8]] = target_clone[:, [2, 3, 7, 8]] * 448
        # convert center point, height and width to left-top and right-bottom point
        four_point_input = torch.zeros(input_clone.shape[0], 8).to(input_clone.device) + 1e-6
        four_point_input[:, [0, 4]] = input_clone[:, [0, 5]] - input_clone[:, [2, 7]] / 2
        four_point_input[:, [1, 5]] = input_clone[:, [1, 6]] - input_clone[:, [3, 8]] / 2
        four_point_input[:, [2, 6]] = input_clone[:, [0, 5]] + input_clone[:, [2, 7]] / 2
        four_point_input[:, [3, 7]] = input_clone[:, [1, 6]] + input_clone[:, [3, 8]] / 2
        four_point_input = torch.clamp(four_point_input, 0, 448) # limit the range from 0 to 448

        four_point_target = torch.zeros(target_clone.shape[0], 8).to(target_clone.device) + 1e-6
        four_point_target[:, [0, 4]] = target_clone[:, [0, 5]] - target_clone[:, [2, 7]] / 2
        four_point_target[:, [1, 5]] = target_clone[:, [1, 6]] - target_clone[:, [3, 8]] / 2
        four_point_target[:, [2, 6]] = target_clone[:, [0, 5]] + target_clone[:, [2, 7]] / 2
        four_point_target[:, [3, 7]] = target_clone[:, [1, 6]] + target_clone[:, [3, 8]] / 2
        four_point_target = torch.clamp(four_point_target, 0, 448)
        # calculate iou of the first box
        left = torch.max(four_point_input[:, 0], four_point_target[:, 0])
        right = torch.min(four_point_input[:, 2], four_point_target[:, 2])
        top = torch.max(four_point_input[:, 1], four_point_target[:, 1])
        bottom = torch.min(four_point_input[:, 3], four_point_target[:, 3])
        w = torch.max(right - left, torch.Tensor([1e-6]).to(right.device))
        h = torch.max(bottom - top, torch.Tensor([1e-6]).to(bottom.device))
        intersection = w * h
        union = (four_point_input[:, 2] - four_point_input[:, 0]) * (four_point_input[:, 3] - four_point_input[:, 1]) \
                + (four_point_target[:, 2] - four_point_target[:, 0]) * (four_point_target[:, 3] - four_point_target[:, 1])
        iou1 = (intersection / (union - intersection))
        # calculate iou of the second box
        left = torch.max(four_point_input[:, 4], four_point_target[:, 4]) # bs x 49
        right = torch.min(four_point_input[:, 6], four_point_target[:, 6])
        top = torch.max(four_point_input[:, 5], four_point_target[:, 5])
        bottom = torch.min(four_point_input[:, 7], four_point_target[:, 7])
        w = torch.max(right - left, torch.Tensor([1e-6]).to(right.device))
        h = torch.max(bottom - top, torch.Tensor([1e-6]).to(bottom.device))
        intersection = w * h
        union = (four_point_input[:, 6] - four_point_input[:, 4]) * (four_point_input[:, 7] - four_point_input[:, 5]) \
                + ((four_point_target[:, 6] - four_point_target[:, 4]) * (four_point_target[:, 7] - four_point_target[:, 5]))
        iou2 = intersection / (union - intersection)

        m = (iou1 < iou2)
        iou = torch.zeros_like(iou1) + 1e-6
        iou[~m] = iou1[~m]
        iou[m] = iou2[m]
        return m, iou



# 测试
def onehot(label):
    one_hot_array = np.zeros((len(label), 20))
    row_index = np.arange(len(label))
    one_hot_array[row_index, label] = 1
    return one_hot_array

def encoder(img, bbox, category):
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
    one_hot = onehot(category)
    target[location_y, location_x, 10:] = one_hot
    return target

if __name__ == "__main__":
    import numpy as np
    import cv2 as cv
    loss = YOLOLoss(lambda_coord=5, lambda_noobj=0.5)
    img = cv.imread("D:\\dataset\\VOC2007\\test\\JPEGIMAGES\\000001.jpg")
    # bbox = np.array([[48, 240, 195, 371], [8, 12, 352, 498]])
    bbox = np.array([[48, 240, 195, 371]]) # map tp 448x448：60, 215, 247, 332
    category = np.array([11])
    target = encoder(img, bbox, category)
    bbox1 = np.array([[49, 241, 190, 370]])  # map tp 448x448：62, 215, 241, 331
    bbox2 = np.array([[118, 240, 265, 371]]) # map tp 448x448：149, 215, 336, 332
    # bbox1 = np.array([[49, 241, 190, 370], [7, 11, 353, 499]])
    # bbox2 = np.array([[118, 240, 265, 371], [78, 12, 422, 498]])
    input_1 = encoder(img, bbox1, category)
    input_2 = encoder(img, bbox2, category)
    input_ = input_1.copy()
    input_[4, 2, 5:10] = input_2[4, 3, :5]
    input_ = torch.from_numpy(input_)
    target = torch.from_numpy(target)
    input_ = input_.permute(2, 0, 1).unsqueeze(0)
    target = target.unsqueeze(0)
    loss(input_=input_, target=target)