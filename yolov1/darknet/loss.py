# -*- coding: utf-8 -*-
# Author: Changxing DENG
# @Time: 2021/10/11 15:06

import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(self, lambda_coord, lambda_noobj, grid=7):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.grid = grid

    def forward(self, input_, target):
        bs = input_.shape[0]
        input_ = input_.permute(0, 2, 3, 1).contiguous()  # bs, 7, 7, 30
        # TODO: to be optimized
        # eliminate the redundant dimension
        target = target.reshape(-1, 30)
        selected_target = torch.zeros_like(target[:, :25])
        selected_target[:, :5] = target[:, :5]
        selected_target[:, 5:] = target[:, 10:]
        hasObj = (selected_target[:, 4] >0)
        mask, iou = self.selectBox(input_, target)  # mask shape:bsx49, when false, select the first box, otherwise select the second box
        # mask, iou = self.selectBox_v2(input_, target)
        # Select one box between the predicted two boxes.
        input_ = input_.reshape(-1, 30)

        input_[:, [2, 3, 7, 8]] = torch.clamp(input_[:, [2, 3, 7, 8]], 1e-6, 1e6)

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
        classification_error = torch.sum(torch.sum((selected_input[:, 5:] - selected_target[:, 5:])**2, dim=-1), dim=0)  # 20211101

        loss = self.lambda_coord * coordinate_error + self.lambda_coord * size_error + confidence_error \
               + self.lambda_noobj * noobj_confidence_error + classification_error
        return loss / bs

    def selectBox(self, input_, target):
        input_clone = input_.clone() # bs, 7, 7, 30
        target_clone = target.clone() # bsx7x7, 30
        each_cell = 448 / self.grid
        # the top-left coordinate of grid cells on original images
        horizontal_lt = torch.arange(self.grid).unsqueeze(-1).unsqueeze(0).unsqueeze(0).to(input_.device) # 1, 1, 7, 1
        horizontal_lt = horizontal_lt.repeat(input_.shape[0], self.grid, 1, 2) # bs, 7, 7, 2
        horizontal_lt = horizontal_lt * each_cell
        vertical_lt = torch.arange(self.grid).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(input_.device) # 1, 7, 1, 1
        vertical_lt = vertical_lt.repeat(input_.shape[0], 1, self.grid, 2) # bs, 7, 7, 2
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
        iou = torch.zeros_like(iou1)
        iou[~m] = iou1[~m]
        iou[m] = iou2[m]
        return m, iou

    def selectBox_v2(self, input_, target):
        input_clone = input_.clone() # bs, 7, 7, 30
        input_clone = input_clone.reshape(-1, 30)
        target_clone = target.clone() # -1, 30
        target_clone[:, 2:4] = target_clone[:, 2:4] * 448
        input_clone[:, 2:4] = input_clone[:, 2:4] * 448
        input_clone[:, 7:9] = input_clone[:, 7:9] * 448
        bbox_gt = target_clone[:,:4]
        bbox_1 = input_clone[:, :4] # bs, 7, 7, 4
        bbox_2 = input_clone[:, 5:9]

        gt_x1 = torch.clamp(bbox_gt[:, 0] - bbox_gt[:, 2] / 2, 0, 448)
        gt_y1 = torch.clamp(bbox_gt[:, 1] - bbox_gt[:, 3] / 2, 0, 448)
        gt_x2 = torch.clamp(bbox_gt[:, 0] + bbox_gt[:, 2] / 2, 0, 448)
        gt_y2 = torch.clamp(bbox_gt[:, 1] + bbox_gt[:, 3] / 2, 0, 448)

        b1_x1 = torch.clamp(bbox_1[:, 0] - bbox_1[:, 2] / 2, 0, 448)
        b1_y1 = torch.clamp(bbox_1[:, 1] - bbox_1[:, 3] / 2, 0, 448)
        b1_x2 = torch.clamp(bbox_1[:, 0] + bbox_1[:, 2] / 2, 0, 448)
        b1_y2 = torch.clamp(bbox_1[:, 1] + bbox_1[:, 3] / 2, 0, 448)

        b2_x1 = torch.clamp(bbox_2[:, 0] - bbox_2[:, 2] / 2, 0, 448)
        b2_y1 = torch.clamp(bbox_2[:, 1] - bbox_2[:, 3] / 2, 0, 448)
        b2_x2 = torch.clamp(bbox_2[:, 0] + bbox_2[:, 2] / 2, 0, 448)
        b2_y2 = torch.clamp(bbox_2[:, 1] + bbox_2[:, 3] / 2, 0, 448)

        # 第一个bbox和gt的iou
        left = torch.max(gt_x1, b1_x1)
        top = torch.max(gt_y1, b1_y1)
        right = torch.min(gt_x2, b1_x2)
        bottom = torch.min(gt_y2, b1_y2)
        width = torch.max(right - left, torch.Tensor([1e-6]).to(left.device))  # 如果都是tensor就会自动广播并逐个元素比较，如果是普通整数/浮点数，则是从中找出最大
        height = torch.max(bottom - top, torch.Tensor([1e-6]).to(top.device))
        intersection = width * height
        union = (gt_x2 - gt_x1) * (gt_y2 - gt_y1) + (b1_x2 - b1_x1) * (b1_y2 - b1_y1) - intersection
        iou1 = intersection / union
        # 第二个bbox和gt的iou
        left = torch.max(gt_x1, b2_x1)
        top = torch.max(gt_y1, b2_y1)
        right = torch.min(gt_x2, b2_x2)
        bottom = torch.min(gt_y2, b2_y2)
        width = torch.max(right - left, torch.Tensor([1e-6]).to(left.device)) # TODO:不设置为1e-6会nan，原因是什么？
        height = torch.max(bottom - top, torch.Tensor([1e-6]).to(top.device))
        intersection = width * height
        union = (gt_x2 - gt_x1) * (gt_y2 - gt_y1) + (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - intersection
        iou2 = intersection / union
        # 选择框并保留对应iou
        m = iou1 < iou2
        iou = torch.zeros_like(iou1)
        iou[~m] = iou1[~m]
        iou[m] = iou2[m]
        return m, iou


if __name__ == "__main__":
    criterion = YOLOLoss(1, 2, 7)
    data = torch.randn(1, 30, 7, 7)
    target = torch.zeros(1, 7, 7, 30)
    target[:, :, :, :5] = torch.randn(1, 7, 7, 5)
    target[:, :, :, 5:10] = target[:, :, :, :5]
    target[:, :, :, 10:] = torch.randn(1, 7, 7, 20)
    print(criterion(data, target))