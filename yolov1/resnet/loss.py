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
        input_ = input_.permute(0, 2, 3, 1) # bs, 7, 7, 30
        mask, iou = self.selectBox(input_, target) # shape:bsx49, when true, select second box.
        input_ = input_.reshape(-1, 30)
        selected_input = torch.zeros_like(input_[:, :25])
        selected_input[~mask, :5] = input_[~mask, :5]
        selected_input[mask, :5] = input_[mask, 5:10]
        selected_input[:, 5:] = input_[:, 10:]
        target = target.reshape(-1, 30)
        selected_target = torch.zeros_like(target[:, :25])
        selected_target[:, :5] = target[:, :5]
        selected_target[:, 5:] = target[:, 10:]

        hasObj = (selected_target[:, 4] == 1)
        no_obj_input = selected_input[~hasObj, :]
        selected_input = selected_input[hasObj, :] # nx25
        selected_target = selected_target[hasObj, :]  # nx25
        iou = iou[hasObj]

        coordinate_error = torch.sum((selected_input[:, 0] - selected_target[:, 0])**2 + (selected_input[:, 1] - selected_target[:, 1])**2)
        size_error = torch.sum((torch.sqrt(selected_input[:, 2]) - torch.sqrt(selected_target[:, 2]))**2 + (torch.sqrt(selected_input[:, 3]) - torch.sqrt(selected_target[:, 3]))**2)
        confidence_error = torch.sum((selected_input[:, 4] - iou)**2)
        noobj_confidence_error = torch.sum((no_obj_input[:, 4])**2)
        classification_error = torch.sum((selected_input[:, 5:] - selected_target[:, 5:])**2)

        loss = self.lambda_coord * coordinate_error + self.lambda_coord * size_error + confidence_error \
               + self.lambda_noobj * noobj_confidence_error + classification_error
        return loss

    def selectBox(self, input_, target):
        input_clone = input_.clone() # bs, 7, 7, 30
        target_clone = target.clone() # bs, 7, 7, 30
        each_cell = 448 / 7
        horizontal_lt = torch.Tensor(list(range(7))).unsqueeze(-1).unsqueeze(0).unsqueeze(0)
        horizontal_lt = horizontal_lt.repeat(input_.shape[0], 7, 1, 2) # bs, 7, 7, 30
        horizontal_lt = horizontal_lt * each_cell
        vertical_lt = torch.Tensor(list(range(7))).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        vertical_lt = vertical_lt.repeat(input_.shape[0], 1, 7, 2) # bs, 7, 7, 30
        vertical_lt = vertical_lt * each_cell

        input_clone = input_clone.reshape(-1, 30)
        target_clone = target_clone.reshape(-1, 30)
        horizontal_lt = horizontal_lt.reshape(-1, 2)
        vertical_lt = vertical_lt.reshape(-1, 2)
        # 中心点坐标绝对值
        input_clone[:, [0, 1, 5, 6]] = input_clone[:, [0, 1, 5, 6]] * each_cell
        input_clone[:, [0, 5]] = input_clone[:, [0, 5]] + horizontal_lt
        input_clone[:, [1, 6]] = input_clone[:, [1, 6]] + vertical_lt
        target_clone[:, [0, 1, 5, 6]] = target_clone[:, [0, 1, 5, 6]] * each_cell
        target_clone[:, [0, 5]] = target_clone[:, [0, 5]] + horizontal_lt
        target_clone[:, [1, 6]] = target_clone[:, [1, 6]] + vertical_lt
        # 绝对宽高
        input_clone[:, [2, 3, 7, 8]] = input_clone[:, [2, 3, 7, 8]] * 448
        target_clone[:, [2, 3, 7, 8]] = target_clone[:, [2, 3, 7, 8]] * 448
        # 转为左上角和右下角坐标
        four_point_input = torch.zeros(input_clone.shape[0], 8).to(input_clone.device)
        four_point_input[:, [0, 4]] = input_clone[:, [0, 5]] - input_clone[:, [2, 7]] / 2
        four_point_input[:, [1, 5]] = input_clone[:, [1, 6]] - input_clone[:, [3, 8]] / 2
        four_point_input[:, [2, 6]] = input_clone[:, [0, 5]] + input_clone[:, [2, 7]] / 2
        four_point_input[:, [3, 7]] = input_clone[:, [1, 6]] + input_clone[:, [3, 8]] / 2
        four_point_input = torch.clamp(four_point_input, 0, 448) # limit the range from 0 to 448

        four_point_target = torch.zeros(target_clone.shape[0], 8).to(target_clone.device)
        four_point_target[:, [0, 4]] = target_clone[:, [0, 5]] - target_clone[:, [2, 7]] / 2
        four_point_target[:, [1, 5]] = target_clone[:, [1, 6]] - target_clone[:, [3, 8]] / 2
        four_point_target[:, [2, 6]] = target_clone[:, [0, 5]] + target_clone[:, [2, 7]] / 2
        four_point_target[:, [3, 7]] = target_clone[:, [1, 6]] + target_clone[:, [3, 8]] / 2
        four_point_target = torch.clamp(four_point_target, 0, 448)

        left = torch.max(four_point_input[:, 0], four_point_target[:, 0])
        right = torch.min(four_point_input[:, 2], four_point_target[:, 2])
        top = torch.max(four_point_input[:, 1], four_point_target[:, 1])
        bottom = torch.min(four_point_input[:, 3], four_point_target[:, 3])

        m = (left < right) & (top < bottom)
        intersection = (right - left)*(bottom - top)
        union = (four_point_input[:, 2] - four_point_input[:, 0]) * (four_point_input[:, 3] - four_point_input[:, 1]) \
                + (four_point_target[:, 2] - four_point_target[:, 0]) * (four_point_target[:, 3] - four_point_target[:, 1])
        iou1 = torch.zeros_like(left)
        iou1[m] = (intersection / (union - intersection))[m]  # bsx49

        left = torch.max(four_point_input[:, 4], four_point_target[:, 4]) # bs x 49
        right = torch.min(four_point_input[:, 6], four_point_target[:, 6])
        top = torch.max(four_point_input[:, 5], four_point_target[:, 5])
        bottom = torch.min(four_point_input[:, 7], four_point_target[:, 7])
        m = (left < right) & (top < bottom)
        intersection = (right - left) * (bottom - top)
        union = (four_point_input[:, 6] - four_point_input[:, 4]) * (four_point_input[:, 7] - four_point_input[:, 5]) \
                + ((four_point_target[:, 6] - four_point_target[:, 4]) * (four_point_target[:, 7] - four_point_target[:, 5]))
        iou2 = torch.zeros_like(left)
        iou2[m] = (intersection / (union - intersection))[m]

        mask = (iou1 < iou2)
        iou = torch.zeros_like(iou1)
        iou[~mask] = iou1[~mask]
        iou[mask] = iou2[mask]
        return mask, iou

if __name__ == "__main__":
    input_ = torch.rand(1, 30, 7, 7)
    input_[:, 4, :, :] = 1

    target = input_.clone()
    target = target.permute(0, 2, 3, 1)
    target[:, :, :, 4] = 1

    criterion = YOLOLoss(lambda_coord=5, lambda_noobj=0.5)
    loss = criterion(input_, target)
    print(loss)