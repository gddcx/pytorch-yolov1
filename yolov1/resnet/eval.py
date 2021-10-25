# -*- coding: utf-8 -*-
# Author: Changxing DENG
# @Time: 2021/10/13 15:26

import os
import glob
import cv2 as cv
import xml.dom.minidom as xdm
import numpy as np
import torch
from torchvision import transforms

from network import ResNet50Det
from collections import defaultdict

all_category = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
                     "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
                     "train", "tvmonitor"]


def calculate_map(prediction, target, thres=0.5):
    ap_list = []
    for category in all_category:
        all_pred = prediction[category]
        N = len(all_pred)  # the number of all prediction
        M = 0  # the number of gt
        key_tuple = target.keys()
        for k1, k2 in key_tuple:
            if k1 == category:
                M += len(target[(k1, k2)])
        P = np.zeros((N, 2))  # 第一列记录confidence， 第二列记录TP:1, FP:0
        Q = np.zeros((N, 2))  # 第一列存储Recall， 第二列存储 Precision
        for n, pred in enumerate(all_pred):
            filename = pred[0]
            confidence = pred[1]
            if (category, filename) not in target:
                continue
            GTBOX = target[(category, filename)]
            for gt_box in GTBOX:
                xmin = np.maximum(gt_box[0], pred[2])
                ymin = np.maximum(gt_box[1], pred[3])
                xmax = np.minimum(gt_box[2], pred[4])
                ymax = np.minimum(gt_box[3], pred[5])
                w = np.maximum(xmax - xmin, 0)
                h = np.maximum(ymax - ymin, 0)
                intersection = w * h
                union = (pred[4] - pred[2]) * (pred[5] - pred[3]) + (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]) - intersection
                iou = intersection / union
                if iou > thres:  # 一个prediction和两个gt的iou都大于thres怎么处理:只匹配第一个gt
                    P[n, 1] = 1
                    GTBOX.remove(gt_box)
                    if len(GTBOX) == 0:
                        del target[(category, filename)]
                    break
            P[n, 0] = confidence
        order = np.argsort(P[:, 0])[::-1]
        P = np.ascontiguousarray(P[order, :])
        cumsum_tp = np.cumsum(P[:, 1])
        Q[:, 0] = cumsum_tp / M  # recall
        Q[:, 1] = cumsum_tp / np.arange(1, N+1)  # precision
        # VOC 2007 with 11 points average
        ap = 0
        for x in np.arange(0.0, 1.1, 0.1):
            if np.sum(Q[:, 0] >= x) == 0: # if the maximal value of recall less than the point x, it's used to advoid throw error from np.max()
                continue
            ap += np.max(Q[:, 1][Q[:, 0] >= x])
        ap /= 11
        ap_list.append(ap)
    return ap_list
#
def nms(out, iou_thres=0.5, confidence_thres=0.1):
    #TODO: consider the category
    mask = out[4, :] > confidence_thres
    out = out[:, mask]
    xmin = out[0, :]
    ymin = out[1, :]
    xmax = out[2, :]
    ymax = out[3, :]

    areas = (xmax - xmin) * (ymax - ymin)
    order = np.ascontiguousarray(out[4, :].argsort()[::-1])  # argsort默认从小到大
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        # calculate intersection
        x1 = np.maximum(xmin[i], xmin[order[1:]])
        y1 = np.maximum(ymin[i], ymin[order[1:]])
        x2 = np.minimum(xmax[i], xmax[order[1:]])
        y2 = np.minimum(ymax[i], ymax[order[1:]])

        w = np.maximum(x2 - x1, 0)
        h = np.maximum(y2 - y1, 0)
        intersection = w * h
        iou = intersection / (areas[i] + areas[order[1:]] - intersection)  # the same order with variate "order"
        index = np.where(iou < iou_thres)[0]
        order = order[index+1]  # due to the index of the second element in variate "order" here is the first one.
    return out[:, keep]


def post_process(out):
    out = out.reshape(-1, 49)  # 30, 49
    confidence1 = out[4, :]
    confidence2 = out[9, :]
    mask = (confidence1 > confidence2)  # select box1 when mask is true, otherwise select box2
    filtered_out = np.zeros((25, 49))
    filtered_out[:5, mask] = out[:5, mask]
    filtered_out[:5, ~mask] = out[5:10, ~mask]
    filtered_out[5:, :] = out[10:, :]
    filtered_out = filtered_out.reshape((25, 7, 7))

    each_cell = 448 / 7
    # the top-left coordinate of grid cells on original images
    horizontal_lt = np.expand_dims(np.arange(7), 0) # 1, 7
    horizontal_lt = horizontal_lt.repeat(7, axis=0)  # 7, 7
    horizontal_lt = horizontal_lt * each_cell
    vertical_lt = np.expand_dims(np.arange(7), 1) #7, 1
    vertical_lt = vertical_lt.repeat(7, axis=1)  # 7, 7
    vertical_lt = vertical_lt * each_cell
    center_x = filtered_out[0, :, :] * each_cell + horizontal_lt
    center_y = filtered_out[1, :, :] * each_cell + vertical_lt
    width = filtered_out[2, :, :] * 448
    height = filtered_out[3, :, :] * 448
    xmin = np.clip(center_x - width / 2, 0, 448)
    ymin = np.clip(center_y - height / 2, 0, 448)
    xmax = np.clip(center_x + width / 2, 0, 448)
    ymax = np.clip(center_y + height / 2, 0, 448)

    res = np.zeros((6, 7, 7))
    res[0, :, :] = xmin
    res[1, :, :] = ymin
    res[2, :, :] = xmax
    res[3, :, :] = ymax
    res[4, :, :] = filtered_out[4, :, :] * np.max(filtered_out[5:, :, :], axis=0)
    res[5:, :, :] = np.argmax(filtered_out[5:, :, :], axis=0)
    res = res.reshape(6, -1)
    return res


def load_data(DATA_PATH):
    target = defaultdict(list)
    image_list = []
    voc2012_test_annotations = glob.glob(os.path.join(DATA_PATH, "Annotations", "*.xml"))
    for path in voc2012_test_annotations:
        DOMTree = xdm.parse(path)
        collection = DOMTree.documentElement
        filename = collection.getElementsByTagName("filename")[0].childNodes[0].data
        img_path = os.path.join(DATA_PATH, "JPEGImages", filename)
        image_list.append(img_path)
        object_ = collection.getElementsByTagName("object")
        for obj in object_:
            category = obj.getElementsByTagName("name")[0].childNodes[0].data
            bndbox = obj.getElementsByTagName("bndbox")[0]
            xmin = int(bndbox.getElementsByTagName("xmin")[0].childNodes[0].data)
            ymin = int(bndbox.getElementsByTagName("ymin")[0].childNodes[0].data)
            xmax = int(bndbox.getElementsByTagName("xmax")[0].childNodes[0].data)
            ymax = int(bndbox.getElementsByTagName("ymax")[0].childNodes[0].data)
            target[(category, filename)].append([xmin, ymin, xmax, ymax])
    return target, image_list


if __name__ == "__main__":
    # DATA_PATH="../../dataset/VOC2007/test/"
    DATA_PATH="D:\\dataset\\VOC2007\\test"
    # DATA_PATH="D:\\dataset\\VOC2012\\test"
    MODEL_PATH = "models/YOLO_ResNet50Det/288/271_75072.pth"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    target, image_list = load_data(DATA_PATH)
    temp_state_dict = torch.load(MODEL_PATH)
    state_dict = {k.replace("module.", ""): v for k, v in temp_state_dict.items()}
    net = ResNet50Det(pretrain=False)
    net.load_state_dict(state_dict)
    net = net.cuda()

    prediction = defaultdict(list)
    with torch.no_grad():
        net.eval()
        for i, image in enumerate(image_list):
            filename = os.path.basename(image)
            img = cv.imread(image)
            h, w, _ = img.shape
            img1 = cv.resize(img, (448, 448))
            img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
            img1 = transform(img1).unsqueeze(0).cuda()
            out = net(img1).squeeze(0)  #30, 7, 7
            out = out.cpu().numpy()
            out = post_process(out)  # 6, 7, 7
            out = nms(out, iou_thres=0.5, confidence_thres=0.1)
            h_factor = h / 448
            w_factor = w / 448
            out[[1, 3], :] *= h_factor
            out[[0, 2], :] *= w_factor
            #画出预测的框框
            # for j in range(out.shape[1]):
            #     xmin = int(out[0, j])
            #     ymin = int(out[1, j])
            #     xmax = int(out[2, j])
            #     ymax = int(out[3, j])
                # cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1, 1)
            for j, category_idx in enumerate(out[5, :]):
                # 画出gt的框框
                # for [xmin, ymin, xmax, ymax] in target[(all_category[int(category_idx)], filename)]:
                #     cv.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 255, 255), 1, 1)
                category = all_category[int(category_idx)]
                prediction[category].\
                    append([filename, out[4, j], out[0, j], out[1, j], out[2, j], out[3, j]])
            print("\r[{}]/[{}]".format(i, len(image_list)), end="")  # \r表示回到行的开头位置
            # cv.imshow("test", img)
            # cv.waitKey()
        # print("")
        # calculate AP
        ap_list = calculate_map(prediction, target)
        for idx, ap in enumerate(ap_list):
            print(all_category[idx], ap)
        print("mAP:", np.mean(ap_list))
