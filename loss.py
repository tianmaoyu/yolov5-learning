from typing import List

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

class BBoxCIou:
    def __call__(self, box1, box2, eps=1e-7):
        """
        :param box1: 边界框1, [x, y, w, h]
        :param box2: 边界框2, [x, y, w, h]
        :param eps:
        :return: ciou
        """
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2

        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
        union = w1 * h1 + w2 * h2 - inter + eps

        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        # 计算最小包围框的对角线距离 (c^2)
        c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
        # 计算两个框的中心点 (center distance)
        d2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared

        v = (4 / torch.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        # with torch.no_grad():
        #     # alpha 不参与计算
        #     alpha = v / (v - iou + (1 + eps))
        alpha = v / (v - iou + (1 + eps))
        # CIoU
        return iou - (d2 / c2 + v * alpha)


class YoloV5Loss(nn.Module):
    def __init__(self, class_num=80):
        super().__init__()
        """
        :feature_map_num:  不同的特征层（不同分辨率）可以取 1，2，3 分别表示 :1: 80*80, 2:40*40; 3: 20*20
        """
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.bbox_ciou = BBoxCIou()
        self.class_num = class_num
        # layer上的 obj_loss weight
        self.layer_obj_loss_weight_list = [4.0, 1.0, 0.4]
        # 缩放比例
        self.layer_scale_list = [8, 16, 32]
        # layer 对应的 anchor 模板
        self.layer_anchors_list = [
            [[10, 13], [16, 30], [33, 23]],
            [[30., 61.], [62., 45.], [59., 119.]],
            [[116., 90.], [156., 198.], [373., 326.]]
        ]

    def __call__(self, predict_layer_list: List, label_list: Tensor):

        device = label_list.device
        batch_size = predict_layer_list[0].shape[0]
        box_loss = torch.zeros(1, device=device)
        cls_loss = torch.zeros(1, device=device)
        obj_loss = torch.zeros(1, device=device)

        for i, predict_layer in enumerate(predict_layer_list):
            layer_scale = self.layer_scale_list[i]
            layer_obj_loss_weight = self.layer_obj_loss_weight_list[i]
            layer_anchor = self.layer_anchors_list[i]

            layer_anchor = torch.tensor(layer_anchor, device=device).float() / layer_scale

            # 变形 [bs,3*(5+class_num),h,w] ->  [bs,3, (5+class_num),h,w] ->  [bs,3,h,w,(5+class_num)]
            bs, channel, height, width = predict_layer.shape
            predict_data = predict_layer.view(bs, 3, channel // 3, height, width).permute(0, 1, 3, 4, 2).contiguous()
            target, mask = self.build_target(predict_layer, label_list, layer_anchor)

            predict_obj = predict_data[..., 4]
            target_obj = torch.zeros(predict_obj.shape, device=device)

            # 是否匹配到正样本
            predict_positive = predict_data[mask]
            if predict_positive.shape[0] > 0:
                # 形状 [bs, 3, 80, 80,4]
                predict_box = self.build_predict_box(predict_data, layer_anchor)

                # 定位损失 :只计算正样本
                predict_box_positive = predict_box[mask]
                target_box_positive = target[mask][:, :4]
                # 这里有bug
                ciou = self.bbox_ciou(predict_box_positive, target_box_positive)
                box_loss += (1.0 - ciou).mean()

                # 根据ciou计算而得
                target_obj[mask] = ciou.detach().clamp(0)

                # 分类损失 :只计算正样本
                predict_cls_positive = predict_data[mask][:, 5:]
                target_cls_positive = target[mask][:, 5:]
                cls_loss += self.bce_loss(predict_cls_positive, target_cls_positive)

            # 目标性损失 :正负样本都计算，并且每个layer 的权重也不一样
            obj_loss += self.bce_loss(predict_obj, target_obj) * layer_obj_loss_weight

        # 总损失 三个损失 权重系数：1; 0.5 ;0.05
        obj_loss = 1.0 * obj_loss
        cls_loss = 0.5 * cls_loss
        box_loss = 0.05 * box_loss
        loss = obj_loss + cls_loss + box_loss

        return loss * batch_size, torch.cat([box_loss, obj_loss, cls_loss, loss]).detach() * batch_size

    def build_predict_box(self, predict_data, layer_anchor):
        """
         对模型 输出的 xywh 进行 "消除敏感度"
        :param predict_data:
        :param layer_anchor:
        :return:
        """
        # [bs,3,h,w,4]
        box = predict_data[..., :4].clone()

        xy = box[..., :2]
        box[..., :2] = 2 * torch.sigmoid(xy) - 0.5

        wh = box[..., 2:4]
        # 调整形状以匹配 [1,3,1,1,2]
        anchor_wh = layer_anchor.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        box[..., 2:4] = (2 * torch.sigmoid(wh)) ** 2 * anchor_wh

        return box

    def build_target(self, predict_layer: Tensor, batch_label_list, layer_anchor):
        """
        :param predict_layer:
        :param batch_label_list:   (image,label,x,y,w,h)
        :param layer_anchor:  [3,2]
        """
        device = predict_layer.device
        batch, channel, height, width = predict_layer.shape

        anchor_num = len(layer_anchor)
        # 初始化目标框张量，形状为 [bs, 3, 80, 80]
        target = torch.zeros([batch, anchor_num, height, width, self.class_num + 5], device=device)
        mask = torch.zeros([batch, anchor_num, height, width], device=device, dtype=torch.bool)

        # 分别循环批次，网格，anchor_num个层
        for b in range(batch):

            # 筛选出 image_index =b 得图片数据
            label_list = batch_label_list[batch_label_list[:, 0] == b]

            for label in label_list:

                class_index = label[1].long()
                true_x, true_y, true_w, true_h = label[2:6]
                # 还原到比例： 比如 特征图 80*80 ，
                layer_x = true_x * width
                layer_y = true_y * height
                layer_w = true_w * width
                layer_h = true_h * height

                # 向下取正 得网格  x,y 坐标
                grid_x = layer_x.long()
                grid_y = layer_y.long()

                for k in range(anchor_num):
                    anchor_w, anchor_h = layer_anchor[k]

                    w_ratio = layer_w / anchor_w
                    h_ratio = layer_h / anchor_h

                    # 两个的比例在 0.25-4 之间
                    if (0.25 < w_ratio < 4) and (0.25 < h_ratio < 4):
                        mask[b, k, grid_y, grid_x] = torch.tensor(True, device=device)
                        # 注意：类别是 one-hot 编码
                        target[b, k, grid_y, grid_x, 5 + class_index] = torch.tensor(1, device=device)
                        # 相对于网格坐标得偏移
                        mod_x, mod_y = layer_x - grid_x, layer_y - grid_y
                        # 计算 x,y,w,h, 注意：x,y 是相对于该grid_x,grid_y坐标的偏移
                        target[b, k, grid_y, grid_x, :4] = torch.tensor([mod_x, mod_y, layer_w, layer_h], device=device)

                        # 匹配 网格 left,top,right,down 是否满足
                        # left
                        if mod_x < 0.5 and grid_x > 0:
                            target[b, k, grid_y, grid_x - 1, :4] = torch.tensor([mod_x + 1, mod_y, layer_w, layer_h],
                                                                                device=device)
                            target[b, k, grid_y, grid_x - 1, 5 + class_index] = torch.tensor(1, device=device)
                            mask[b, k, grid_y, grid_x - 1] = torch.tensor(True, device=device)
                            # top
                        if mod_y < 0.5 and grid_y > 0:
                            target[b, k, grid_y - 1, grid_x, :4] = torch.tensor([mod_x, mod_y + 1, layer_w, layer_h],
                                                                                device=device)
                            target[b, k, grid_y - 1, grid_x, 5 + class_index] = torch.tensor(1, device=device)
                            mask[b, k, grid_y - 1, grid_x] = torch.tensor(True, device=device)
                        # right
                        if mod_x > 0.5 and grid_x < width - 1:
                            target[b, k, grid_y, grid_x + 1, :4] = torch.tensor([mod_x - 1, mod_y, layer_w, layer_h],
                                                                                device=device)
                            target[b, k, grid_y, grid_x + 1, 5 + class_index] = torch.tensor(1, device=device)
                            mask[b, k, grid_y, grid_x + 1] = torch.tensor(True, device=device)
                        # down
                        if mod_y > 0.5 and grid_y < height - 1:
                            target[b, k, grid_y + 1, grid_x, :4] = torch.tensor([mod_x, mod_y - 1, layer_w, layer_h],
                                                                                device=device)
                            target[b, k, grid_y + 1, grid_x, 5 + class_index] = torch.tensor(1, device=device)
                            mask[b, k, grid_y + 1, grid_x] = torch.tensor(True, device=device)

        return target, mask


if __name__ == '__main__':
    labels = torch.tensor([
        [45, 0.479492, 0.688771, 0.955609, 0.5955],
        [45, 0.736516, 0.247188, 0.498875, 0.476417],
        [50, 0.637063, 0.732938, 0.494125, 0.510583],
        [45, 0.339438, 0.418896, 0.678875, 0.7815],
        [49, 0.646836, 0.132552, 0.118047, 0.0969375],
        [49, 0.773148, 0.129802, 0.0907344, 0.0972292],
        [49, 0.668297, 0.226906, 0.131281, 0.146896],
        [49, 0.642859, 0.0792187, 0.148063, 0.148062]
    ])
    layer1 = torch.rand([1, 3 * 85, 80, 80])
    layer2 = torch.rand([1, 3 * 85, 40, 40])
    layer3 = torch.rand([1, 3 * 85, 20, 20])
    layer_list = [layer1, layer2, layer3]

    loss = YoloV5Loss()

    result = loss(layer_list, labels)
    print(result)
