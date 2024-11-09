import logging

import cv2
import numpy as np
import  torch
from PIL import Image,ImageDraw
from torch import Tensor, nn
from torchvision.transforms import functional

def draw_rectangle(image:Image.Image,labels)-> Image.Image:
    """
    画矩形
    :param image:
    :param labels:
    :return:
    """
    # image= img.copy()
    width,height= image.size[0:2]
    draw = ImageDraw.Draw(image)
    for label in labels:
        x, y, w, h = label[2:]
        x1 = (x - w / 2) * width
        y1 = (y - h / 2) * height
        x2 = (x + w / 2) * width
        y2 = (y + h / 2) * height
        rectangle = (x1, y1, x2, y2)
        # 定义矩形的边框颜色和宽度
        outline_color = 'red'
        line_width = 3
        # 在图片上画矩形
        draw.rectangle(rectangle, outline=outline_color, width=line_width)

    return image

def draw_rectangle_xyxy(image:Image.Image,boxes)-> Image.Image:
    """
    画矩形
    :param image:
    :param labels:
    :return:
    """
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2=box
        rectangle = (x1, y1, x2, y2)
        draw.rectangle(rectangle, outline='red', width=3)
    return image

def letterbox(img:Image.Image, labels, new_shape=(640, 640),stride=32,scaleFill=False, color=(114, 114, 114)):
    """
     调整，填充 图片
    :param img:
    :param labels:
    :param new_shape:
    :param stride:
    :param color:
    :return:
    """
    # Resize and pad image while meeting stride-multiple constraints
    img = np.array(img)
    shape = img.shape[:2]  # current shape [height, width]

    # Scale ratio (new / old)
    min_ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])


    new_unpad = int(round(shape[1] * min_ratio)), int(round(shape[0] * min_ratio))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
     # minimum rectangle
    dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    ratio = min_ratio, min_ratio

    # 缩放到 640 * 640
    if scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    # 重新计算label
    height, width = shape
    new_height, new_width = img.shape[:2]
    new_labels = []
    for label in labels:
        image_index, label_index, x, y, w, h = label
        # Apply scaling and padding adjustments
        new_x = (x * width * ratio[0] + left) / new_width
        new_y = (y * height * ratio[1] + top) / new_height
        new_w = w * width * ratio[0] / new_width
        new_h = h * height * ratio[1] / new_height
        new_label = [image_index, label_index, new_x, new_y, new_w, new_h]
        new_labels.append(new_label)

    return img, new_labels


def image_pad(image: Tensor,scale=32) -> Tensor:
    # 填充到最小能被 scale 整除
    width,height= functional.get_image_size(image)
    pad_height = (scale - height % scale) % scale
    pad_width = (scale - width % scale) % scale
    # 表示在左、右,上、下、四个方向 mode：指定填充模式，可以是 “constant”、“reflect” 或 “replicate”；
    pad_image = nn.functional.pad(image, (0, pad_width , 0, pad_height), mode='reflect')
    return pad_image



def xywh2xyxy(data: Tensor):
    temp = data.clone()
    x1 = temp[..., 0] - temp[..., 2] / 2  # x1 = center_x - width / 2
    y1 = temp[..., 1] - temp[..., 3] / 2  # y1 = center_y - height / 2
    x2 = temp[..., 0] + temp[..., 2] / 2  # x2 = center_x + width / 2
    y2 = temp[..., 1] + temp[..., 3] / 2  # y2 = center_y + height / 2

    data[..., 0] = x1
    data[..., 1] = y1
    data[..., 2] = x2
    data[..., 3] = y2
    return data


class BBoxCIoU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, box1, box2, eps=1e-7) -> Tensor:
        """
        计算两个边界框之间的 CIoU 值  box1, box2   是 xywh
        :param box1: 边界框1, [x, y, w, h]
        :param box2: 边界框2, [x, y, w, h]
        :param xywh
        :return: CIoU 值
        """

        # 计算交集区域的坐标 (intersection)
        x1_inter = torch.max(box1[..., 0], box2[..., 0])  # 交集区域的左上角 x 坐标
        y1_inter = torch.max(box1[..., 1], box2[..., 1])  # 交集区域的左上角 y 坐标
        x2_inter = torch.min(box1[..., 2], box2[..., 2])  # 交集区域的右下角 x 坐标
        y2_inter = torch.min(box1[..., 3], box2[..., 3])  # 交集区域的右下角 y 坐标

        # 计算交集区域的宽度和高度
        inter_width = torch.clamp(x2_inter - x1_inter, min=0)
        inter_height = torch.clamp(y2_inter - y1_inter, min=0)

        # 交集面积
        inter_area = inter_width * inter_height

        # 计算每个框的面积
        box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
        box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

        # 计算并集面积 (Union) eps 防止除0错误
        union_area = box1_area + box2_area - inter_area + eps

        # 计算 IoU
        iou = inter_area / union_area

        # 计算最小包围框的坐标 (enclosing box)
        x1_enclose = torch.min(box1[..., 0], box2[..., 0])
        y1_enclose = torch.min(box1[..., 1], box2[..., 1])
        x2_enclose = torch.max(box1[..., 2], box2[..., 2])
        y2_enclose = torch.max(box1[..., 3], box2[..., 3])

        # 计算最小包围框的对角线距离 (c^2)
        enclose_width = x2_enclose - x1_enclose
        enclose_height = y2_enclose - y1_enclose
        c2 = enclose_width ** 2 + enclose_height ** 2 + eps

        # 计算两个框的中心点 (center distance)
        center_box1_x = (box1[..., 0] + box1[..., 2]) / 2
        center_box1_y = (box1[..., 1] + box1[..., 3]) / 2
        center_box2_x = (box2[..., 0] + box2[..., 2]) / 2
        center_box2_y = (box2[..., 1] + box2[..., 3]) / 2
        d2 = (center_box1_x - center_box2_x) ** 2 + (center_box1_y - center_box2_y) ** 2

        # 计算 v 和 α (避免除以0的情况)
        w1 = box1[..., 2] - box1[..., 0]
        h1 = box1[..., 3] - box1[..., 1]
        w2 = box2[..., 2] - box2[..., 0]
        h2 = box2[..., 3] - box2[..., 1]

        v = (4 / (torch.pi ** 2)) * (torch.atan(w1 / (h1 + eps)) - torch.atan(w2 / (h2 + eps))) ** 2
        alpha = v / (1 - iou + v + eps)

        # 计算 CIoU
        ciou = iou - (d2 / c2) - alpha * v

        return ciou


def config_logger(name="train"):
    # 设置日志的基本配置
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)

    # 创建一个handler，用于写入日志文件
    file_handler = logging.FileHandler('app.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # # 再创建一个handler，用于输出到控制台
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(formatter)
    # logger.addHandler(stream_handler)

    return logger
