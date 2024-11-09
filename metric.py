import torch

import torch
import torchmetrics
import torchvision
from colorama import Fore
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics.detection  import MeanAveragePrecision
from tqdm import tqdm

import utils
from data import CocoDataset
from loss import YoloV5Loss

layer_anchors_list = torch.tensor([
    [[10, 13], [16, 30], [33, 23]],
    [[30., 61.], [62., 45.], [59., 119.]],
    [[116., 90.], [156., 198.], [373., 326.]]
])
layer_stride_list = torch.tensor([8, 16, 32])

# 计算 IoU
def calculate_iou(boxA, boxB):
    """

    :param boxA:  [xyxy]
    :param boxB:  [xyxy]
    :return:
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

class YoloV5Metric:

    def __init__(self):
        self.metric= MeanAveragePrecision()
        self.pred_dic_list=[]
        self.target_dic_list=[]


    def batch_update(self,batch_image:Tensor,batch_label,batch_predict_layer_list:[Tensor]):
        """
         批量计算
        :param batch_image:
        :param batch_label:
        :param batch_predict_layer_list:
        :return:
        """
        batch= batch_image.shape[0]

        # 分离出每张图片
        for b in range(batch):

            # 每张图片的 三个特征层也要分离出来
            predict_layer_list=[]

            for batch_predict_layer in batch_predict_layer_list:
                predict_layer=batch_predict_layer[b].unsqueeze(dim=0)
                predict_layer_list.append(predict_layer)

            label = batch_label[batch_label[:, 0] == b]
            image = batch_image[b].unsqueeze(dim=0)

            self.update(image,label,predict_layer_list)


    def update(self,image:Tensor,label, predict_layer_list: [Tensor] ):

        output_list = []
        device = predict_layer_list[0].device

        for i, layer in enumerate(predict_layer_list):
            layer_stride = layer_stride_list[i].to(device)
            layer_anchor = layer_anchors_list[i].to(device)

            # 变形 [bs,3*(5+class_num),h,w] ->  [bs,3, (5+class_num),h,w] ->  [bs,3,h,w,(5+class_num)]
            bs, channel, height, width = layer.shape

            data = layer.view(bs, 3, channel // 3, height, width).permute(0, 1, 3, 4, 2).contiguous()
            # predict=torch.sigmoid(predict)

            grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
            #  [ny, nx, 2]  -> [1,1,ny,nx,2]
            grid_xy = torch.stack([grid_x, grid_y], 2).view(1, 1, height, width, 2).float().to(device)
            # [3,2]->[1,3,1,1,2]
            anchor_wh = layer_anchor.unsqueeze(0).unsqueeze(2).unsqueeze(3)

            data = torch.sigmoid(data)

            xy = data[..., :2]
            wh = data[..., 2:4]
            obj_conf = data[..., 4:5]
            cls = data[..., 5:] * obj_conf

            output = torch.zeros([bs, 3, height, width, 7], device=device)
            score, label_index = torch.max(cls, dim=4, keepdim=True)
            # 还原到原图片大小
            output[..., 0:2] = (xy * 2.0 - 0.5 + grid_xy) * layer_stride
            output[..., 2:4] = (wh * 2.0) ** 2 * anchor_wh
            output[..., 4:5] = score
            output[..., 5:6] = label_index
            output[..., 6:7] = obj_conf

            # [1,3,h,w,6] -> [-1,7] 因为 检测时 bs=1  因为 batched_nms
            output = output.view(-1, 7)
            output_list.append(output)

        output = torch.cat(output_list, dim=0)
        # obj_cof 置信度过滤
        output = output[output[..., 6] > 0.01]

        output = utils.xywh2xyxy(output)
        # nms
        indices = torchvision.ops.batched_nms(boxes=output[..., :4],
                                              scores=output[..., 4],
                                              idxs=output[..., 5],
                                              iou_threshold=0.45)
        # (x,y,x,y,score,label_index,obj_confidence )
        output_nms = output[indices]

        pred_dic = self.build_pred_dic(output_nms)
        target_dic = self.build_target_dic(label, image)

        self.pred_dic_list.append(pred_dic)
        self.target_dic_list.append(target_dic)

        # self.metric.update([pred_dic],[target_dic])



    def build_pred_dic(self, output_nms):
        """
        :param output_nms:
        :return:
        {
            "boxes": torch.tensor([[100, 150, 200, 250], [300, 400, 500, 600]]),  # 边界框的坐标
            "scores": torch.tensor([0.9, 0.75]),
            "labels": torch.tensor([1, 2])
        }
        """

        data = {
            "boxes": output_nms[..., :4],
            "scores": output_nms[..., 4].T,
            "labels": output_nms[..., 5].T.long()
        }
        return data

    def build_target_dic(self, label, image:Tensor):
        """
        :param target:  (image_index,label_index,x,y,w,h)
        :return:
        {
            "boxes": torch.tensor([[110, 140, 200, 250], [310, 410, 510, 610]]),  # 边界框的坐标
            "labels": torch.tensor([1, 1])
        }
        """
        height, width = image.shape[2:]



        label[:, 2] *= width
        label[:, 3] *= height
        label[:, 4] *= width
        label[:, 5] *= height

        boxs = utils.xywh2xyxy(label[:, 2:])
        data = {
            "boxes": boxs,
            "labels": label[:, 1].T.long()
        }

        return  data

    def compute(self):
        precision, recall, f1_score =self.compute_P_R_F1()
        #单张图片上最大 max_detection_thresholds
        result_dict= self.metric(self.pred_dic_list,self.target_dic_list)
        mAP50=result_dict["map_50"]

        result=[precision,recall,f1_score,mAP50]
        return torch.tensor(result)
        # result= {"P":precision,"R":recall, "F1":f1_score,"mAP@50": mAP50.item() }
        # return result



    def reset(self):
        self.pred_dic_list=[]
        self.target_dic_list=[]
        self.metric.reset()


    def compute_P_R_F1(self,  iou_threshold = 0.5):

        # 初始化 TP, FP, FN 计数
        tp, fp, fn = 0, 0, 0

        pred_list=self.pred_dic_list
        gt_list=self.target_dic_list
        # 遍历每张图片
        for pred, gt in zip(pred_list, gt_list):
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]
            pred_labels = pred["labels"]

            gt_boxes = gt["boxes"]
            gt_labels = gt["labels"]

            # 存储已匹配的真实框索引
            matched_gt = set()

            # 遍历每个预测框
            for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
                matched = False
                for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):

                    if j in matched_gt:
                        continue  # 跳过已匹配的真实框
                    if pred_label == gt_label:
                        iou = calculate_iou(pred_box, gt_box)
                        if iou >= iou_threshold:
                            matched_gt.add(j)  # 匹配成功，记录真实框索引
                            tp += 1
                            matched = True
                            break
                if not matched:
                    fp += 1  # 没有匹配到的预测框记为 FP

            # 漏掉的
            fn += len(gt_boxes) - len(matched_gt)

        # 计算精确度、召回率和 F1 得分
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return  precision,recall,f1_score




if __name__ == '__main__':

    image_path = "coco8/images/train2017"
    label_path = "coco8/labels/train2017"
    dataset = CocoDataset(image_path, label_path,scaleFill=True)
    eval_dataloader = DataLoader(dataset=dataset, batch_size=2, collate_fn=CocoDataset.collate_fn)

    model_path = "./out/yolov5-719.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = YoloV5Loss(class_num=80).to(device)
    metric = YoloV5Metric()
    model = torch.load(model_path, map_location=device, weights_only=False)

    # 验证 --------------------------------------------------------------------
    eval_bar = tqdm(eval_dataloader, total=len(eval_dataloader), leave=True, colour="green", postfix=Fore.GREEN)
    model.eval()
    eval_total_loss = torch.zeros(4).to(device)
    metric_total_value = torch.zeros(4).to(device)

    with torch.no_grad():
        for step, (image, label) in enumerate(eval_bar):
            image, label = image.to(device), label.to(device)

            predict_layer_list = model(image)

            loss_value, loss_detail = loss(predict_layer_list, label)

            eval_total_loss += loss_detail

            metric.batch_update(image,label,predict_layer_list)
            # 日志 平均得分
            box_loss, obj_loss, cls_loss, yolo_loss = eval_total_loss.cpu().numpy().tolist()
            log = {
                "loss": yolo_loss,
                "box": box_loss,
                "obj": obj_loss,
                "cls": cls_loss,
            }
            eval_bar.set_postfix(log)

    metric_value = metric.compute()
    p, r, f1, mAP05 = metric_value.numpy().tolist()

    metric_log=f"P: {p} R:{r} F1:{f1} mAP50:{mAP05}"
    print(metric_log)


