

import torch
from colorama import Fore
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from data import CocoDataset
from net import YoloV5
from loss import YoloV5Loss
from metric import YoloV5Metric
import utils
from tqdm import  tqdm


total_epoch = 300
logger = utils.config_logger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = CocoDataset("coco8/images/train", "coco8/labels/train", scaleFill=True)
eval_dataset = CocoDataset("coco8/images/train", "coco8/labels/train", scaleFill=True)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=2, collate_fn=CocoDataset.collate_fn)
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=2, collate_fn=CocoDataset.collate_fn)

logger.info(f"训练数据: {len(train_dataset)}  评估练数据: {len(eval_dataset)} ")

model = YoloV5(class_num=80).to(device)
loss = YoloV5Loss(class_num=80).to(device)
metric = YoloV5Metric()
optimizer = Adam(model.parameters(), 0.01)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epoch, eta_min=0.002)
scaler = torch.amp.GradScaler()

for epoch in range(total_epoch):

    # 训练--------------------------------------------------------------------
    train_bar = tqdm(train_dataloader, total=len(train_dataloader), leave=True, postfix=Fore.WHITE)
    train_bar.set_description(f"[{epoch}/{total_epoch}]")
    model.train()
    train_total_loss = torch.zeros(4).to(device)

    for step, (image, label) in enumerate(train_bar):
        image, label = image.to(device), label.to(device)

        predict_layer_list = model(image)
        loss_value, loss_detail = loss(predict_layer_list, label)
        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_total_loss += loss_detail
        # 日志
        box_loss, obj_loss, cls_loss, yolo_loss = train_total_loss.cpu().numpy().round(5)
        train_bar.set_postfix(loss=yolo_loss, box=box_loss, obj=obj_loss, cls=cls_loss, )

    # 动态调整学习率
    scheduler.step()


    # 验证 --------------------------------------------------------------------
    eval_bar = tqdm(eval_dataloader, total=len(eval_dataloader), leave=True, colour="green", postfix=Fore.GREEN)
    eval_bar.set_description(f"[{epoch}/{total_epoch}]")
    model.eval()
    eval_total_loss = torch.zeros(4).to(device)
    with torch.no_grad():
        for step, (image, label) in enumerate(eval_bar):
            image, label = image.to(device), label.to(device)

            predict_layer_list = model(image)

            loss_value, loss_detail = loss(predict_layer_list, label)

            metric.batch_update(image, label, predict_layer_list)

            eval_total_loss += loss_detail
            # 日志
            box_loss, obj_loss, cls_loss, yolo_loss = eval_total_loss.cpu().numpy().round(5)
            eval_bar.set_postfix(loss=yolo_loss, box=box_loss, obj=obj_loss, cls=cls_loss)

    # 性能指标
    metric_value = metric.compute()
    #重置性能指标
    metric.reset()
    p, r, f1, mAP05 = metric_value.cpu().numpy().round(5)
    logger.info(f"第epoch:{epoch}  P: {p}  R:{r}  F1:{f1}  mAP50:{mAP05} ")

    # 保存模型--------------------------------------------------------------------
    torch.save(model, f"out/yolov5-{epoch}.pth")
    logger.info( f"第epoch:{epoch} eval:{eval_total_loss.cpu().numpy().round(5)} train:{train_total_loss.cpu().numpy().round(5)}  pth: yolov5-{epoch}.pth")


