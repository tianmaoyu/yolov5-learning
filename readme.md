# 掌控yolov5-用最简洁的代码复现

用最简单，最少的代码(700行左右)复现官方yolov5， 帮助想要学习它，搞明白它的同学一个参考。做一个对业余人士最友好的源码教程。

> 在 Visdrone2019 tain数据集训练，在val 上验证的效果：

![image-20241108230642280](data/image-20241108230642280.png)

![image-20241108230712951](data/image-20241108230712951.png)

## 一，运行项目

### 1.1，下载源代码

```shell
git clone https://github.com/tianmaoyu/yolov5-learning
cd yolov5-learning
#安装依赖
pip install -r requirements.txt
```

### 1.2，目录介绍

- coco8目录 : coco 数据集中的前 8张图片，要来训练和验证 模型是否能学习到东西
- data目录 :  visdrone 无人机数据集几张测试图片
- data/yolov5-156.pth : 在 VisDrone 数据集中训练的156 个epoch 权重和模型文件
- data/yolov5-structure.png ： 由噼里啪啦博主手画的yolov5网络图
- out目录  : 训练过程中，每一轮保存的 yolov5-{epoch}.pth 文件（记得清理，不然磁盘就满了）
- **data.py** : 数据加载，和预处理
- **loss.py** :  loss 函数定义
- **net.py :** 网络定义
- **metric.py:** 性能指标的定义
- **detect.py:** 检测头， 使用训练好的 模型进行 预测。

### 1.3，detect.py 运行

>  代码加载yolov5-156.pth 模型，和一张图片进行检测； 下面是默认效果

![image-20241108230712951](img/image-20241108230712951.png)



### 1.4，训练 train.py

默认使用 coco8 （八张）图片进行训练和验证。

![image-20241109192337071](img/image-20241109192337071.png)

**日志**：

![image-20241109192827228](img/image-20241109192827228.png)

**注意：**记得自行清理 .pth 文件

![image-20241109192433519](img/image-20241109192433519.png)



## 二，Pytorch基础

官方文档：https://pytorch.org/docs/stable/generated/torch.stack.html

stack 理解：https://blog.csdn.net/weixin_44201525/article/details/109769214

#### Tensor 基础

- `torch.randn`：生成一个随机张量，服从标准正态分布。

  ```python
  tensor_rand = torch.randn(2, 3)  # 2行3列的随机张量
  print(tensor_rand)
  ```

- `torch.arange`：生成一个等差数列。

  ```python
  tensor_arange = torch.arange(0, 10, 2)  # 从0开始，步长为2
  print(tensor_arange)
  ```

- `torch.zeros`：生成一个全零张量。

  ```python
  tensor_zeros = torch.zeros(2, 3)
  print(tensor_zeros)
  ```
  
- `view()` 方法用来改变张量的形状，==不会改变数据本身==，只是返回一个新的张量。要注意，新的形状必须是原张量形状的重排，不会改变数据的顺序。

  ```
  x = torch.randn(4, 4)
  x_reshaped = x.view(16)  # 改变形状为 1x16
  print(x_reshaped)
  ```

- `permute()` 方法可以重新排列维度的顺序，适用于需要调整维度顺序的情况，如图像数据（通常是 CHW 顺序，转为 HWC）。

  ```
  x = torch.randn(2, 3, 4)  # shape: (2, 3, 4)
  x_permuted = x.permute(2, 0, 1)  # shape: (4, 2, 3)
  print(x_permuted)
  ```

- `permute()` 操作会导致张量的==内存布局==发生变化，使用 `contiguous()` 可以确保返回一个有着连续内存的张量。

- ```
  x = torch.randn(2, 3, 4)
  x_permuted = x.permute(2, 0, 1)  # 改变了维度顺序
  x_contiguous = x_permuted.contiguous()
  print(x_contiguous.is_contiguous())  # 输出 True
  ```

- 增加维度：使用 `unsqueeze()` 来增加一个维度。

  ```python
  x = torch.tensor([1, 2, 3])
  x_unsqueezed = x.unsqueeze(0)  # 增加一个维度，变成 (1, 3)
  print(x_unsqueezed)
  ```

- 减少维度：使用 `squeeze()` 来去除长度为1的维度。

  ```python
  x = torch.tensor([[1], [2], [3]])
  x_squeezed = x.squeeze()  # 去除第一个维度
  print(x_squeezed)
  ```

- tensor 到 list

  ```
  tensor_to_list = tensor_from_list.tolist()
  print(tensor_to_list)
  ```

- list 到 tensor

  ```python
  my_list = [1, 2, 3]
  tensor_from_list = torch.tensor(my_list)
  print(tensor_from_list
  ```

- 普通索引:

  ```
  x = torch.tensor([1, 2, 3, 4, 5])
  print(x[2])  # 输出 3
  ```

- 高级索引

  ```
  x = torch.tensor([[1, 2], [3, 4], [5, 6]])
  print(x[:, 1])  # 输出第二列: tensor([2, 4, 6])
  ```

- `stack`：将多个张量沿==新==维度连接。

  ```python
  x = torch.randn(2, 3)
  y = torch.randn(2, 3)
  stacked = torch.stack([x, y], dim=0)  # 沿第0维堆叠
  print(stacked.shape)  # 输出 torch.Size([2, 2, 3])
  ```

- `cat`：沿==现==有维度连接张量。

  ```python
  x = torch.randn(2, 3)
  y = torch.randn(2, 3)
  concatenated = torch.cat([x, y], dim=1)  # 沿第1维拼接
  print(concatenated.shape)  # 输出 torch.Size([2, 6])
  ```

 `torch.min`, `torch.max`

```python
x = torch.tensor([1, 2, 3, 4])
min_value = torch.min(x)
max_value = torch.max(x)
print(min_value, max_value)  # 输出 1, 4
```

 `sigmoid` 函数

```python
x = torch.tensor([-1.0, 0.0, 1.0])
sigmoid_result = torch.sigmoid(x)
print(sigmoid_result)  # 输出 [0.2689, 0.5, 0.7311]
```

张量需要在同一设备上进行计算，因此张量之间的操作要求它们在相同的设备上。

```python
x_cpu = torch.tensor([1, 2, 3])
x_gpu = x_cpu.to('cuda')  # 将张量移动到GPU
result = x_gpu + x_gpu  # 计算时两个张量在同一设备上
print(result)
```

####  `torchvision.ops.nms`

`nms`（非极大值抑制）用于在目标检测中去除多余的重叠框。

```python
import torchvision.ops as ops

boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12]], dtype=torch.float)
scores = torch.tensor([0.9, 0.75, 0.8])
keep = ops.nms(boxes, scores, 0.5)
print(keep)  # 输出 [0, 2]
```

####  `torchmetrics.detection.MeanAveragePrecision`

mAP是衡量目标检测模型性能的重要指标。

```python
from torchmetrics.detection import MeanAveragePrecision

# 模拟一些检测框
preds = [{"boxes": torch.tensor([[0, 0, 10, 10], [1, 1, 12, 12]]),
          "scores": torch.tensor([0.9, 0.8]),
          "labels": torch.tensor([1, 2])}]
targets = [{"boxes": torch.tensor([[0, 0, 10, 10], [1, 1, 12, 12]]),
            "labels": torch.tensor([1, 2])}]

metric = MeanAveragePrecision()
mAP = metric(preds, targets)
print(mAP.compute())
```

####  `torchvision.ops.batched_nms`

`batched_nms` 适用于处理批量数据的非极大值抑制。

```python
boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [2, 2, 12, 12]], dtype=torch.float)
scores = torch.tensor([0.9, 0.75, 0.8])
batch_idx = torch.tensor([0, 0, 0])  # 只有一个batch
keep = ops.batched_nms(boxes, scores, batch_idx, 0.5)
print(keep)  # 输出 [0, 2]
```



## 三，Data源码讲解



## 四，Net源码讲解



## 五，Loss源码讲解



## 六，Detect源码讲解



## 七，Metric源码讲解



## 八，Train源码讲解



