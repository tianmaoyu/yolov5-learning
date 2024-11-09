import torch
from torch import nn

class ConvBNSiLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.layer(x)


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, add_flag=True):
        super().__init__()
        self.add = add_flag
        self.cv1 = ConvBNSiLU(in_channels, out_channels, 1, 1, 0)
        self.cv2 = ConvBNSiLU(out_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        if self.add:
            return x + self.cv2(self.cv1(x))

        return self.cv2(self.cv1(x))


class C3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bottleneck_num=3, add_flag=True):
        super().__init__()
        temp = out_channels // 2
        self.cv1 = ConvBNSiLU(in_channels, temp, kernel_size, stride, padding)
        self.cv2 = ConvBNSiLU(in_channels, temp, kernel_size, stride, padding)
        bottleneck_list = [Bottleneck(temp, temp, add_flag) for _ in range(bottleneck_num)]
        self.bottlenecks = nn.Sequential(*bottleneck_list)
        self.cv3 = ConvBNSiLU(out_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.bottlenecks(self.cv2(x))
        x3 = torch.cat((x1, x2), dim=1)
        return self.cv3(x3)


class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = ConvBNSiLU(in_channels, out_channels // 2, 1, 1, 0)
        self.max_pool = nn.MaxPool2d(5, 1, padding=2)
        self.cv2 = ConvBNSiLU(out_channels * 2, out_channels, 1, 1, 0)

    def forward(self, x):
        o1 = self.cv1(x)
        o2 = self.max_pool(o1)
        o3 = self.max_pool(o2)
        o4 = self.max_pool(o3)
        out = torch.cat([o1, o2, o3, o4], dim=1)
        return self.cv2(out)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        # 注意：此模块本来是focus; 可以使用卷积 k=6,s=2,p=2 替代
        self.cv_p1_1 = ConvBNSiLU(3, 64, 6, 2, 2)

        self.cv_p2_2 = ConvBNSiLU(64, 128, 3, 2, 1)

        self.c3_C1_3 = C3(128, 128, 1, 1, 0, 3)
        self.cv_P3_4 = ConvBNSiLU(128, 256, 3, 2, 1)

        self.c3_C2_5 = C3(256, 256, 1, 1, 0, 6)
        self.cv_P4_6 = ConvBNSiLU(256, 512, 3, 2, 1)

        self.c3_C3_7 = C3(512, 512, 1, 1, 0, 9)
        self.cv_P5_8 = ConvBNSiLU(512, 1024, 3, 2, 1)

        #这里有点点不同： 官方的应该是 先 spp是第9层 第10 层是 c3
        self.c3_C4_9 = C3(1024, 1024, 1, 1, 0, 3)
        self.sppf_10 = SPPF(1024, 1024)

    def forward(self, x):
        o1 = self.cv_p1_1(x)
        o2 = self.cv_p2_2(o1)

        o3 = self.c3_C1_3(o2)
        o4 = self.cv_P3_4(o3)
        # feature1
        o5 = self.c3_C2_5(o4)
        o6 = self.cv_P4_6(o5)
        # feature2
        o7 = self.c3_C3_7(o6)

        o8 = self.cv_P5_8(o7)
        # todo 官方是先 spp 再 c3,待实验验证
        o9 = self.c3_C4_9(o8)
        # feature3
        o10 = self.sppf_10(o9)

        return o5, o7, o10


class YoloV5(nn.Module):
    def __init__(self,class_num,detect_flag=False):
        super().__init__()
        self.detect_flag=detect_flag
        self.backbone=Backbone()

        self.cv_1= ConvBNSiLU(1024,512,1,1,0)
        # 上采样使用的是 nearest
        self.up_2= nn.Upsample(scale_factor=2,mode="nearest")

        self.c3_4= C3(1024,512,1,1,0,bottleneck_num=3, add_flag=False)
        self.cv_5= ConvBNSiLU(512,256,1,1,0)
        self.up_6= nn.Upsample(scale_factor=2,mode="nearest")

        self.c3_8 = C3(512, 256, 1, 1, 0, add_flag=False)
        self.cv_9 = ConvBNSiLU(256, 256, 3, 2, 1)

        self.c3_11 = C3(512, 512, 1, 1, 0, add_flag=False)
        self.cv_12 = ConvBNSiLU(512, 512, 3, 2, 1)

        self.c3_14 = C3(1024, 1024, 1, 1, 0, add_flag=False)

        self.conv2d_feature1 = nn.Conv2d(256, (5 + class_num) * 3, 1, 1, 0)
        self.conv2d_feature2 = nn.Conv2d(512, (5 + class_num) * 3, 1, 1, 0)
        self.conv2d_feature3 = nn.Conv2d(1024, (5 + class_num) * 3, 1, 1, 0)
        # todo 推理部分
        self.detect=nn.Identity()


    def forward(self,x):
        feature1, feature2, feature3 = self.backbone(x)

        o1= self.cv_1(feature3)
        o2 = self.up_2(o1)
        o3= torch.cat([feature2,o2],dim=1)
        o4=self.c3_4(o3)
        o5=self.cv_5(o4)
        o6=self.up_6(o5)
        o7=torch.cat([feature1,o6],dim=1)
        o8=self.c3_8(o7)


        o9=self.cv_9(o8)
        o10=torch.cat([o5,o9],dim=1)
        o11=self.c3_11(o10)


        o12=self.cv_12(o11)
        o13=torch.cat([o1,o12],dim=1)
        o14=self.c3_14(o13)

        out1 = self.conv2d_feature1(o8)
        out2 = self.conv2d_feature2(o11)
        out3 = self.conv2d_feature3(o14)
        # todo 检测模块
        # if self.detect_flag:
        #     return

        return out1,out2,out3

if __name__ == '__main__':

    input_data=torch.randn(1,3,640,640)
    net=YoloV5(class_num=3)
    feature1,feature2,feature3= net(input_data)
    print(feature1.shape,feature2.shape,feature3.shape)