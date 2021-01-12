---
author: "xingyu"
author_link: ""
title: "Pytorch搭建Yolov3目标检测平台"
date: 2021-01-11T17:37:30+08:00
lastmod: 2021-01-11T17:37:30+08:00
draft: false
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["Pytorch", "Deep Learning"]
categories: ["Pytorch"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

记录 yolo3 相关知识，并用pytorch实现

<!--more-->

# yolov3 实现

## 预测部分

### 主干网络　DarkNet-53

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210111174132.png)

左边部分：

主干特诊提取网络 Darknet-53：

提取特诊

输入 416 * 416 * 3 -> 进行下采样，宽高会不断的被压缩，通道数不断的扩张-> 可以获得一堆的特征层，可以表示输入进来图片的特征。

**52, 52, 256** 会和 26 *26 * 256 特征层上采样之后的结果进行堆叠。

52 * 52 * 75 -> 52 * 52* 3 * 25 -> 52 * 52 * 3 * (20 + 1 +4)  

**26, 26, 512**   会和 13 * 13 * 1024 特征层上采样之后的结果进行堆叠， 是一个构建特征金字塔的过程，对堆叠之后的结果进行五次卷积， 卷积过后进行分类预测和回归预测。

26 * 26 * 75 -> 26 * 26 * 3 * 25 -> 26 * 26 * 3 * (20+ 1 + 4)

**13, 13, 1024**  五次卷积操作 

​	第一条路径：13, 13, 75  > 13, 13, 3, 25 > 13, 13, 3, 20(属于某一个类的概率), 1(是否有物体), 4(调整参数，画出框框) 

​	第二条路径：上采样， 上采样会使得特征层的宽高得到扩张。

### 残差网络

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210111181459.png)

如图即为一个残差块。

残差网络： 两条路径

1. 主干边：进行一系列的卷积还有激活函数，还有标准化的处理
2. 不经过任何处理，

两个处理过的内容会对处理过后的内容相加。

```python
class BasicBlock(LightningModule):
    def __init__(self, inplanes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        
        out += residual
        return out
```



## 构建DarkNet

```python
class DarkNet(LightningModule):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)
        
        self.layer1 = self._make_layer([32,64], layers[0])
        self.layer2 = self._make_layer([64,128], layers[1])
        self.layer3 = self._make_layer([128,256], layers[2])
        self.layer4 = self._make_layer([256,512], layers[3])
        self.layer5 = self._make_layer([512,1024], layers[4])
        
        self.layers_out_filters = [64, 128, 256, 512, 1024]
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.add.fill_(1)
                m.bias.data.zero_()
    def _make_layer(self, planes, blocks):
        layers = []
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,stride=2, padding=1, bias=False)))
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))
        
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))
        return nn.Sequential(OrderedDict(layers))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
    	# 由于后续yolo要用到这三个，估分开
        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    model = DarkNet([1, 2, 8, 8, 4])
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got[{}]".format(pretrained))
        return model
```



## 关键字

**特征金字塔：** 利用特征金字塔可以进行多尺度特征融合， 提取出更有效的特征。

**上采样：** 会使得特征层的宽高得到扩张。

**下采样：**宽高会不断压缩， 通道数不断的扩张， 可以获得一堆的特征层。



