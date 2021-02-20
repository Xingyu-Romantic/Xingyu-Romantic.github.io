---
author: "xingyu"
author_link: ""
title: "Tianchi_Defect_detection"
date: 2021-02-20T12:30:09+08:00
lastmod: 2021-02-20T12:30:09+08:00
draft: false
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["Pytorch", "数字图像处理"]
categories: ["Pytorch"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

参加全球人工智能技术创新大赛【热身赛一】，记录相关操作

由DataWhale引导

学会了基本Docker操作，并使用Docker训练网络。 Docker真香

<!--more-->

## 相关地址

比赛地址： https://tianchi.aliyun.com/competition/entrance/531864/information

阿里云镜像服务：https://cr.console.aliyun.com/

赵佬博客地址：https://blog.csdn.net/qq_26751117/article/details/113853150

Baseline：https://github.com/datawhalechina/team-learning-cv/tree/master/DefectDetection

Baseline讲解：https://mp.weixin.qq.com/s/-GpT6IBAYPMEUHmULis3rA

yolov5 官方教程：https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data

**基础 baseline   epochs = 70  batchsize=2 **  应该调整imgsize的 :)

![image-20210220125543509](/home/xingyu/.config/Typora/typora-user-images/image-20210220125543509.png)



## 赛题

###　数据介绍

花色布数据包含原始图片、模板图片和瑕疵的标注数据。标注数据详细标注出疵点所在的具体位置和疵点类别，，数据示例如下。

![enter image description here](https://tianchi-public.oss-cn-hangzhou.aliyuncs.com/public/files/forum/156519719116060451565197190838.jpeg)



### 评估指标

赛题分数计算方式:**0.2ACC+0.8mAP** 

**ACC**：是有瑕疵或无瑕疵的分类指标，考察瑕疵检出能力。

其中提交结果name字段中出现过的测试图片均认为有瑕疵，未出现的测试图片认为是无瑕疵。

**mAP**：参照PASCALVOC的评估标准计算瑕疵的mAP值。

参考链接：[https://github.com/rafaelpadilla/Object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics?spm=5176.12281978.0.0.478528815XGHTv)。具体逻辑见**evaluator**文件。

需要指出，本次大赛评分计算过程中，分别在检测框和真实框的交并比(IOU)在阈值0.1，0.3，0.5下计算mAP，最终mAP取三个值的平均值。

## 操作

>先下载baseline，熟悉Docker操作，有了基础分数

```python
# train.sh
python convertTrainLabel.py
python process_data_yolo_train.py # 改路径
python process_data_yolo_val.py # 原版
rm -rf ./convertor
python train.py --weights weights/yolov5x.pt --cfg models/yolo5x.yaml --batch-size 3

```

**注意：** 

* 坑1： 要自己下载权重，对应模型
* 坑2： pytroch1.7版本训练的话， 需要修改train.py，具体看赵佬博客
* 坑3： 处理数据只有val，没有train， 需要自行修改process_data_yolo.py 相关路径
* 坑4： Dockerfile 

```dockerfile
# Dockerfile
# Base Images
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/pytorch:1.4-cuda10.1-py3
## 把当前文件夹里的文件构建到镜像的根目录下
ADD . /
## 指定默认工作目录为根目录(需要把 run.sh 和生成的结果文件都放在该文件夹下,提交后才能运行)

WORKDIR /
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install scipy
RUN pip install tensorboard
RUN apt update && apt install -y libgl1-mesa-glx && apt-get install -y libglib2.0-0
RUN mkdir t
## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]
```

一系列配置好之后，

```shell
docker run --gpus all 
-v ~/Desktop/项目/defect_detection/train_data:/train_data 
-it c74b37c3c398 
sh train.sh
```

`-v` 映射目录文件    本机（绝对）地址:Docker（绝对）地址

`--gpus all`  使用gpu

`-it`  容器镜像id

`sh train.sh` 操作

## Docker 操作

本地操作

```
docker images # 列出本地镜像
docker ls # 列出正在运行的镜像
docker run -it xxxx(images id) /bin/bash # 进入镜像
ctrl + p + q # 正在运行的镜像 退出，不关闭
docker commit xxxx(正在运行镜像id) tag(例如：local_test:1.0)
```
推送镜像
```shell
$ sudo docker login --username=xxxx registry.cn-shenzhen.aliyuncs.com
$ sudo docker tag [ImageId] registry.cn-shenzhen.aliyuncs.com/xxxx:1.0
$ sudo docker push registry.cn-shenzhen.aliyuncs.com/xxxx:1.0
```

