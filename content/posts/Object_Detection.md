---
author: "xingyu"
author_link: ""
title: "Object_Detection"
date: 2021-03-27T19:50:19+08:00
lastmod: 2021-03-27T19:50:19+08:00
draft: false
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["Pytorch", "深度学习", "目标检测"]
categories: ["深度学习"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

系统学习目标检测，目标检测相关介绍

<!--more-->

## 什么是目标检测

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210327195711.png)

## 存在的挑战

* 环境影响
  * 光照
  * 模糊
* 密集 （ crowded ）
* 遮挡 （ occluded ）
* 重叠 （ highly overlapped ）
* 多尺度
  * 小目标 （ extremely small ）
  * 大目标 （ very large ）
* 小样本
* 旋转框

体积、 功耗、如何实时检测

## 目标检测发展

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210327200000.png)

### Two stage

1. 先由算法形成一系列作为样本的候选框，再通过卷积神经网络进行样本分类。
2. 对于Two-stage的目标检测网络，主要通过一个卷积神经网络来完成目标检测过程，其提取的是CNN卷积特征，在训练网络时，其主要训练两个部分，第一步是训练RPN网络，第二步是训练目标区域检测的网络。网络的准确度高、速度相对One-stage慢。

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210327200228.png)



![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210327200243.png)

基本流程：

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210327200334.png)

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210327200401.png)

## Anchor  Anchor-Based  Anchor-Free

### Anchor

* 预先设定好比例的一组候选框集合
* 滑动窗口提取

### Anchor Based Methods

* 使用Anchor提取目标框
* 在特征图上的每一个点，对Anchor进行分类和回归

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210327201409.png)

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210327201423.png)

**Anchor 缺点：**

* 手工设计   设置多少个？ 设置多少？ 长宽比如何设置？
* 数量多   如何解决正负样本不均衡问题？
* 超参数   如何对不同数据设置？

### Anchor-Free

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210327201632.png)

### 算法小结

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210327201649.png)

### 三类算法对比

|          | Anchor-Based单阶段 | Anchor-Based两阶段 | Anchor-Free |
| -------- | ------------------ | ------------------ | ----------- |
| 网络结构 | 简单               | 复杂               | 简单        |
| 精度     | 优                 | 更优               | 较优        |
| 预测速度 | 快                 | 稍慢               | 快          |
| 超参数   | 较多               | 多                 | 相对少      |
| 扩展性   | 一般               | 一班               | 较好        |

## 基本概念

**BBox: Bounding Box, 边界框**

* 绿色为人工标注的groud-truth, 红色为预测结果
* xyxy：左上+右下
* xywh：左上+宽高

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210327202801.png)

**Anchor:锚框**

* 人为设定不同长宽比、面积的先验框
* 在单阶段SSD检测算法中也称Prior box

**RoI: Region of Interest**

* 特定的感兴趣区域

**Region Proposal**

* 候选区域/框

**RPN: Region Proposal Network**

* Anchor-based 的两阶段提取候选框的网络

**IoU：Intersaction over Union**

* 评价预选框的质量，IoU越大则预测框与标注越接近。

**mAP**

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210327202821.png)

计算累计Precision、Recall

* P-R 曲线：以Precision、Recall为纵、横坐标点曲线
* AP（Average Precision）：某一类P-R曲线下的面积
* mAP（mean Average Precision）：所有类别AP平均

**NMS：非极大值抑制， Non-Maximum Suppression**

删除重叠、多余候选框

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210327203113.png)\

```python
def nms(dets, thresh):
	x1, y1, x2, y2 = dets[:,0], dets[:,1], dets[:,2], dets[:,3]
	scores = dets[:,4]
	areas = (x2 - x1 + 1) * (y2 - y1 + 1) # 各box 的面积
	order = scores.argsort()[::-1] # boxes 按照scores排序
	keep = []  # 记录保留下的boxes
	while order.size > 0:
		i = order[0] # score最大的box对应的index
		keep.append(i) #将本轮score最大的box都index 保留
		
		# 计算剩余boxes 与 当前box的重叠程度 IoU
		xx1 = np.maximum(x1[i], x1[order[1:]])
		yy1 = np.maximum(y1[i], y1[order[1:]])
		xx2 = np.minimum(x2[i], x2[order[1:]])
		yy2 = np.minimum(y2[i], y2[order[1:]])
		
		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w* h
		# IoU
		ovr = inter / (areas[i] + areas[order[1:]] - inter)
		# 保留IoU小于设定阈值的boxes
		inds = np.where(ovr < thresh)[0]
		order = order[inds + 1]
	return keep
```

