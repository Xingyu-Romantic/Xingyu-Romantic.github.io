---
author: "xingyu"
author_link: ""
title: "Digital_Image_Process_2"
date: 2020-12-21T18:35:58+08:00
lastmod: 2020-12-21T18:35:58+08:00
draft: true
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["数字图像处理"]
categories: ["数字图像处理"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: false
---

>https://wenku.baidu.com/view/055297b327fff705cc1755270722192e45365883.html

# CH2  数字图像基础

## 图像的采样和量化

大多数传感器的输出是连续电压波形，为了产生一幅数字图像，需要把连续的感知数据转化为数字形式，这包括两种处理：取样和量化。

取样：图像空间坐标的数字化

量化：图像函数值（灰度值）的数字化

## 图像采样

空间坐标$(x,y)$的数字化被称为图像采样，确定水平和垂直方向上的像素个数M、N

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20201221190315.png)

## 图像的量化

函数取值的数字化被称为图像的量化，如量化到256个灰度级。

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20201221190357.png)

## 数字图像表示

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20201221190508.png)

## 图像的采样与数字图像的质量

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20201221190644.png)

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20201221190657.png)

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20201221190711.png)

## 图像内插

内插通常在图像放大、缩小、旋转和集合校正等任务中使用。

内插是用已知舒俱来估计位置的值的过程。

### 最近邻内插

假设一个大小为500$\times$500像素点一幅图像要放大1.5倍，即放大到$750\times750$像素。一种简单的 放大方法是，创建一个大小为$750\times750$像素点假想网格，网格的像素间隔完全与原图像的像素间隔相同，然后收缩网格，使它完全与原图想重叠。显然，收缩后的$750\times750$网格的像素间隔要小于原图想的像素间隔。为了对上覆图像中的每个点赋灰度值，我们在下伏原图像中找到最接近的像素，并把该像素的灰度赋给$750\times750$网格的新像素。为上覆网格中的所有点赋灰度值后，可将图像展开到指定的大小，得到放大后的图像。

