---
author: "Xingyu"
author_link: ""
title: "Pytorch basic usage"
date: 2020-12-12T20:58:39+08:00
lastmod: 2020-12-12T20:58:39+08:00
draft: false
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["cv"]
categories: ["cv"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

<!--more-->

## Tensors

Tensors(张量)的概念可以类比Numpy中的ndarrays，本质上就是一个多维数组，是任何运算和操作间数据流动的最基础形式。

| 函数                                  | 说明                                       |
| ------------------------------------- | ------------------------------------------ |
| `x = torch.empty(5,3)`                | 未初始化的5*3的空矩阵（张量）              |
| `torch.zeros`    `torch.ones`         |                                            |
| `torch.zeros(m, n, dtype=torch.long)` | 通过dtype属性来指定tensor的数据类型        |
| `torch.rand(5,3)`                     | 生成服从区间[0,1)均匀分布的随机张量        |
| `torch.randn(5,3)`                    | 生成服从均值为0、方差为1正态分布的随机张量 |
| `torch.tensor([5.5, 3])`              | 利用现有数据进行张量的初始化               |
| `x.size` -> `(row, col)`              | 获取tensor的size                           |
| `torch.arange(1,10,1)`                | 生成一定范围内等间隔的一维数组             |

## Operations

Operations(操作)涉及的语法和函数很多，大多数都是相通的，下面我们列举一些常用操作及其用法示例。

| 函数                                                         | 说明                   |
| ------------------------------------------------------------ | ---------------------- |
| `z3 = x+y`  `z3 = torch.add(x, y)`  `torch.add(x, y, out=z3)`  `y.add_(x)#覆盖在y上` | 张量加法               |
| `torch.sub(a,b)` `torch.mul(a,b)` `torch.div(a,b)`           | 减、点乘、除           |
| `torch.mm`                                                   | 矩阵乘法               |
| `torch.abs`                                                  | 计算绝对值             |
| `torch.pow`                                                  | 求幂                   |
| `x = torch.rand(3,5)`    -> `x[:, 1]`                        | 索引                   |
| `x.view(16)`  `x.view(-1, 8)`                                | -1时，根据其他维度推导 |
| `x.item()`                                                   | 获得对应Python形式     |
| `torch.clamp(a, -0.5, 0.5)`                                  | 对于越界的填充         |

## Numpy桥梁

Pytorch中可以很方便的将Torch的Tensor同Numpy的ndarray进行互相转换，相当于在Numpy和Pytorch间建立了一座沟通的桥梁，这将会让我们的想法实现起来变得非常方便。

注：Torch Tensor 和 Numpy ndarray 底层是分享内存空间的，也就是说**改变其中之一会同时改变另一个**（前提是你是在CPU上使用Torch Tensor）。

```python
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)

#tensor([1., 1., 1., 1., 1.])
#[1. 1. 1. 1. 1.]
```

将一个Numpy Array 转换为 Torch Tensor

```python
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b) 

#[2. 2. 2. 2. 2.]
#tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
```

注：所有CPU上的Tensors，除了CharTensor均支持与Numpy的互相转换

## CUDA Tensors

```python
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together! 

#tensor([0.5906], device='cuda:0')
#tensor([0.5906], dtype=torch.float64)
```