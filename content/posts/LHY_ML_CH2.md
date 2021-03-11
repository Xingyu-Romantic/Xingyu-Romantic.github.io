---
author: "xingyu"
author_link: ""
title: "LHY_ML_CH2"
date: 2021-03-11T22:08:43+08:00
lastmod: 2021-03-11T22:08:43+08:00
draft: true
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["机器学习", "Pytorch"]
categories: ["机器学习"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

第二章学习， 值得记录

<!--more-->

## 首先查看Loss

###　train_loss 不够低

**Model Bias**

The model is too simple

**Optimization issue**

gradient function 不给力

20-layer loss   <   56-layer loss   

* Gaining the insights from comparison

* Start from shallower networks(or other models), which are easier to optimize.

* if deeper networks do not botain smaller loss on training data, then there is optimization issue.

  

|             | 1 layer | 2 layer | 3 layer | 4 layer | 5 layer |
| ----------- | ------- | ------- | ------- | ------- | ------- |
| 2017 - 2020 | 0.28k   | 0.18k   | 0.14k   | 0.10k   | 0.34k   |

* Solution: More powerful optimization technology (next lecture)

```python
self.net = nn.Sequential(
            nn.Linear(n_feature, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
# Adam(self.parameters(), lr=1e-3)
# 500 epochs   128 batchsize    Loss : 0.98左右
```

