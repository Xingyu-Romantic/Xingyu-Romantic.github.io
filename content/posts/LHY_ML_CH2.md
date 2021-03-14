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

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210312222314.png)

###　Training data Loss

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

### Testing data Loss

**Overfitting：** traning data loss 足够小，  testing data loss  大

更有弹性的module更可能会overfitting

解决： 

* More training data

* 限制module  Less parameters, sharing parameters  例如  CNN  Fully-connected
* Early stopping
* **Regularization**
* Dropout

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210312223824.png)

**mismatch**

### P5  类神经网络训练不起来怎么办？

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210313231910.png)

Optimization Fails becase 

* local minima
* saddle point 

gradient不下降，卡在critical point 

**判断 critical point  到底是 local minima  or saddle point**

**Taler Series Approximation**
$$
L(\theta) \approx L(\theta') + (\theta - \theta')^T g + \frac 12(\theta-\theta')^TH(\theta-\theta')
$$
![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210313234134.png)

>Gradient **g** is a **vector**   图中 绿色
>
>$g = \nabla L(\theta')$          $g_i = \frac {\partial L(\theta')}{\partial \theta_i}$
>
>Hessian **H**  is a **matrix**  图中 红色
>
>$H_{ij} = \frac {\part^2}{\part \theta_i\part\theta_j}L(\theta')$



$ (\theta - \theta')^T g + \frac 12(\theta-\theta')^TH(\theta-\theta') =  v^THv$

**For  all v**

$v^THv> 0$ == > Around $\theta':L(\theta) > L(\theta') $    ==> Local minima

**For all v**

$v^THv < 0$ == > Around $\theta':L(\theta) < L(\theta') $    ==> Local maxima

Sometimes $v^THv>0$,  Sometimes $v^THv<0$   ==> Saddle point

**H may tell  us parameter update direction !**

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210314091003.png)

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210314092020.png)

