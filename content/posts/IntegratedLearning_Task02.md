---
author: "xingyu"
author_link: ""
title: "集成学习_Task02"
date: 2021-03-16T13:30:59+08:00
lastmod: 2021-03-16T13:30:59+08:00
draft: false
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["Algorithm", "机器学习"]
categories: ["机器学习"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

Datawhale 集成学习 Task02 笔记

<!--more-->

一个完整的机器学习项目分为以下步骤：

* 明确项目任务： 回归 / 分类
* 收集数据集并选择合适的特征。
* 选择度量模型性能的指标。
* 选择具体的模型并进行训练以优化模型。
* 评估模型的性能并调参。

数据约定如下：

* 第i个样本：$x_i=(x_{i1},x_{i2},...,x_{ip},y_i)^T,i=1,2,...,N$     
* 因变量$y=(y_1,y_2,...,y_N)^T$        
* 第k个特征:$x^{(k)}=(x_{1k},x_{2k},...,x_{Nk})^T$     
* 特征矩阵$X=(x_1,x_2,...,x_N)^T$

## 构建完整回归项目

采用Boston房价数据集。

```python
from sklearn import datasets
boston = datasets.load_boston()     # 返回一个类似于字典的类
X = boston.data
y = boston.target
features = boston.feature_names
boston_data = pd.DataFrame(X,columns=features)
boston_data["Price"] = y
boston_data.head()
```

>各个特征的相关解释：
>
>- CRIM：各城镇的人均犯罪率
>- ZN：规划地段超过25,000平方英尺的住宅用地比例
>- INDUS：城镇非零售商业用地比例
>- CHAS：是否在查尔斯河边(=1是)
>- NOX：一氧化氮浓度(/千万分之一)
>- RM：每个住宅的平均房间数
>- AGE：1940年以前建造的自住房屋的比例
>- DIS：到波士顿五个就业中心的加权距离
>- RAD：放射状公路的可达性指数
>- TAX：全部价值的房产税率(每1万美元)
>- PTRATIO：按城镇分配的学生与教师比例
>- B：1000(Bk - 0.63)^2其中Bk是每个城镇的黑人比例
>- LSTAT：较低地位人口
>- Price：房价

**度量模型性能的指标：**

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210316135313.png)

* MSE均方误差：$\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2.$

  ```python
  from sklearn.metrics import mean_squared_error
  y_true = [3, -0.5, 2, 7]
  y_pred = [2.5, 0.0, 2, 8]
  mean_squared_error(y_true, y_pred)
  # 0.375
  ```

* MAE平均绝对误差：$\text{MAE}(y, \hat{y}) = \frac 1{n_{samples}}\sum^{n_{samples} - 1}_{i =0}|y_i - \hat{y}_i|$

  ```python
  from sklearn.metrics import mean_absolute_error
  y_true = [3, -0.5, 2, 7]
  y_pred = [2.5, 0.0, 2, 8]
  mean_absolute_error(y_true, y_pred)
  # 0.5
  ```

* $R^2$决定系数：$R^2(y, \hat{y}) = 1 - \frac {\sum^n_{i=1}(y_i - \hat{y}_i)^2}{\sum^n_{i=1}(y_i-\hat{y})^2}$

  ```python
  from sklearn.metrics import r2_score
  y_true = [3, -0.5, 2, 7]
  y_pred = [2.5, 0.0, 2, 8]
  r2_score(y_true, y_pred)
  # 0.948...
  ```

* 解释方差得分:$explained\_{}variance(y, \hat{y}) = 1 - \frac{Var\{ y - \hat{y}\}}{Var\{y\}}$

  The best possible score is 1.0, lower values are worse.

  ```python
  from sklearn.metrics import explained_variance_score
  y_true = [3, -0.5, 2, 7]
  y_pred = [2.5, 0.0, 2, 8]
  explained_variance_score(y_true, y_pred)
  # 0.957...
  ```

* 点击查看更多。。。https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics

### 线性回归模型



假设：数据集$D = \{(x_1,y_1),...,(x_N,y_N) \}$，$x_i \in R^p,y_i \in R,i = 1,2,...,N$，$X = (x_1,x_2,...,x_N)^T,Y=(y_1,y_2,...,y_N)^T$ 

则 假设 X 和 Y 之间存在线性关系， $\hat{y} = f(w) = w^Tx$  $w \in R^{n}, x \in R^{p} $

有关这部分的详细解释： [白板推到系列](https://www.bilibili.com/video/BV1aE411o7qd?p=9)

(a) 最小二乘估计：
$$
L(w) = \sum\limits_{i=1}^{N}||w^Tx_i - y_i||^2 = \sum\limits_{i=1}^{N}(w^Tx_i-y_i)^2 =(w^TX^T-Y^T)(w^TX^T-Y^T)^T \\ 
= w^TX^TXw - w^TX^TY - Y^TXw + Y^TY \\
采用公式： \\
(AB)^T = B^TA^T \\
(A + B)^T = A^TB^T
$$

因此需要找到使得$L(w)$ 最小时对应的参数$w$,  即：
$$
\frac {\part L(w)}{\part w} = 2X^TXw - 2 X^TY \\
所用公式 （常见矩阵求导）:\\
\frac {\part w^TX^TXw}{w} = 2X^TXw\\
\frac {\part w^TX^TY}{\part w} = XY^T \\
\frac {\part \beta w}{\part w} = \beta ^T
$$

即，令$\frac {\part L(w)}{\part w} = 0$, 即 $w = (X^TX)^{-1}XY^T$

(b) 几何解释：

 <img src="https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210316214308.png" style="zoom:50%;" />

两向量垂直可得：$<a, b> = a . b = a^Tb =0$

而平面X的法向量为$Y-Xw$,  可得 $X(Y-Xw) = 0$ 即 $w = (X^TX)^{-1}X^TY$

(c) 概率视角：

**LSE**（Least Square Estimate 最小二乘估计）<==> **MLE** (Maximum Likelihood Estimate  极大似然估计)

 $\varepsilon～N(0, \sigma ^2)$

$y = f(w) + \varepsilon \\ f(w) = w^Tx \\ y = w^X + \varepsilon$

$y|x_iw ～N(w^Tx, \varepsilon^2)$

正态分布 公式： $P(y|x_iw) = \frac 1{\sqrt{2\pi}\sigma} e^{\frac {(y-w^Tx)^2}{2\sigma^2}}$

MLE (极大似然估计):

$$
L(w) = lnP(Y|x_iw) = ln\prod\limits_{i=1}^{N}P(y_i|x_iw)=\sum\limits_{i=1}^NlnP(y_i|x_iw)\\
=\sum\limits_{i=1}^N ln \frac{1}{\sqrt{2\pi}\sigma} + \frac {1}{2\sigma^2}(y_i-w^Tx_i)^2
$$

$$
\hat{w} = argmax_wL(w) \\
 = argmax_w   - \frac 1{2\sigma^2}(y_i-w^Tx_i)^2\\
 = argmin_w (y_i-w^Tx_i)^2
$$



