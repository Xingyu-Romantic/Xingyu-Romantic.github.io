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

