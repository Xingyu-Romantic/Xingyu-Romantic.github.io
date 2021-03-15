---
author: "xingyu"
author_link: ""
title: "集成学习_Task01"
date: 2021-03-15T19:11:29+08:00
lastmod: 2021-03-15T19:11:29+08:00
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

Datawhale 集成学习 Task01 笔记

本次学习内容：

* 了解sklearn中读取数据、生成数据。 
* 并简单介绍回归与分类。
* 以下案例内容采用Datawhale资料，所有函数并不在此过多赘述，详情查看下方API。

<!--more-->

sklearn 中 所有内置数据集封装在datasets对象中，返回的对象中：

* data: 特征X的矩阵(ndarray)
* target: 因变量的向量 (ndarray)
* feature_names: 特征名称 (ndarray)

```
from sklearn import boston, iris
boston = datasets.load_boston()  #boston 房价数据集
iris = datasets.load_iris()  # iris 数据集
```

关于 seaborn、matplotlib绘图详见 [动手数据分析](https://www.involute.top/2020/08/hands-on-data-analysis/)

关于使用sklearn生成数据集，详见 本节 [**1.3 无监督学习**](#13-无监督学习) 内容

## 1.1 回归

boston 房价数据集加载 + 可视化

```python
from sklearn import datasets
boston = datasets.load_boston()     # 返回一个类似于字典的类
X = boston.data
y = boston.target
features = boston.feature_names
boston_data = pd.DataFrame(X,columns=features)
boston_data["Price"] = y
boston_data.head()

sns.scatterplot(boston_data['NOX'],boston_data['Price'],color="r",alpha=0.6)
plt.title("Price~NOX")
plt.show()
```

## 1.2 分类

iris 数据集 加载 + 可视化

```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
features = iris.feature_names
iris_data = pd.DataFrame(X,columns=features)
iris_data['target'] = y
iris_data.head()

# 可视化特征
marker = ['s','x','o']
for index,c in enumerate(np.unique(y)):
    plt.scatter(x=iris_data.loc[y==c,"sepal length (cm)"],y=iris_data.loc[y==c,"sepal width (cm)"],alpha=0.8,label=c,marker=marker[c])
plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.legend()
plt.show()
```

## 1.3 无监督学习

sklearn 官方 API  [此处](https://scikit-learn.org/stable/modules/classes.html?highlight=datasets#module-sklearn.datasets)

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210315192647.png)

**生成月牙型非凸集

```python
from sklearn import datasets
x, y = datasets.make_moons(n_samples=2000, shuffle=True,
                  noise=0.05, random_state=None)
for index,c in enumerate(np.unique(y)):
    plt.scatter(x[y==c,0],x[y==c,1],s=7)
plt.show()
```

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210315192733.png)

**生成符合正态分布的聚类数据**

```python
from sklearn import datasets
x, y = datasets.make_blobs(n_samples=5000, n_features=2, centers=3)
for index,c in enumerate(np.unique(y)):
    plt.scatter(x[y==c, 0], x[y==c, 1],s=7)
plt.show()
```

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210315192850.png)





