---
author: "xingyu"
author_link: ""
title: "IntegratedLearning_Task04"
date: 2021-03-24T19:00:45+08:00
lastmod: 2021-03-24T19:00:45+08:00
draft: true
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

Task 05

对模型超参数进行调优（调参）

<!--more-->

在之前的讨论中，对模型的优化都是对模型算法本身的改进。比如：岭回归对线性回归的优化在于线性回归的损失函数中加入正则化项从而牺牲无偏性降低方差。但是，在L2正则化中参数$\lambda $应该是多少？0.01、0.1、还是1？ 到目前为止，我们只能凭借经验或者瞎猜。事实上，找到最佳参数的问题本质上属于最优化的内容，因为从一个参数集合中找熬最佳的值的本身就是最优化的任务之一，我们脑海中浮现出来的算法无非就是：梯度下降法、牛顿法等无约数优化算法或者约束优化算法，但是在具体验证这个想法是否可行之前，我们必须先认识两个最本质概念的区别。

* 参数与超参数

  以岭回归中的参数$\lambda$和参数$w$为例子。参数$w$是我们通过设定一个具体的$\lambda$后使用类似于最小二乘法、梯度下降法等方式优化出来的，我们总是设定了$\lambda$是多少后才优化出来的参数$w$。因此，类似于参数$w$，使用最小二乘法或者梯度下降法等最优化算法优化出来的数我们称为参数，类似与$\lambda$一样，我们无法使用最小二乘法或者梯度下降法算法优化出来的数我们称为超参数。

  模型参数是模型内部的配置变量，其值可以根据数据进行估计。

  * 进行预测时需要参数。
  * 参数是从数据估计获悉的。
  * 参数通常不由编程者手动设置。
  * 参数通常被保存为学习算法模型的一部分。
  * 参数是机器学习算法的关键，他们通常由过去的训练数据中总结得出。模型超参数是模型外部的额外配置，其值无法从数据中估计。
  * 超参数通常用于帮助估计模型参数
  * 超参数通常由人工指定。
  * 超参数可以使用启发式设置。
  * 超参数经常被调整为给定的预测建模问题。

* 网格搜索GridSearchCV()

  网格搜索：https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html?highlight=gridsearchcv#sklearn.model_selection.GridSearchCV
  网格搜索结合管道：https://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html?highlight=gridsearchcv

  网格搜索的思想非常简单，比如有两个超参数要去选择，就把所有超参数选择列出来分别做排列组合。然后针对每组超参数分别建立一个模型，然后选择误差最小的那组超参数。

* 随机搜索RandomizedSearchCV():

  https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html?highlight=randomizedsearchcv#sklearn.model_selection.RandomizedSearchCV
  网格搜索相当于暴力地从参数空间中每个都尝试一遍，然后选择最优的那组参数，这样的方法显然是不够高效的，因为随着参数类别个数的增加，需要尝试的次数呈指数级增长。有没有一种更加高效的调优方式呢？那就是使用随机搜索的方式，这种方式不仅仅高校，而且实验证明，随机搜索法结果比稀疏化网格法稍好(有时候也会极差，需要权衡)。参数的随机搜索中的每个参数都是从可能的参数值的分布中采样的。与网格搜索相比，这有两个主要优点：

  - 可以独立于参数数量和可能的值来选择计算成本。
  - 添加不影响性能的参数不会降低效率。

## 案例分析

```python
# 我们先来对未调参的SVR进行评价： 
from sklearn.svm import SVR     # 引入SVR类
from sklearn.pipeline import make_pipeline   # 引入管道简化学习流程
from sklearn.preprocessing import StandardScaler # 由于SVR基于距离计算，引入对数据进行标准化的类
from sklearn.model_selection import GridSearchCV  # 引入网格搜索调优
from sklearn.model_selection import cross_val_score # 引入K折交叉验证
from sklearn import datasets


boston = datasets.load_boston()     # 返回一个类似于字典的类
X = boston.data
y = boston.target
features = boston.feature_names
pipe_SVR = make_pipeline(StandardScaler(),
                                                         SVR())
score1 = cross_val_score(estimator=pipe_SVR,
                                                     X = X,
                                                     y = y,
                                                     scoring = 'r2',
                                                      cv = 10)       # 10折交叉验证
print("CV accuracy: %.3f +/- %.3f" % ((np.mean(score1)),np.std(score1)))

# CV accuracy: 0.187 +/- 0.649
```



```python
# 下面我们使用网格搜索来对SVR调参：
from sklearn.pipeline import Pipeline
pipe_svr = Pipeline([("StandardScaler",StandardScaler()),
                                                         ("svr",SVR())])
param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid = [{"svr__C":param_range,"svr__kernel":["linear"]},  # 注意__是指两个下划线，一个下划线会报错的
                            {"svr__C":param_range,"svr__gamma":param_range,"svr__kernel":["rbf"]}]
gs = GridSearchCV(estimator=pipe_svr,
                                                     param_grid = param_grid,
                                                     scoring = 'r2',
                                                      cv = 10)       # 10折交叉验证
gs = gs.fit(X,y)
print("网格搜索最优得分：",gs.best_score_)
print("网格搜索最优参数组合：\n",gs.best_params_)

# 网格搜索最优得分： 0.6081303070817233
# 网格搜索最优参数组合：
# {'svr__C': 1000.0, 'svr__gamma': 0.001, 'svr__kernel': 'rbf'}
```

```python
# 下面我们使用随机搜索来对SVR调参：
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform  # 引入均匀分布设置参数
pipe_svr = Pipeline([("StandardScaler",StandardScaler()),
                                                         ("svr",SVR())])
distributions = dict(svr__C=uniform(loc=1.0, scale=4),    # 构建连续参数的分布
                     svr__kernel=["linear","rbf"],                                   # 离散参数的集合
                    svr__gamma=uniform(loc=0, scale=4))

rs = RandomizedSearchCV(estimator=pipe_svr,
                                                     param_distributions = distributions,
                                                     scoring = 'r2',
                                                      cv = 10)       # 10折交叉验证
rs = rs.fit(X,y)
print("随机搜索最优得分：",rs.best_score_)
print("随机搜索最优参数组合：\n",rs.best_params_)

# 随机搜索最优得分： 0.30021249798866756
# 随机搜索最优参数组合：
# {'svr__C': 1.4195029566223933, 'svr__gamma': 1.8683733769303625, 'svr__kernel': 'linear'}
```

