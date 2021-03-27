---
author: "xingyu"
author_link: ""
title: "集成学习_Task05"
date: 2021-03-26T17:40:02+08:00
lastmod: 2021-03-26T17:40:02+08:00
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

Task05  使用sklearn构建完整的分类项目

<!--more-->

##　收集数据 选择特征

数据集采用： IRIS鸢尾花数据集

```python
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target
feature = iris.feature_names
data = pd.DataFrame(X,columns=feature)
data['target'] = y
data.head()
```

>各个特征的相关解释：
>
>- sepal length (cm)：花萼长度(厘米)
>- sepal width (cm)：花萼宽度(厘米)
>- petal length (cm)：花瓣长度(厘米)
>- petal width (cm)：花瓣宽度(厘米)

## 度量模型性能指标：

度量分类模型的指标和回归的指标有很大的差异，首先是因为分类问题本身的因变量是离散变量，

因此像定义回归的指标那样，单单衡量预测值和因变量的相似度可能行不通。其次，在分类人物中，我们对于每个类别犯错的代价不尽相同，例如：我们将癌症患者错误预测为无癌症和无癌症错误预测为癌症患者，在医院和个人的代价都是不同的，前者会使患者无法得到及时的救治而耽误了最佳治疗时间甚至付出生命的的代价，而后者只需要在后续的治疗过程中继续取证就好了，因此我们不希望出现前者，当我们发生了前者这样的错误的时候会认为建立的模型是很差的。为了结解决这些问题，我们必须将各种情况分开讨论，然后给出评价指标。

* 真阳性TP：预测值和真实值都为正例。
* 真阴性TN：预测值和真实值都为负例。
* 假阳性FP：预测值为正，实际值为负。
* 假阴性FN：预测值为负，实际值为正。

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210326194941.png)

分类模型的指标：                    
   - 准确率：分类正确的样本数占总样本的比例，即：$ACC = \frac{TP+TN}{FP+FN+TP+TN}$.                                
   - 精度：预测为正且分类正确的样本占预测值为正的比例，即：$PRE = \frac{TP}{TP+FP}$.                     
   - 召回率：预测为正且分类正确的样本占类别为正的比例，即：$REC =  \frac{TP}{TP+FN}$.                     
   - F1值：综合衡量精度和召回率，即：$F1 = 2\frac{PRE\times REC}{PRE + REC}$.                                     
   - ROC曲线：以假阳率为横轴，真阳率为纵轴画出来的曲线，曲线下方面积越大越好。                                                          
https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics                           

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210326195052.png)



