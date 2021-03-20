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
L(w) = \sum \limits _{i=1}^{N}||w^Tx_i - y_i||^2 = \sum \limits _{i=1}^{N}(w^Tx_i-y_i)^2 =(w^TX^T-Y^T)(w^TX^T-Y^T)^T \\
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
L(w) = lnP(Y|x_iw) = ln\prod \limits _{i=1}^{N}P(y_i|x_iw)=\sum\limits_{i=1}^NlnP(y_i|x_iw)\\
=\sum\limits_{i=1}^N ln \frac{1}{\sqrt{2\pi}\sigma} + \frac {1}{2\sigma^2}(y_i-w^Tx_i)^2
$$

$$
\hat{w} = argmax_wL(w) \\
 = argmax_w   - \frac 1{2\sigma^2}(y_i-w^Tx_i)^2\\
 = argmin_w (y_i-w^Tx_i)^2
$$

### 线性回归的推广

(a) 多项式回归：

将标准的线性回归模型： 
$$
y_i = w_0 + w_1 x_i + \epsilon_i
$$
换成一个多项式函数：
$$
y_i = w_0 + w_1 x_i +  w_2 x_i^2 + ... + w_dx_i^d + \epsilon
$$
对于多项式的阶数d不能取过大，一般不大于3或者4,因为d越大，多项式曲线就越光滑，在X的边界处有异常的波动。

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210319122248.png)`

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210319122321.png)

```python
from sklearn.preprocessing import PolynomialFeatures
X_arr = np.arange(6).reshape(3, 2)
print("原始X为：\n",X_arr)

poly = PolynomialFeatures(2)
print("2次转化X：\n",poly.fit_transform(X_arr))

poly = PolynomialFeatures(interaction_only=True)
print("2次转化X：\n",poly.fit_transform(X_arr))
```



(b) 广义可加模型(GAM):

广义可加模型GAM实际上是线性模型推广至非线性模型的一个框架，在这个框架中，每一个变量都用一个非线性函数来代替，但是模型本身保持整体可加性。

**GAM框架模型:**
$$
y_i = w_0 + \sum_\limits{j=1}^{p}f_j(x_{ij}+\epsilon_i)
$$
优点 and 缺点

* 优点：简单容易操作，能够很自然地推广线性回归模型至非线性模型，使得模型的预测精度有所上升；由于模型本身是可加的，因此GAM还是能像线性回归模型一样把其他因素控制不变的情况下单独对某个变量进行推断，极大地保留了线性回归的易于推断的性质。
* 缺点：GAM模型会经常忽略一些有意义的交互作用，比如某两个特征共同影响因变量，不过GAM还是能像线性回归一样加入交互项$x^{(i)} \times x^{(j)}$ 的形式进行建模；但是GAM模型本质上还是一个可加模型，如果我们能摆脱可加模型形式，可能还会提升模型预测精度。

```python
from pygam import LinearGAM
gam = LinearGAM().fit(boston_data[boston.feature_names], y)
gam.summary()
```

### 回归树

基于树的回归方法主要是依据分层和分割的方式将特征空间划分为一系列简单的区域。对某个给定的待预测的自变量，用他所属区域中训练集的平均数或者众数对其预测。由于划分特征空间的分裂规则可以用树的形式进行概括，因此这类方法称为决策树方法。

决策树由结点(node)和有向边(direcdcted edge)组成。

结点有两种类型：内部结点(internal node)和叶结点(leaf node)。内部结点有一个特征或属性，叶结点表示一个类别或者某个值。

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210319180115.png)

**建立回归树的过程：**

* 将自变量的特征空间(即 $x^{(1)}, x^{(2)}, x^{(3)},...,x^{(p)}$) 的可能取值构成的集合分割成J个互不重叠的区域$R_1, R_2, ..., R_j$ 。

*  对落入区域$R_j$的每个观测值作相同的预测，预测值等于$R_j$上训练集的因变量的简单算术平均。

  * a. 选择最优切分特征j以及该特征上的最优点s:

    遍历特征j以及固定j后遍历切分点s， 选择使得下式最小的(j, s)

    $min_{j,s}[min_{c1} \sum_\limits{x_i\in R_{1}(j,s)} (y_i-c_1)^2 + min_{c2} \sum\limits_{x_i\in R_2(j,s)}(y_i - c_2)^2]$

  * b. 按照(j, s) 分裂特征空间：$R_1(j,s) = \{x|x^{j} \le s \}和R_2(j,s) = \{x|x^{j} > s \},\hat{c}_m = \frac{1}{N_m}\sum\limits_{x \in R_m(j,s)}y_i,\;m=1,2$       

  * c. 继续调用步骤a, b 直到满足停止条件， 就是每个区域的样本数小于等于5.

  * d. 将特征空间划分为J个不同的区域， 生成回归树： $f(x) = \sum\limits_{m=1}^{J}\hat{c}_mI(x \in R_m)$                

  ![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210319181710.png)

回归树与线性模型的比较：

线性回归模型的模型形式与树模型的模型形式有着本质的区别，具体而言，线性回归对模型形式做了如下假定：$f(x) = w_0 + \sum\limits_{j=1}^{p}w_jx^{(j)}$ , 而回归树则是$f(x) = \sum\limits_{m=1}^{J}\hat{c}_mI(x\in R_m)$。

树模型的优缺点：

* 树模型的解释性强，在解释性方面可能比线性回归还要方便。
* 树模型更接近人的决策方式。
* 树模型可以用图来表示，非专业人士也可以轻松解读。
* 树模型可以直接做定性的特征而不需要像线性回归一样哑元化。
* 树模型能很好处理缺失值和异常值，对异常值不敏感，但是这个对线性模型来说却是致命的。
* 树模型的预测准确性一般无法达到其他回归模型的水平，但是改进的方法很多。

```python
from sklearn.tree import DecisionTreeRegressor    
reg_tree = DecisionTreeRegressor(criterion = "mse",min_samples_leaf = 5)
reg_tree.fit(X,y)
reg_tree.score(X,y)
```

>criterion：{“ mse”，“ friedman_mse”，“ mae”}，默认=“ mse”。衡量分割标准的函数 。
>splitter：{“best”, “random”}, default=”best”。分割方式。
>max_depth：树的最大深度。
>min_samples_split：拆分内部节点所需的最少样本数，默认是2。
>min_samples_leaf：在叶节点处需要的最小样本数。默认是1。
>min_weight_fraction_leaf：在所有叶节点处（所有输入样本）的权重总和中的最小加权分数。如果未提供sample_weight，则样本的权重相等。默认是0。
>
>https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html?highlight=tree#sklearn.tree.DecisionTreeRegressor

### 支持向量回归 (SVR)

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210320112604.png)

在线性回归的理论中，每个样本点都要计算平方损失，但是SVR却是不一样的。SVR认为：落在$f(x)$的$\epsilon$邻域空间中的样本点不需要计算损失，这些都是预测正确的，其余的落在$\epsilon$邻域空间以外的样本才需要计算损失，因此：  ![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210320112709.png)
$$
min_{w,b,\xi_i,\hat{\xi}_i} \frac{1}{2}||w||^2 +C \sum\limits_{i=1}^{N}(\xi_i,\hat{\xi}_i)\\
   s.t.\;\;\; f(x_i) - y_i \le \epsilon + \xi_i\\
   \;\;\;\;\;y_i - f(x_i) \le  \epsilon +\hat{\xi}_i\\
   \;\;\;\;\; \xi_i,\hat{\xi}_i \le 0,i = 1,2,...,N
$$
引入拉格朗日函数：                  
$$
   \begin{array}{l}
L(w, b, \alpha, \hat{\alpha}, \xi, \xi, \mu, \hat{\mu}) \\
\quad=\frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N}\left(\xi_{i}+\widehat{\xi}_{i}\right)-\sum_{i=1}^{N} \xi_{i} \mu_{i}-\sum_{i=1}^{N} \widehat{\xi}_{i} \widehat{\mu}_{i} \\
\quad+\sum_{i=1}^{N} \alpha_{i}\left(f\left(x_{i}\right)-y_{i}-\epsilon-\xi_{i}\right)+\sum_{i=1}^{N} \widehat{\alpha}_{i}\left(y_{i}-f\left(x_{i}\right)-\epsilon-\widehat{\xi}_{i}\right)
\end{array}
$$
   再令$L(w, b, \alpha, \hat{\alpha}, \xi, \xi, \mu, \hat{\mu})$对$w,b,\xi,\hat{\xi}$求偏导等于0，得： $w=\sum_{i=1}^{N}\left(\widehat{\alpha}_{i}-\alpha_{i}\right) x_{i}$。                             
   上述过程中需满足KKT条件，即要求：                 
$$
   \left\{\begin{array}{c}
\alpha_{i}\left(f\left(x_{i}\right)-y_{i}-\epsilon-\xi_{i}\right)=0 \\
\hat{\alpha_{i}}\left(y_{i}-f\left(x_{i}\right)-\epsilon-\hat{\xi}_{i}\right)=0 \\
\alpha_{i} \widehat{\alpha}_{i}=0, \xi_{i} \hat{\xi}_{i}=0 \\
\left(C-\alpha_{i}\right) \xi_{i}=0,\left(C-\widehat{\alpha}_{i}\right) \hat{\xi}_{i}=0
\end{array}\right.
$$
   SVR的解形如:$f(x)=\sum_{i=1}^{N}\left(\widehat{\alpha}_{i}-\alpha_{i}\right) x_{i}^{T} x+b$                 

