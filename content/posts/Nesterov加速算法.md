---
author: "xingyu"
author_link: ""
title: "最优化 Nesterov加速算法"
date: 2021-05-18T13:45:17+08:00
lastmod: 2021-05-18T13:45:17+08:00
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


# 8.2 最优化 Nesterov加速算法


## 理论介绍

### Lipschitz 连续

Lipschitz 连续： 在一个连续函数$f$上面额外施加了一个限制，要求存在一个常数$K \geq 0$使得定义域内的任意两个元素$x_1$ 和 $x_2$ 都满足
$$
|f(x_1) - f(x_2)| \leq K |x_1 - x_2|
$$
此时称函数 $f$的Lipschitz常数为 **K**。

简单理解，就是 $f'(x) \leq K, \forall x \in R$ ，$R$为$f$的定义域

### 梯度下降法：

考虑以下线性转化问题： 
$$
b = Ax + w
$$

例如在图像模糊问题中， A为模糊模板（由未模糊图像通过转换而来）， b为模糊图像， w为噪声。 并且， A 和 b已知， x为待求的系数。

求解该问题的传统方法为最小二乘法，思想很简单，使得重构误差$|| Ax - b||^2$最小，即：
$$
\hat{x} = arg\min_x ||Ax- b||^2
$$
对 $f(x) = ||Ax - b||^2$求导，可得其导数为：$f'(x) = 2A^T(Ax-b)$。 对于该问题，令导数为零即可以取得最小值（函数$f(x)$为凸函数，其极小值为最小值）。

1. 如果A为非奇异矩阵，即A可逆的话，那么可得该问题的精确解为$x = A^{-1}b$。
2. 如果A为奇异矩阵，即A不可逆，则该问题没有精确解。退而求其次，我们求一个近似解就好， $||Ax-b||^2 \leq\epsilon$。

$$
min||x||_1  \\
s.t. ||Ax-b||^2 \leq \epsilon
$$

其中， $||x||_1$为惩罚项，用以规范化参数$x$。该例子使用L1范数作为惩罚项，是希望$x$尽量稀疏（非零元素个数尽可能的少）,即b是A的一个稀疏表示。$||Ax-b||^2 \leq \epsilon$则为约束条件，即重构误差最小。问题(3)也可以描述为：
$$
\min_x{F(x) \equiv ||Ax - b||^2 + \lambda||x||_1}
$$
式子(4)即为一般稀疏表示的优化问题。希望重构误差尽可能的小，同时参数的个数尽可能的少。

注： 惩罚项也可以是L2或者其他范数。

####  梯度下降法的缺陷

考虑更为一般的情况，我们来讨论梯度下降法。有无约束的优化问题如下：
$$
\min_x {F(x) \equiv f(x)}
$$
梯度下降法基于这样的观察：如果实值函数$F(x)$在点$a$处可微且有定义，那么函数$F(x)$在点$a$沿着梯度想反的方向$-\triangledown F(a)$下降最快。

基于此，我们假设$f(x)$连续可微。如果存在一个足够小的数值$t>0$使得$x_2 = x_1 - t\triangledown F(a)$，那么：
$$
F(x_1) \geq F(x_2) \\
x_0 \in R^n, x_k = x_{k-1} - t_k \triangledown f(x_{k-1})
$$
梯度下降法的核心就是通过式子(6)找到序列${x_k}$，使得$F(x_k) \geq F(x_{k-1})$。

![](E:\项目\8.2 最优化\梯度下降法.png)

从上图可以看出：初始点不同，获得的最小值也不同。因为梯度下降法求解的是局部最小值，受初值的影响较大。如果函数$f(x)$为凸函数的话，则局部最小值亦为全局最小值。这时，初始点只对迭代速度有影响。

再回头看一下式子(6)，我们使用步长$t_k$和倒数$\triangledown F(x_k)$来控制每一次迭代时$x$的变化量。再看一下上面的那张图，对于每一次迭代，我们当然希望$F(x)$的值降得越快越好，这样我们就能更快速的获得函数的最小值。因此，步长$t_k$的选择很重要。

如果步长$t_k$太小，则找到最小值的迭代次数非常多，即迭代速度非常慢，或者说收敛速度很慢；而步长太大的话，则会出现overshoot the minimum的现象，即不断在最小值左右徘徊，跳来跳去的。

然而，$t_k$最后还是作用在$x_{k-1}$上，得到$x_k$。因此，更为朴素的思想应该是：序列${x_k}$的个数尽可能小，即每一次迭代步伐尽可能大，函数值减少得尽可能的多。那么就是关于序列${x_k}$的选择了，如何更好的选择每一个点$x_k$，使得函数值更快的趋紧其最小值。

### ISTA 算法

ISTA（Iterative shrinkage-thresholding algorithm）， 即 迭代阈值收缩算法。

先从无约束优化问题开始， 即上面的式子(5)：

这时候，我们还假设$f(x)$满足Lipschitz连续条件，即$f(x)$的导数有下界，其最小下界称为Lipshitz常数$L(f)$。这时，对于任意的$L \geq L(f)$， 有：
$$
f(x) \leq f(y) + <x - y, \triangledown f(y)> + \frac L2||x-y||^2 \space \space \forall x,y \in R^n
$$
基于此，在点$x_k$附近可以把函数值近似为：
$$
\hat{f}(x, x_k) = f(x_k) + <\triangledown f(x_k), x-x_k> + \frac L2 ||x-x_k||^2
$$
在梯度下降的每一步迭代中，将点$x_{k-1}$处的近似函数取得最小值的点作为下一次迭代的起始点$x_k$，这就是所谓的[Proximal Gradient Descent for L1 Regularization](https://www.cnblogs.com/breezedeus/p/3426757.html)算法（其中，$t_k = \frac 1L$）。
$$
x_k = arg\min_x\{f(x_{k-1} + <x - x_{k-1}, \triangledown f(x_{k-1})> + \frac 1{2t_k}||x-x_{k-1}||^2)\}
$$
上面的方法只适合解决非约束问题。而ISTA要解决的是带惩罚项的优化问题，引入范数规范函数$g(x)$对参数$x$进行约束，如下：
$$
min\{F(x)\equiv f(x) + g(x): x\in R^n\}
$$
使用更为一般的二次近似模型来求解上述的优化问题，在点$y, F(x):=f(x) + g(x)$的二次近似函数为：
$$
Q_L(x, y) := f(y) + <x-y, \triangledown f(y)> + \frac L2||x-y||^2 + g(x)
$$
该函数的最小值表示为 ：            $P_L$是proximal（近端算子的简写形式）、
$$
P_L(y) := arg\min_x\{Q_L(x,y): x\in R^n\}
$$
忽略其常数项$f(y)$和$\triangledown F(y)$，这些有和没有对结果没有影响。再结合式子(11)和(12), $P_L(y)$可以写成：
$$
P_L(y) = arg\min_x\{g(x) + \frac L2||x-(y-\frac 1L \triangledown f(y))||^2\}
$$
显然，使用ISTA解决带约束的优化问题时的基本迭代步骤为：
$$
x_k = P_L(x_k - 1)
$$
固定步长的ISTA的基本迭代步骤如下（步长 $t = 1 / L(f)$）:

>ISTA with constant stepsize
>
>**Input:** $L := L(f) - A $ Lipschitz constant of $\triangledown f$.
>
>**Step 0:**  Take $x_0 \in R^n$
>
>Step k: (k >= 1) Compute
>$$
>x_k = P_L(x_{k-1})
>$$

然而， 固定步长的ISTA的缺点是： Lipschitz常数$L(f)$不一定可知或者计算。例如， L1范数约束的优化问题，其Lipschitz常数依赖于$A^TA$的最大特征值。而对于大规模的问题，非常难计算。因此，使用以下带回溯（backtracking）的ISTA：

>ISTA with backtracking
>
>**Step 0.**  Take $L_0 > 0$，some $\eta > 1$, and $x_0 \in R^n$
>
>**Step k.** $(k \geq 1)$ Find the smallest nonnegative intergers $i_k$ such that with $\bar{L} = \eta^{i_k} L_{k-1}$
>$$
>F(P_{\bar{L}}(x_{k-1})) \leq Q_{\bar{L}}(P_{\bar{L}}(x_{k-1}, x_{k-1}))
>$$
>Set $L_k = \eta ^ {i_k}L_{k-1}$  and compute
>$$
>x_k = P_{L_k}(x_{k-1})
>$$

### FISTA （A fast iterative shrinkage-thresholding algorithm）是一种快速的迭代阈值收缩算法（ISTA）。

FISTA 与 ISTA的区别在于迭代步骤中近似函数起始点y的选择。ISTA使用前一次迭代求得的近似函数最小值$x_{k-1}$，而FISTA则使用另一种方法来计算y的位置。

>FISTA with constant stepsize
>
>**Input: ** $L = L(f) - A$   Lipschitz constant of $\triangledown f$.
>
>**Step 0.**  Take $y_1 = x_0 \in R^n$， $t_1 = 1$.
>
>**Step k.**  $(k \geq 1)$ Compute
>$$
>x_k = P_L(y_k) \\
>t_{k+1} = \frac {1 + \sqrt{1+4t^2_k}}{2}, \\
>y_{k+1} = x_k + \frac {t_k - 1}{t_{k+!}}(x_k - x_{k-1})
>$$

当然，考虑到与ISTA同样的问题：问题规模大的时候，决定步长的Lipschitz常数计算复杂。FISTA与ISTA一样，亦有其回溯算法。在这个问题上，FISTA与ISTA并没有区别，上面也说了，FISTA与ISTA的区别仅仅在于每一步迭代时近似函数起始点的选择。更加简明的说：FISTA用一种更为聪明的办法选择序列$\{x_k\}$, 使得其基于梯度下降思想的迭代过程更加快速地趋近问题函数$F(x)$的最小值。

### 第二类Nesterov加速算法

对于 LASSO 问题：
$$
\min_x{ \frac 1 2||Ax - b||^2 + \mu||x||_1}
$$
利用第二类 Nesterov 加速的近似点梯度法进行优化。

该算法被外层连续化策略调用，在连续化策略下完成某一固定正则化系数的内层迭代优化。第二类 Nesterov 加速算法的迭代格式如下:
$$
z^{k} = (1-\gamma_k)x^{k-1} +\gamma_ky^{k-1}, \\
y^{k} = prox_{\frac{t_k}{\gamma_k}h}(y^{k-1}-\frac{t_k}{\gamma_k}A^T(Az^k-b)), \\
x^{k} = (1-\gamma_k)x^{k-1}+\gamma_ky^k.
$$
和经典FISTA算法的一个重要区别在于，第二类Nesterov加速算法中的三个序列{$x^k$}，{y^k}和{z^k}都可以保证在定义域内.而FISTA算法中的序列{$y^k$}不一定在定义域内.
$$
y^{k} = prox_{\frac{t_k}{\gamma_k}h}(y^{k-1}-\frac{t_k}{\gamma_k}A^T(Az^k-b))
$$
![image-20210517192923458](C:\Users\Maple\AppData\Roaming\Typora\typora-user-images\image-20210517192923458.png)

### 第三类Nesterov加速算法

同样的，对于LASSO问题：
$$
\min_x{ \frac 1 2||Ax - b||^2 + \mu||x||_1}
$$
利用第三类 Nesterov 加速的近似点梯度法进行优化。其迭代格式如下:
$$
z^{k} = (1-\gamma_k)x^{k-1} +\gamma_ky^{k-1}, \\
y^{k} = prox_{(t_k\sum_{i=1}^{k}{\frac{1}{\gamma_i}})h}(-t_{k}\sum_{i=1}^{k}{\frac{1}{\gamma_i}}\nabla{f(z^i)}, \\
x^{k} = (1-\gamma_k)x^{k-1}+\gamma_ky^k.
$$
该算法和第二类Nesterov加速算法的区别仅仅在于$y^k$的更新，第三类Nesterow加速算法计算$y^k$时需要利用全部已有的$\nabla{f(z^i)},i=1,2,\cdots,k$.

## 比较



![image-20210517184956548](C:\Users\Maple\AppData\Roaming\Typora\typora-user-images\image-20210517184956548.png)

可以看到：就固定步长而言，FISTA算法相较于第二类Nesterov加速算法收敛得略快一些，也可以注意到FISTA算法是非单调算法.同时，BB步长和线搜索技巧可以加速算法的收敛速度.此外，带线搜索的近似点梯度法可以比带线搜索的FISTA算法更加收敛.

## 应用













## 杂项

梯度下降法中面临的挑战问题：

1. 病态条件： 不同方向有不同的梯度， 学习率选择困难  （悬崖，跑到对面悬崖）
2. 局部最小： 陷入局部最小点
3. 鞍点： 
   1. 梯度为0， Hessian矩阵同时存在正值和负值，
   2. Hessian矩阵的所有特征值为正值的概率很低
   3. 对于高维情况，鞍点和局部最小点的数量很多， 使用二阶优化算法会有问题。
4. 平台区域， 梯度为0， Hessian矩阵也为0，  **加入噪音使得从平台区域跳出**

动量法：

* 主要想法：在参数更新时考虑历史梯度信息   保持惯性

* 参数更新规则
  $$
  v_t \leftarrow \rho v_{t-1} - \eta \triangledown J(\theta_t) \\
  \theta_{t+1} \leftarrow \theta_{t} +v_t
  $$
  参数 $\rho \in [0, 1)$为历史梯度贡献的衰减速率，一般为 0.5、0.9或0.99

  $\eta$ : 学习率

  $\rho$也可以随着迭代次数的增大而变大，随着时间推移调整$\rho$比收缩$\eta$更重要 ， 动量法克制了SGD中的两个问题

  * Hessian矩阵的病态问题（右图图解）
  * 随机梯度的方差带来的不稳定

Nesterov 动量法：

* 受 Nesterov 加速梯度算法 NGA 的启发

* 梯度计算在施加当前速度之后

* 在动量法的基础上添加了一个校正因子(correction factor)
  $$
  v_t \leftarrow \rho v_{t-1} - \eta \triangledown J(\theta_t + \rho v_{t-1}) \\
  \theta_{t+1} \leftarrow \theta_{t} +v_t
  $$

AdaGrad

* 学习率自适应：与梯度历史平方值的综合的平方根成反比

* 效果： 更为平缓的倾斜方向上回去的更大的进步，可能逃离鞍点

* 问题：理解题都平方和增长过快，导致学习率较小，提前终止学习。
  $$
  s_t \leftarrow s_{t-1}  + \triangledown J(\theta_t) * \triangledown J(\theta_t) \\
  \theta_{t+1} \leftarrow \theta_t - \frac \eta{\sqrt{s_t}}\triangledown J(\theta_t)
  $$

RMSProp

* 在AdaGrad基础上，降低了对早期历史梯度的依赖

* 通过设置衰减系数$\beta_2$实现

* 建议 $\beta_2$ = 0.9
  $$
  s_t \leftarrow \beta_2 s_{t-1} + (1 - \beta_2) \\
  \theta_{t+1} \leftarrow \theta_t - \frac {\eta}{\sqrt{s_t}} \triangledown J(\theta_t)
  $$

Adam

* 同时考虑动量和学习率自适应

* $\beta_1$通常设置成0.9， $\beta_2$设置成0.999
  $$
  v_t \leftarrow \beta_1 v_{t-1} - (1 - \beta_1)\triangledown J(\theta_t) \\
  s_t \leftarrow \beta_2 s_{t-1} - (1 - \beta_2) \triangledown J(\theta_t) * \triangledown J(\theta_t) \\
  \theta_{t+1} \leftarrow \theta_t - \eta \frac {v_t}{\sqrt{s_t}}
  $$



# 参考文献

* [FISTA的由来：从梯度下降法到ISTA & FISTA_huang1024rui的专栏-CSDN博客_fista](https://blog.csdn.net/huang1024rui/article/details/51534524)
* [2] Beck A , Teboulle M . A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems[J]. Siam J Imaging Sciences, 2009, 2(1):183-202.