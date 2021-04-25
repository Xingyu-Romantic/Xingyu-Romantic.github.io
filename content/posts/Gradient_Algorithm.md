---
author: "xingyu"
author_link: ""
title: "Gradient_Algorithm 梯度类算法"
date: 2021-04-24T21:02:53+08:00
lastmod: 2021-04-24T21:02:53+08:00
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

《最优化理论》 课程作业， 梯度类算法 解决 LASSO问题

<!--more-->

## LASSO问题

对于 LASSO问题
$$
\displaystyle\min_x \frac{1}{2}\|Ax-b\|_2^2 + \mu \|x\|_1.
$$
此问题为回归问题，而对于回归问题其实本质就是一个函数拟合的过程，模型不能太过复杂，否则很容易发生过拟合现象，所以我们就要加入正则化项，而对于LASSO问题，采用L1正则化，会使得部分学习到的特征权值为0, 从而达到稀疏化和特征选择的目的。

**为什么Ｌ１正则更容易导致稀疏解？**

假设只有一个参数$w$, 损失函数为$L(w)$, 加上Ｌ１正则项后有：
$$
J_{L1}(w) =  L(w) + \lambda|w|_1
$$
假设$L(w)$在０处的导数为$d_0$, 即
$$
\frac {\partial L(w)}{\partial w} = d_0
$$
则可以推导使用Ｌ１正则的导数
$$
\frac {\partial J_{L1}(w)}{\partial w}|_{w=0^-} = d_0 - \lambda \\
\frac {\partial J_{L1}(w)}{\partial w}|_{w=0^+} = d_0 + \lambda
$$
引入Ｌ１正则后，代价函数在0处的导数有一个突变，从$d_0 + \lambda$ 到　$d_0 - \lambda$, 若　$d_0 + \lambda$ 和$d_0 - \lambda$异号，则在0处会是一个极小值点。因此，优化 时，可能优化到该极小值点上，即$w=0$处。

## LASSO 问题的梯度下降

### LASSO问题的连续化策略

目的：寻找一个合适的$\mu_t$, 求解相应的LASSO问题。

方法：从较大的$\mu_t$ 逐渐减小到 $\mu_0$。

**代码解析：**

1. 更新梯度阈值，他们随着外层迭代的进行逐渐减小，对子问题求解的精度逐渐提高。
2. 当内层循环达到收敛条件退出时，缩减正则化系数$\mu_t$，并判断收敛。
3. 外层循环收敛的条件：当$\mu$ 已经减小到与$\mu_0$相同并且函数值或梯度值满足收敛条件。

### BB步长梯度下降法

对于可微的目标函数$f(x)$，梯度下降法通过使用如下重复迭代格式
$$
x^{k+1} = x^k - \alpha\bigtriangledown f(x^k)
$$
求解$f(x)$的最小值，其中$a_k$为第k步的步长。

令 $s^k=x^{k+1}-x^{k}$, $y^k=\nabla f(x^{k+1})-\nabla f(x^k)$ , 定义两种BB步长，$\displaystyle\frac{(s^k)^\top s^k}{(s^k)^\top y^k}$ 和 $\displaystyle\frac{(s^k)^\top y^k}{(y^k)^\top y^k}$。

理论解释：

如果我们记 $g^{k} = \bigtriangledown f(x^{(k)})$  和 $F^{k} = \bigtriangledown ^2f(x^{(k)})$， 那么**一阶方法**就是 $x^{k+1} = x^k - \alpha_kg(x^{(k)})$，其中步长$\alpha_k$是固定的，也可以使线搜索获得的，一阶方法简单但是收敛速度慢，**牛顿方法**就是$x^{(k+1)} = x^{(k)} - (F^{(k)})^{-1} g^{(k)}$ ，其收敛速度更快，但是海森矩阵计算代价较大，而**BB方法**就是用$\alpha_kg^{(k)}$来近似$(F^{(k)})^{-1}g^{(k)}$。

定义 $s^k=x^{k+1}-x^{k}$ 和 $y^{(k-1)} = g^{(k)} - g^{(k-1)}$， 那么海森矩阵实际上就是
$$
F^{(k)}s^{(k-1)} = y^{(k-1)}
$$
用 $(a_kI)^{-1}$ 来近似 $F^{(k)}$， 那么就应该有
$$
(\alpha_kI)^{-1}s^{(k-1)} = y^{(k-1)}
$$
利用最小二乘法：
$$
\alpha_k^{-1} = \mathop{\arg\min}\limits_{\beta} \frac 12 ||s^{(k-1)}\beta - y ^{(k-1)}|| ^2 => \alpha_k^{1} = \displaystyle\frac{(s^{k-1})^\top s^{k-1}}{(s^{k-1})^\top y^{k-1}} \\
\alpha_k = \mathop{\arg\min}\limits_{\alpha} \frac 12 ||s^{(k-1)}\beta - y ^{(k-1)}\alpha|| ^2 => \alpha_k^{2} = \displaystyle\frac{(s^{k-1})^\top y^{k-1}}{(y^{k-1})^\top y^{k-1}}
$$
**BB方法特点：**

1. 几乎不需要额外的计算，但是往往会带来极大的性能收益
2. 实际应用中两个表达式都可以用，甚至可以交换使用，但是优劣需结合具体问题
3. 收敛性很难证明。

退出搜索条件：$f(x^k+\tau d^k)\le C_k+\rho\tau (g^k)^\top d^k$  或 进行超过10次步长衰减后退出搜索。

`FDiff`：表示函数值的相对变化

`|XDiff|`：表示$x$与上一步迭代$xp$之前的相对变化。

### Huber 光滑梯度法

将LASSO问题转化为光滑函数，
$$
\displaystyle\ell_\sigma(x)=\left\{
\begin{array}{ll}
\frac{1}{2\sigma}x^2, & |x|<\sigma; \\
|x|-\frac{\sigma}{2}, & \mathrm{otherwise}.
\end{array} \right.
$$
使用 $\displaystyle L_\sigma(x)=\sum_{i=1}^n\ell_\sigma(x_i)$ 替代 $||x||_1$ , 得到如下优化问题：
$$
\displaystyle\min_x f(x) := \frac{1}{2}\|Ax-b\|_2^2 + \mu L_{\sigma}(x).
$$
在 $x$ 点处$f$的梯度为：
$$
\displaystyle\nabla f(x)=A^\top (Ax-b)+\mu\nabla L_{\sigma}(x),
$$
其中
$$
\displaystyle(\nabla L_{\sigma}(x))_i=\left\{ \begin{array}{ll}
\mathrm{sign}(x_i), & |x_i|>\sigma; \\
\frac{x_i}{\sigma}, & |x_i|\le\sigma.
\end{array} \right.
$$
**代码解析:**

1. 针对光滑化之后的函数进行 梯度下降。
2. 内层循环的收敛条件：当当前梯度小于阈值或者目标函数比那化小于阈值，内层迭代终止
3. 采用线搜索循环选择合适步长并更新$x$。在步长不符合线搜索条件的情况下，对当前步长以$\eta$进行衰减，线搜索次数加1。

### 线搜索方法(LineSearch)

线搜索的迭代过程是 $x_{k+1} = x_k + \alpha_kp_k$, 其中$\alpha_k$和$p_k$分别表示搜索步长和搜索方向。

$p_k$是一个下降方向，满足$\bigtriangledown f_kp_k \le  0$ ，则 $p_k = - B_k^{-1}\bigtriangledown f_k$, B为对称非奇异矩阵，根据$B_k$的选择会产生以下几个方向。

1. $B_k = I$ 时，搜索方向为负梯度方向，该方法为最速下降方向。
2. $B_k = \bigtriangledown ^2 f_k$时， 该方法为牛顿方法。
3. $B_k$ 需要满足对称正定矩阵，该方法为拟牛顿方法。

## 总结

对Matlab的代码采用Python重构，对算法的流程有了比较深入的了解，但是对示例代码进行运行时，生成的迭代次数-函数值图像，相比与网站中给出的同等Matlab生成图像更快的收敛。

## 附录

###  Huber 光滑化梯度法

附上本人Github地址：https://github.com/Xingyu-Romantic/Machine-learning

```python
import numpy as np
opts = {'maxit': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'alpha0': 0.01, 
           'sigma': 0.1, 'verbose': 0}
def LASSO_grad_huber_inn(x, A, b, mu, mu0, opt):
    for i in opts.keys():
        if opt.get(i, -1) == -1:
            opt[i] = opts[i]
    tic = time.time()
    r = np.matmul(A, x) - b
    g = np.matmul(A.T, r)
    huber_g = np.sign(x)
    idx = abs(x) < opt['sigma']
    huber_g[idx] = x[idx] / opt['sigma']
    
    g = g + mu * huber_g
    nrmG= np.linalg.norm(g, 2)
    
    f = 0.5 * np.linalg.norm(r, 2) ** 2 + \
                mu * (np.sum(np.square(x[idx])/(2 * opt['sigma'])) \
                      + np.sum(np.abs(x[abs(x) >= opt['sigma']]) - \
                               opt['sigma'] / 2))
    out = {}
    
    out['fvec'] = 0.5 * np.linalg.norm(r, 2) ** 2 + mu0 * np.linalg.norm(x, 1)
    alpha = opt['alpha0']
    eta = 0.2
    
    rhols = 1e-6
    gamma = 0.85
    Q = 1
    Cval = f
    for k in range(opt['maxit']):
        fp = f
        gp = g
        xp = x
        nls = 1
        while 1:
            x = xp - alpha * gp
            r = np.dot(A, x) - b
            g = np.dot(A.T, r)
            huber_g = np.sign(x)
            idx = abs(x) < opt['sigma']
            huber_g[idx] = x[idx] / opt['sigma']
            f = 0.5 * np.linalg.norm(r, 2) ** 2 + \
                mu * (np.sum(x[abs(x) >= opt['sigma']] - opt['sigma'] / 2))
            g = g + mu * huber_g
            if f <= Cval - alpha * rhols * nrmG ** 2 or nls >= 10:
                break
            alpha = eta * alpha 
            nls += 1
        nrmG = np.linalg.norm(g, 2)
        forg = 0.5 * np.linalg.norm(r, 2) ** 2 + mu0 * np.linalg.norm(x, 1)
        out['fvec'] = [out['fvec'], forg]
        if opt['verbose']:
            print('%4d\t %.4e \t %.1e \t %.2e \t %2d \n'%(k, f, nrmG, alpha, nls))
        if nrmG < opt['gtol'] or abs(fp - f) < opt['ftol']:
            break
        dx = x - xp
        xg = g - gp
        dxg = abs(np.matmul(dx.T, dx))
        if dxg > 0:
            if k % 2 == 0:
                alpha = np.matmul(dx.T, dx) / dxg
            else:
                alpha = dxg / np.matmul(dg.T, dg)
            alpha = max(min(alpha, 1e12), 1e-12)
        Qp = Q
        Q = gamma * Qp + 1
        Cval = (gamma * Qp * Cval + f) / Q
    out['flag'] = k == opt['maxit']
    out['fval'] = f
    out['itr'] = k
    out['tt'] = time.time() - tic
    out['nrmG'] = nrmG
    return [x, out]
```

### 连续化策略

```python
import time
import numpy as np

optsp = {'maxit': 30, 'maxit_inn':1, 'ftol': 1e-8, 'gtol': 1e-6, 
        'factor': 0.1, 'verbose': 1, 'mul': 100, 'opts1':{},
        'etaf': 1e-1, 'etag': 1e-1}
optsp['gtol_init_ratio'] = 1 / optsp['gtol']
optsp['ftol_init_ratio'] = 1e5
def prox(x, mu):
    y = np.max(np.abs(x) - mu, 0)
    y = np.dot(np.sign(x), y)
    return y
def Func(A, b, mu0, x):
    w = np.dot(A, x) - b
    f = 0.5 * (np.matmul(w.T, w)) + mu0 * np.linalg.norm(x, 1)
    return f
def LASSO_con(x0, A, b, mu0, opts):
    L = max(np.linalg.eig(np.matmul(A.T, A))[0])
    for i in optsp.keys():
        if opts.get(i, -1) == -1:
            opts[i] = optsp[i]
    if not opts['alpha0']: opts['alpha0'] = 1 / L
    out = {}
    out['fvec'] = []
    k = 0
    x = x0
    mu_t = opts['mul']
    tic = time.time()
    f = Func(A, b, mu_t, x)
    opts1 = opts['opts1']
    opts1['ftol'] = opts['ftol'] * opts['ftol_init_ratio']
    opts1['gtol'] = opts['gtol'] * opts['gtol_init_ratio']
    out['itr_inn'] = 0
    while k < opts['maxit']:
        opts1['maxit'] = opts['maxit_inn']
        opts1['gtol'] = max(opts1['gtol'] * opts['etag'], opts['gtol'])
        opts1['ftol'] = max(opts1['ftol'] * opts['etaf'], opts['ftol'])
        opts1['verbose'] = opts['verbose'] > 1
        opts1['alpha0'] = opts['alpha0']
        if opts['method'] == 'grad_huber':
            opts1['sigma'] = 1e-3 * mu_t
        fp = f
        [x, out1] = LASSO_grad_huber_inn(x, A, b, mu_t, mu0, opts1)
        f = out1['fvec'][-1]
        out['fvec'].extend(out1['fvec'])# = [out['fvec'], out1['fvec']]
        k += 1
        nrmG = np.linalg.norm(x - prox(x - np.matmul(A.T, (np.matmul(A, x) - b)), mu0),2)
        if opt['verbose']:
            print('itr: %d\tmu_t: %e\titr_inn: %d\tfval: %e\tnrmG: %.1e\n'%(k, mu_t, out1.itr, f, nrmG))
        if not out1['flag']:
            mu_t = max(mu_t * opts['factor'], mu0)
        if mu_t == mu0 and (nrmG < opts['gtol'] or abs(f - fp) < opts['ftol']):
            break
        out['itr_inn'] = out['itr_inn'] + out1['itr']
    out['fval'] = f
    out['tt'] = time.time() - tic
    out['itr'] = k
    return [x, out]
```

##　参考文献

[为什么Ｌ１正则化导致稀疏解](https://blog.csdn.net/b876144622/article/details/81276818)

[线搜索方法](https://blog.csdn.net/fangqingan_java/article/details/46405669)

[凸优化笔记15：梯度下降法](https://zhuanlan.zhihu.com/p/137274399)