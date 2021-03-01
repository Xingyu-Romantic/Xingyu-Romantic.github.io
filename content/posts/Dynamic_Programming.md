---
author: "Xingyu"
author_link: ""
title: "Dynamic_Programming"
date: 2021-03-01T09:28:49+08:00
lastmod: 2021-03-01T09:28:49+08:00
draft: true
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["Algorithm"]
categories: ["Algorithm"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

动态规划总结

<!--more-->

### 题目特点

**1. 计数**

​	1）有多少种方式走到最下角

​	2）有多少种方法选出k个数使得和是Sum

**2. 求最大最小值**

​	1）从左上角走到右下角路径的最大数字和

​	2）最长上升子序列长度

**3. 求存在性**

​	1）取石子游戏，先手是否必胜

​	2）能不能选出k个数使得和是Sum

## 例题分析

### 零钱兑换  （Coin-Charge)

https://leetcode-cn.com/problems/coin-change/submissions/

**最后一步：**

* 虽然不知道最优策略一定有最后的硬币：$a_k$
* 除掉这枚硬币，前面硬币的面值加起来是27 - $a_k$

关键1：不关心前面的k-1枚硬币是怎么拼出27 - $a_k$的

关键2： 因为是最优策略，所以拼出27 - $a_k$的硬币数量一定要最少

**子问题**

* 子：最少用多少枚硬币可以拼出 27 - $a_k$
* 原：原问题是最少用多少枚硬币拼出27
* 将原问题转化为了一个子问题， 而且规模更小：27 - $a_k$
* 简化定义， 设状态$f(x)$ = 最少用多少枚硬币拼出X

**转移方程**

$f(X) = min\{f(X-2) +1,f(X-5)+1,f(X-7)+1\}$

**初始条件和边界情况**

* X-2，X-5，X-7，小于0？ 什么时候停
* 如果不能拼出Y， 定义f(Y) = inf
* 所以$f(1) = min\{f(-1)+1,f(-4)+1,f(-6)+1\}$ 为inf，拼不出来。

* 初始条件：$f(0) =0$

**计算顺序**

* 拼出X所需的最少硬币数：$f(X) = min\{f(X-2) +1,f(X-5)+1,f(X-7)+1\}$
* 初始条件：$f(0) = 0$
* 然后计算$f(1), f(2), ... , f(27)$
* 当我们计算到$f(X)$时，$f(X-2), f(X-5), f(X-7)$都已经得到结果了







