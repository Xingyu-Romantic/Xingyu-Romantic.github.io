---
author: "xingyu"
author_link: "xingyu"
title: "DJBCTF_童话镇_writeup"
date: 2021-01-24T16:18:22+08:00
lastmod: 2021-01-24T16:18:22+08:00
draft: false
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["CTF", "机器学习"]
categories: ["CTF"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

CTFshow  DJBCTF， 题目新颖。 

https://ctf.show/

<!--more-->

## 童话镇

CTF + 机器学习，  MISC选手逐渐成为全栈。　狸题，　ｙｙｄｓ

###　题目描述

一曲童话镇，多少断肠人？

### 题解

下载后为一段音频童话镇.mp3

```sh
foremost 童话镇.mp3
```

分离后得到压缩包，密码hint为思念。

没搞懂，根据群里师傅说的为作者直播间号，那么根据baidu

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210124163234.png)

压缩密码为 67373

之后就是得到flag.txt 和 t.txt，

第三个hint：爱(AI)

那么t.txt 为有标签 为训练集， flag.txt 为测试集

```python
from sklearn.neighbors import (NeighborhoodComponentsAnalysis,KNeighborsClassifier)
import numpy as np

with open('t.txt') as f:
    t = f.readlines()

with open('flag.txt') as f:
    flag = f.readlines()

X = np.zeros((len(t), 6))
y = np.zeros((len(t), 1))
X_test = np.zeros((len(flag), 6))

for i in range(len(t)):
    k = t[i].strip('\n').split('\t')
    y[i] = k[0]
    idx = 0
    for j in k[1].lstrip('[').rstrip(']').split(','):
        X[i, idx] = int(j)
        idx += 1

for i in range(len(flag)):
    idx = 0
    for j in flag[i].strip('\n').lstrip('[').rstrip(']').split(','):
        X_test[i, idx] = int(j)
        idx += 1

knn = KNeighborsClassifier(n_neighbors=2)

knn.fit(X.reshape(-1, 6), np.ravel(y))
result = ''.join([str(int(i)) for i in knn.predict(X_test).tolist()])
print(len(result))
with open('flagto.txt', "w") as f:
    f.write(result)
```



直接KNN， 然后得到一串长度为 78289的二进制串， 中间无空格，不能转字符串

猜想为图片，正好在做这个题的时候，正在刷每日算法，正好写了一个差不多的函数，求真约数

```python
def perfect(n):
    sum = 0
    for i in range(1, n//2 + 1):
        if n % i == 0:
            sum += i
            print(i)
print(perfect(78289))
1
79
991
```

然后转换为图片。

```python
import cv2
import numpy as np
with open('flagto.txt') as f:
    flag = f.read().strip('\n')

img = np.zeros((78,991,1), np.uint8)
for i in range(78):
    for j in range(991):
        #print(tmp[i][j])
        img[i,j,:] = 255 if int(flag[i*991+j]) else 0

#tmp = np.array(tmp, np.uint8)
#print(tmp)
print(img)
cv2.imshow('Img', img)
cv2.waitKey(0)
```

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210124163912.png)

