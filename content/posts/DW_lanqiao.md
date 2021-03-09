---
author: "xingyu"
author_link: ""
title: "DW_lanqiao"
date: 2021-03-09T14:52:46+08:00
lastmod: 2021-03-09T14:52:46+08:00
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

DataWhale 组织蓝桥杯刷题，记录

<!--more-->

## Task1

### 01字符串

```
对于长度为5位的一个01串，每一位都可能是0或1，一共有32种可能。它们的前几个是：
00000
00001
00010
00011
00100
请按从小到大的顺序输出这32种01串。
```

```python
# My Code
for i in range(32):
    print(bin(i)[2:].rjust(5, '0'))
    
# Official code
for i in range(32):
    t1=i
    temp=[0]*5
    for j in range(5)[::-1]:
        if 2**j<=t1:
            temp[j]=1
            t1=t1-2**j
    print(''.join(map(str,reversed(temp))))
```

### 1083、Hello, World!

**题目描述**

这是要测试的第一个问题。 由于我们都知道ASCII码，因此您的工作很简单：输入数字并输出相应的消息。 

**输入**

> 输入将包含一个由空格（空格，换行符，TAB）分隔的正整数列表。 请处理到文件末尾（EOF）。 整数将不少于32。

**输出**

> 输出相应的消息。 请注意，输出末尾没有换行符。

**样例输入**

```
72 101 108 108 111 44 32 119 111 114 108 100 33
```

**样例输出**

```
Hello, world!
```

```python
# My Code
k = input().split(' ')
res = ''
for i in k:
    res += chr(int(i))
print(res)

# Official Code
while True:
    num=list(map(int,input().strip().split()))
    for i in num:
        print(chr(i),end='')
```

### 用筛法求之N内的素数。

时间限制: 1Sec 内存限制: 64MB 提交: 11990 解决: 7204

**题目描述**

用筛法求之N内的素数。

**输入**

> N

**输出**

> 0～N的素数

**样例输入**

```
100
```

**样例输出**

```
2
3
5
7
11
13
17
19
23
29
31
37
41
43
47
53
59
61
67
71
73
79
83
89
97
```