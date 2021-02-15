---
author: "xingyu"
author_link: ""
title: "Project_Euler_刷题 10 ～ 20"
date: 2021-01-19T23:41:39+08:00
lastmod: 2021-01-19T23:41:39+08:00
draft: false
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

会随着刷题不断更新。。。。

前面的没想着要写笔记，就没有存下代码，故从Problem11来记录代码

全由Python实现。 10～20题

<!--more-->

## [Largest product in a grid](https://projecteuler.net/problem=11)

### 题目描述

In the 20×20 grid below, four numbers along a diagonal line have been marked in red.

08 02 22 97 38 15 00 40 00 75 04 05 07 78 52 12 50 77 91 08
49 49 99 40 17 81 18 57 60 87 17 40 98 43 69 48 04 56 62 00
81 49 31 73 55 79 14 29 93 71 40 67 53 88 30 03 49 13 36 65
52 70 95 23 04 60 11 42 69 24 68 56 01 32 56 71 37 02 36 91
22 31 16 71 51 67 63 89 41 92 36 54 22 40 40 28 66 33 13 80
24 47 32 60 99 03 45 02 44 75 33 53 78 36 84 20 35 17 12 50
32 98 81 28 64 23 67 10 **26** 38 40 67 59 54 70 66 18 38 64 70
67 26 20 68 02 62 12 20 95 **63** 94 39 63 08 40 91 66 49 94 21
24 55 58 05 66 73 99 26 97 17 **78** 78 96 83 14 88 34 89 63 72
21 36 23 09 75 00 76 44 20 45 35 **14** 00 61 33 97 34 31 33 95
78 17 53 28 22 75 31 67 15 94 03 80 04 62 16 14 09 53 56 92
16 39 05 42 96 35 31 47 55 58 88 24 00 17 54 24 36 29 85 57
86 56 00 48 35 71 89 07 05 44 44 37 44 60 21 58 51 54 17 58
19 80 81 68 05 94 47 69 28 73 92 13 86 52 17 77 04 89 55 40
04 52 08 83 97 35 99 16 07 97 57 32 16 26 26 79 33 27 98 66
88 36 68 87 57 62 20 72 03 46 33 67 46 55 12 32 63 93 53 69
04 42 16 73 38 25 39 11 24 94 72 18 08 46 29 32 40 62 76 36
20 69 36 41 72 30 23 88 34 62 99 69 82 67 59 85 74 04 36 16
20 73 35 29 78 31 90 01 74 31 49 71 48 86 81 16 23 57 05 54
01 70 54 71 83 51 54 69 16 92 33 48 61 43 52 01 89 19 67 48

The product of these numbers is 26 × 63 × 78 × 14 = 1788696.

What is the greatest product of four adjacent numbers in the same direction (up, down, left, right, or diagonally) in the 20×20 grid?

Answer: **70600674**

### 代码

```python
with open('maze.txt') as f:
    temp = f.readlines()

grid = []
for i in temp:
    tmp = i.strip('\n').split(' ')
    grid.append(tmp)

for i in range(0, len(grid)):
    for j in range(0, len(grid)):
        grid[i][j] = int(grid[i][j])

def mul(a: list):
    result = 1
    for i in a:
        result *= i
    return result

max = 0
result_max = []
for i in range(20):
    for j in range(0, 20):
        try:
            tmp = mul(grid[i][j:j+4])
            if tmp > max: max = tmp
        except:
            continue
result_max.append(max)

max = 0
for i in range(20):
    for j in range(20):
        try:
            a = [grid[i][j], grid[i+1][j], grid[i+2][j], grid[i+3][j]]
            tmp = mul(a)
            if tmp > max: 
                #print(a)
                max = tmp
        except:
            continue
result_max.append(max)

max = 0
for i in range(0, 20):
    for j in range(0, 20):
        try:
            a =[grid[i][j], grid[i+1][j+1], grid[i+2][j+2], grid[i+3][j+3]]
            tmp = mul(a)
        #print(a, tmp)
            if tmp > max: 
                max=tmp
        except:
            continue
result_max.append(max)
max = 0
for i in range(0, 20):
    for j in range(0, 20):
        try:
            a =[grid[i][j], grid[i+1][j-1], grid[i+2][j-2], grid[i+3][j-3]]
            
            tmp = mul(a)
            #print(a, tmp)
            if tmp > max: 
                max=tmp
        except:
            continue

result_max.append(max)

print(result_max)
```

## [Highly divisible triangular number](https://projecteuler.net/problem=12)

### 题目描述

The sequence of triangle numbers is generated by adding the natural numbers. So the 7th triangle number would be 1 + 2 + 3 + 4 + 5 + 6 + 7 = 28. The first ten terms would be:

1, 3, 6, 10, 15, 21, 28, 36, 45, 55, ...

Let us list the factors of the first seven triangle numbers:

>  **1**: 1
>  **3**: 1,3
>  **6**: 1,2,3,6
> **10**: 1,2,5,10
> **15**: 1,3,5,15
> **21**: 1,3,7,21
> **28**: 1,2,4,7,14,28

We can see that 28 is the first triangle number to have over five divisors.

What is the value of the first triangle number to have over five hundred divisors?

### 代码

自己写的暴力太拉跨了，贼慢。这里放上百(谷)度(歌)到的代码

```python
from math import sqrt 
import time 
def natSum(n):
    x = 1
    count = 0
    sum = 0
    while count <= n:
        sum += x
        count = 0
        for i in range(1,int(sqrt(sum))+1):
            if sum % i == 0:
                count += 2
        if sqrt(sum)==int(sqrt(sum)): 
                count -= 1
        print(x,sum,count,n)

        x += 1
    
natSum(500)
```

## [Large sum](https://projecteuler.net/problem=13)

### 题目描述

Work out the first ten digits of the sum of the following one-hundred 50-digit numbers.

100个50位的数字，太长就不放了

###　题解

```python
with open('50位.txt') as f:
    nums = f.readlines()

nums = [int(i) for i in nums]

result = 0
for i in nums:
    result += i
print(str(result)[:10])
```

## [Longest Collatz sequence](https://projecteuler.net/problem=14)

### 题目描述

The following iterative sequence is defined for the set of positive integers:

n → n/2 (n is even)
n → 3n + 1 (n is odd)

Using the rule above and starting with 13, we generate the following sequence:

13 → 40 → 20 → 10 → 5 → 16 → 8 → 4 → 2 → 1

It can be seen that this sequence (starting at 13 and finishing at 1) contains 10 terms. Although it has not been proved yet (Collatz Problem), it is thought that all starting numbers finish at 1.

Which starting number, under one million, produces the longest chain?

**NOTE:** Once the chain starts the terms are allowed to go above one million.

### 题解

#### １.暴力

```python
import time
def iterative(n):
    if n % 2 == 0:
        return n / 2
    else:
        return 3 * n + 1


def produce(start):
    collatz = [start]
    #length = 1
    while collatz[-1] != 1:
        collatz.append(iterative(collatz[-1]))
        #length += 1
        #del collatz[0]
    return len(collatz)

max = 0
k = 0
for i in range(2, 100 * 10000):
    d = produce(i)
    if d > max: 
        max = d
        k = i
    if i % 1000: print(i)
print(k, max)
```

#### 2. 放入字典，重复不再计算

比上面的快了100倍

```python
length = dict()
length[1] = 1
def iterative(k):
    if k in length:
        return length[k]
    if k == 1:
        return 1
    if k % 2 == 0:
        if k / 2 in length:
            return 1 + length[k/2]
        length[k/2] = 1 + iterative(k / 2)
        return length[k/2]
    else:
        if 3 * k + 1 in length:
            return 1 + length[3 * k + 1]
        return 1 + iterative(3 * k + 1)

print(iterative(13))
max = 0
result = 0
for i in range(1, 100 * 10000):
    temp = iterative(i)
    if temp > max:
        max = temp
        result = i
        print(i)
print(result, max)
```

## [Lattice paths](https://projecteuler.net/problem=15)

### 题目描述

Starting in the top left corner of a 2×2 grid, and only being able to move to the right and down, there are exactly 6 routes to the bottom right corner.

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210121131450.png)

How many such routes are there through a 20×20 grid?

###　题解

是一道最最基础的动态规划，所以我能做出来。。

```python
N = 22
dp = [[0 for i in range(N)] for j in range(N)]

for i in range(N):
    dp[0][i] = 1
    dp[i][0] = 1

for i in range(N):
    for j in range(N):
        if i!=0 and j!=0:
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

print(dp[20][20])
```

##　[Power digit sum](https://projecteuler.net/problem=16)

###　题目描述

$2^{15} =32768$  and the sum of its digits is 3 + 2 + 7 + 6 + 8 = 26.

What is the sum of the digits of the number $2^{1000}$?

### 题解

```python
sum = 0
for i in str(2**1000):
    sum += int(i)
print(sum)
```

## [Number letter counts](https://projecteuler.net/problem=17)

### 题目描述

If the numbers 1 to 5 are written out in words: one, two, three, four, five, then there are 3 + 3 + 5 + 4 + 4 = 19 letters used in total.

If all the numbers from 1 to 1000 (one thousand) inclusive were written out in words, how many letters would be used?



**NOTE:** Do not count spaces or hyphens. For example, 342 (three hundred and forty-two) contains 23 letters and 115 (one hundred and fifteen) contains 20 letters. The use of "and" when writing out numbers is in compliance with British usage.

### 题解

one t

## [Maximum path sum I](https://projecteuler.net/problem=18)

### 题目描述

By starting at the top of the triangle below and moving to adjacent numbers on the row below, the maximum total from top to bottom is 23.

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210121193438.png)

That is, 3 + 7 + 4 + 9 = 23.

Find the maximum total from top to bottom of the triangle below:

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210121193424.png)

**NOTE:** As there are only 16384 routes, it is possible to solve this problem by trying every route. However, [Problem 67](https://projecteuler.net/problem=67), is the same challenge with a triangle containing one-hundred rows; it cannot be solved by brute force, and requires a clever method! ;o)

### 题解

dfs，从下往上，逐步减小，

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210122110035.png)

```python

with open('triangle.txt') as f:
    triangle = f.readlines()

N = 100
curr = []
next = []
F = [[0 for i in range(N)] for i in range(N)]
for i in range(len(triangle)):
    curr.append(triangle[i].strip('\n').split(' '))
def dfs(i, k):
    if i == len(triangle) - 1:
        return  int(curr[i][k])
    if F[i][k] != 0: return F[i][k]
    F[i][k] = max(dfs(i+1, k), dfs(i+1, k+1)) + int(curr[i][k])
    return F[i][k]

print(dfs(0,0))
```

## [Counting Sundays](https://projecteuler.net/problem=19)

### 题目描述

You are given the following information, but you may prefer to do some research for yourself.

- 1 Jan 1900 was a Monday.
- Thirty days has September,
  April, June and November.
  All the rest have thirty-one,
  Saving February alone,
  Which has twenty-eight, rain or shine.
  And on leap years, twenty-nine.
- A leap year occurs on any year evenly divisible by 4, but not on a century unless it is divisible by 400.

How many Sundays fell on the first of the month during the twentieth century (1 Jan 1901 to 31 Dec 2000)?

### 题解

对datetime的使用， 很不错的库

```python
import datetime

start = datetime.date(1901, 1, 1)
end = datetime.date(2000, 12, 31)



result = 0
while start < end:
    if start.strftime("%d") == '01' and  start.isoweekday() == 7:
        result+=1
    start = start + datetime.timedelta(days=1)
    print(start, start.strftime("%d"))
print(result)

```

## [Factorial digit sum](https://projecteuler.net/problem=20)

###　题目描述

*n*! means *n* × (*n* − 1) × ... × 3 × 2 × 1

For example, 10! = 10 × 9 × ... × 3 × 2 × 1 = 3628800,
and the sum of the digits in the number 10! is 3 + 6 + 2 + 8 + 8 + 0 + 0 = 27.

Find the sum of the digits in the number 100!

### 题解

```python
import math
sum = 0 
k = math.factorial(100)
for i in range(len(str(k))):
    sum += int(str(k)[i])

print(sum)
```
