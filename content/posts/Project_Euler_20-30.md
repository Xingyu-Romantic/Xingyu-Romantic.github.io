---
author: "xingyu"
author_link: ""
title: "Project_Euler_20 30"
date: 2021-01-23T11:11:46+08:00
lastmod: 2021-01-23T11:11:46+08:00
draft: true
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["Algorithm"]
categories: ["Algortihm"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: false
auto_collapse_toc: true
math: false
---

<!--more-->

## [Amicable numbers](https://projecteuler.net/problem=21)

### 题目描述

Let d(*n*) be defined as the sum of proper divisors of *n* (numbers less than *n* which divide evenly into *n*).
If d(*a*) = *b* and d(*b*) = *a*, where *a* ≠ *b*, then *a* and *b* are an amicable pair and each of *a* and *b* are called amicable numbers.

For example, the proper divisors of 220 are 1, 2, 4, 5, 10, 11, 20, 22, 44, 55 and 110; therefore d(220) = 284. The proper divisors of 284 are 1, 2, 4, 71 and 142; so d(284) = 220.

Evaluate the sum of all the amicable numbers under 10000.

### 题解

```python
def d(n):
    sum = 0
    for i in range(1, n//2 + 1):
        if n % i == 0:
            sum += i
    return sum

result = 0
for i in range(1, 10000):
    t = d(i)
    if i == d(t) and i != t:
        result += i
print(result)
```

## [Names scores](https://projecteuler.net/problem=22)

### 题目描述

Using [names.txt](https://projecteuler.net/project/resources/p022_names.txt) (right click and 'Save Link/Target As...'), a 46K text file containing over five-thousand first names, begin by sorting it into alphabetical order. Then working out the alphabetical value for each name, multiply this value by its alphabetical position in the list to obtain a name score.

For example, when the list is sorted into alphabetical order, COLIN, which is worth 3 + 15 + 12 + 9 + 14 = 53, is the 938th name in the list. So, COLIN would obtain a score of 938 × 53 = 49714.

What is the total of all the name scores in the file?

### 题解

```python

with open('p022_names.txt') as f:
    names = f.read()
names = names.strip('"').split('","')
names = sorted(names)

result = 0

for i in range(len(names)):
    temp_sum = 0
    for j in names[i]:
        temp_sum += (ord(j) - 64)
    result += (temp_sum * (i+1))
print(result)

```

