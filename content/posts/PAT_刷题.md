---
author: "xingyu"
author_link: ""
title: "PAT_刷题"
date: 2021-02-01T11:32:57+08:00
lastmod: 2021-02-01T11:32:57+08:00
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

PAT  https://pintia.cn/problem-sets/994805342720868352/problems/type/7

<!--more-->

## **1001** **A+B Format**

Calculate *a*+*b* and output the sum in standard format -- that is, the digits must be separated into groups of three by commas (unless there are less than four digits).

### Input Specification:

Each input file contains one test case. Each case contains a pair of integers *a* and *b* where −106≤*a*,*b*≤106. The numbers are separated by a space.

### Output Specification:

For each test case, you should output the sum of *a* and *b* in one line. The sum must be written in the standard format.

### Sample Input:

```in
-1000000 9
```

### Sample Output:

```out
-999,991
```

### 题解

主要是对最后千分位符的处理

```python
def prin(s):
    tmp = str(s)[::-1]
    result = ''
    for i in range(len(tmp)):
        if i % 3 == 0 and i > 0:
            result += ','
        result += tmp[i]
    return result[::-1]
a, b = input().split(' ')
sum = eval(a) + eval(b)
if sum < 0:
    print('-' + prin(-sum))
else:
    print(prin(sum))
```

## **1002** **A+B for Polynomials**

This time, you are supposed to find *A*+*B* where *A* and *B* are two polynomials.

### Input Specification:

Each input file contains one test case. Each case occupies 2 lines, and each line contains the information of a polynomial:

*K* *N*1 *a**N*1 *N*2 *a**N*2 ... *N**K* *a**N**K*

where *K* is the number of nonzero terms in the polynomial, *N**i* and *a**N**i* (*i*=1,2,⋯,*K*) are the exponents and coefficients, respectively. It is given that 1≤*K*≤10，0≤*N**K*<⋯<*N*2<*N*1≤1000.

### Output Specification:

For each test case you should output the sum of *A* and *B* in one line, with the same format as the input. Notice that there must be NO extra space at the end of each line. Please be accurate to 1 decimal place.

### Sample Input:

```in
2 1 2.4 0 3.2
2 2 1.5 1 0.5
```

### Sample Output:

```out
3 2 1.5 1 2.9 0 3.2
```

### 题解

