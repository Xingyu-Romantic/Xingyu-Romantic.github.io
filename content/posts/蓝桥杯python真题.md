---
author: "xingyu"
author_link: ""
title: "蓝桥杯python真题"
date: 2021-02-04T16:46:49+08:00
lastmod: 2021-02-04T16:46:49+08:00
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

看看真题

<!--more-->

## 填空题

### 在计算机存储中，12.5MB是多少个字节

```
12.5 * 1024 * 1024 * 8 = 104857600
```

### 一共包含有2019个结点的有向图，最多包含多少条边

2019个结点，两个结点增加一个入度，一个出度。

2个结点， 2条边

3个结点， 6条边

4个结点， 12条边

```
2019 * 2018 = 4074342
```

### 将LANQIAO的字母重新排列，可以得到不同的单词，如LANQIAO、AAILNOQ等，注意这7个字母都要被用上，单词不一定有具体的英文意义。

两个字母相同，全排列有 7 ！， 则 

```python
7! / 2 = 5040 / 2 = 2520
```

```python
import collections

def permutations(arr, start, end):
    if start == end:
        result.add(''.join(arr))
    else:
        for index in range(start, end):
            arr[index], arr[start] = arr[start], arr[index]
            permutations(arr, start+1, end)
            arr[index], arr[start] = arr[start], arr[index]

k = 'LANQIAO'
arr = []
arr.extend(k)
result = set()
permutations(arr, 0, len(arr))

print(len(result))
```

### 由1对括号，可以组成一种合法括号序列：（）。由两对括号，可以组成两种合法括号序列：（）、（（）），由4对括号组成的合法序列一共有多少种？

思路：全排列，然后判断是否合法。 14种

```python
from collections import deque
def permutations(arr, start, end):
    if start == end:
        result.add(''.join(arr))
    else:
        for i in range(start, end):
            arr[start], arr[i] = arr[i], arr[start]
            permutations(arr, start + 1, end)
            arr[start], arr[i] = arr[i], arr[start]

def islegal(arr):
    stack = deque()
    for i in arr:
        if i == '(':
            stack.append(i)
        elif i == ')':
            if len(stack) == 0:
                return False
            if stack[-1] == '(':
                stack.pop()
    print(stack)
    if len(stack) == 0:
        return True
    return False

s = '()()()()'
arr = []
arr.extend(s)
idx = 0
result = set()
permutations(arr, 0, len(s))
for i in result:
    if islegal(i):
        idx += 1
        print(i, idx)
print(idx)
```

## 编程题

### 反倍数

给定三个整数a, b, c, 如果一个整数既不是a整数倍，也不是b的整数倍，也不是c的整数倍，则这个整数为反倍数。请问在1至n中有多少个反倍数。

【输入格式】
输入的第一行包含一个整数 n。
第二行包含三个整数 a, b, c，相邻两个数之间用一个空格分隔。

【输出格式】
输出一行包含一个整数，表示答案。
【样例输入】
30
2 3 6
【样例输出】
10
【样例说明】
以下这些数满足要求：1, 5, 7, 11, 13, 17, 19, 23, 25, 29。
【评测用例规模与约定】
对于 40% 的评测用例，1 <= n <= 10000。
对于 80% 的评测用例，1 <= n <= 100000。
对于所有评测用例，1 <= n <= 1000000，1 <= a <= n，1 <= b <= n，1 <= c <= n。

```python
n = eval(input())
a, b, c = input().split(' ')
a, b, c = eval(a), eval(b), eval(c)
ans = 0
for i in range(n + 1):
	if i % a != 0 and i % b != 0 and i % c !=0:
		ans += 1
print(ans)
```

### 凯撒密码

给定一个单词，请使用凯撒密码将这个单词加密。凯撒密码是一种替换加密的技术，单词中的所有字母都在字母表上向后偏移3位后被替换成密文。即a变为d，b变为e，...，w变为z，x变为a，y边为b，z变为c。

```python
s = input()
ans = ''
for i in s:
	ans += chr((ord(i) - ord('a') + 3) % 26 + ord('a'))
print(ans)

```

### 螺旋

对于一共n行m列的表格，我们可以使用螺旋的方式给表格依次填上正整数，我们称填好的表格为一个螺旋矩阵，

例如，一个4行5列的螺旋矩阵如下：

| 1    | 2    | 3    | 4    | 5    |
| ---- | ---- | ---- | ---- | ---- |
| 14   | 15   | 16   | 17   | 6    |
| 13   | 20   | 19   | 18   | 7    |
| 12   | 11   | 10   | 9    | 8    |

【输入格式】
输入的第一行包含两个整数 n, m，分别表示螺旋矩阵的行数和列数。
第二行包含两个整数 r, c，表示要求的行号和列号。
【输出格式】
输出一个整数，表示螺旋矩阵中第 r 行第 c 列的元素的值。
【样例输入】
4 5
2 2
【样例输出】
15
【评测用例规模与约定】
对于 30% 的评测用例，2 <= n, m <= 20。
对于 70% 的评测用例，2 <= n, m <= 100。
对于所有评测用例，2 <= n, m <= 1000，1 <= r <= n，1 <= c <= m。

```python
n, m = (eval(i) for i in input().split(' '))
r, c = (eval(i) for i in input().split(' '))

map = [[0 for i in range(m)] for j in range(n)]
up, down, left, right = 0, n, 0, m
num = 1
while num < n * m + 1:
	for x in range(left, right):
		map[up][x] = num
		num += 1
	up += 1
	for y in range(up, down):
		map[y][right-1] = num
		num += 1
	right -= 1
	for x in range(right-1, left - 1, -1):
		map[down-1][x] = num
		num += 1
	down -= 1
	for y in range(down-1, up-1, -1):
		map[y][left] = num
		num += 1
	left += 1
print(map[r-1][c-1])
```

### 摆动序列

如果一个序列的奇数项都比前一项大，偶数项都比前一项小，则称为一个摆动序列。即a[2i], a[2i-1], a[2i+1], a[2i] 小明想知道，长度为m，每个数都是1到n之间的正整数的摆动序列一共有多少个。

【输入格式】
输入一行包含两个整数 m，n。
【输出格式】
输出一个整数，表示答案。答案可能很大，请输出答案除以10000的余数。
【样例输入】
3 4
【样例输出】
14
【样例说明】
以下是符合要求的摆动序列：
2 1 2
2 1 3
2 1 4
3 1 2
3 1 3
3 1 4
3 2 3
3 2 4
4 1 2
4 1 3
4 1 4
4 2 3
4 2 4
4 3 4
【评测用例规模与约定】
对于 20% 的评测用例，1 <= n, m <= 5；
对于 50% 的评测用例，1 <= n, m <= 10；
对于 80% 的评测用例，1 <= n, m <= 100；
对于所有评测用例，1 <= n, m <= 1000。

**DFS 版本**

