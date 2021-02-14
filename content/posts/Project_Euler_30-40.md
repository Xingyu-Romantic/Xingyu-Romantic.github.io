---
author: "xingyu"
author_link: ""
title: "Project_Euler_30 40"
date: 2021-01-28T15:44:36+08:00
lastmod: 2021-01-28T15:44:36+08:00
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

<!--more-->

## [Coin sums](https://projecteuler.net/problem=31)

### 题目描述

In the United Kingdom the currency is made up of pound (£) and pence (p). There are eight coins in general circulation:

> 1p, 2p, 5p, 10p, 20p, 50p, £1 (100p), and £2 (200p).

It is possible to make £2 in the following way:

> 1×£1 + 1×50p + 2×20p + 1×5p + 1×2p + 3×1p

How many different ways can £2 be made using any number of coins?

### 题解

```python
# 动态规划，  组合问题
coins = [1, 2, 5, 10, 20, 50, 100, 200]
target = 200
dp = [0] * (target + 1)
dp[0] = 1
for i in coins:
    for j in range(1, target + 1):
        if j < i:continue
        dp[j] += dp[j - i]
        print(dp[i])
print(dp[200])
```

### 扩展， 动态规划问题

DP定义三部曲：a. 定义子问题   b. 定义状态数组  c.定义状态转移方程

问题：多少种方法，组合为target

子问题：$dp[j] = dp[j - 1] + dp [j - 2] + dp[j-5]+... +dp[j - 200]$，即求解第j即求解第j-1、j-2、j-5、...、j-200，之和

状态数组：dp[j]

状态转移方程：$dp[j] = dp[j - 1] + dp [j - 2] + dp[j-5]+... +dp[j - 200]$

## [Pandigital products](https://projecteuler.net/problem=32)

### 题目描述

We shall say that an n-digit number is pandigital if it makes use of all the digits 1 to n exactly once; for example, the 5-digit number, 15234, is 1 through 5 pandigital.

The product 7254 is unusual, as the identity, 39 × 186 = 7254, containing multiplicand, multiplier, and product is 1 through 9 pandigital.

Find the sum of all products whose multiplicand/multiplier/product identity can be written as a 1 through 9 pandigital.

HINT: Some products can be obtained in more than one way so be sure to only include it once in your sum.

### 题解

```python
import math

k = set([str(i) for i in range(1, 10)])
print(k)
def ispandigital(m):
    product = set()
    for i in range(1, int(math.sqrt(m)) + 1):
        if m % i == 0:
            for j in str(i):
                product.add(j)
            for j in str(m // i):
                product.add(j)
            for j in str(m):
                product.add(j)
            if product == k:
                return True
            product = set()
    return False
sum = 0
for i in range(1, 10000):
    if ispandigital(i):
        sum += i
    if i % 1000 :
        print(i)
print(sum)
```

## [ Digit cancelling fractions](https://projecteuler.net/problem=33)

### 题目描述

The fraction 49/98 is a curious fraction, as an inexperienced mathematician in attempting to simplify it may incorrectly believe that 49/98 = 4/8, which is correct, is obtained by cancelling the 9s.

We shall consider fractions like, 30/50 = 3/5, to be trivial examples.

There are exactly four non-trivial examples of this type of fraction, less than one in value, and containing two digits in the numerator and denominator.

If the product of these four fractions is given in its lowest common terms, find the value of the denominator.

### 题解

最后求最简分数的分母

```python
def isequal(fenzi, fenmu):
    if fenzi % 10 == 0 and fenmu % 10 == 0:
        return False
    
    div = fenzi / fenmu
    fenzi = str(fenzi)
    fenmu = str(fenmu)
    if set(fenzi) &  set(fenmu) == set():
        return False
    fenzi_ = ''.join(set(fenzi) - (set(fenmu) & set(fenzi)))
    fenmu_ = ''.join(set(fenmu) - (set(fenmu) & set(fenzi)))
    try:
        if eval(fenzi_) / eval(fenmu_) == div and fenzi != fenzi_ and fenmu != fenmu_:
            return True
    except:
        return False
    return False

def gcd(a, b):
    yu = a % b
    if yu == 0:
        return b
    return gcd(b, yu)

result = []
for i in range(10,100):
    for j in range(i+1, 100):
        if isequal(i, j):
            result.append((i, j))
ans_1 = 1
ans_2 = 1
for i in result:
    ans_1 *= i[0]
    ans_2 *= i[1]

print(ans_1, ans_2, ans_2 // gcd(ans_1, ans_2))
```

## [Digit factorials](https://projecteuler.net/problem=34)

### 题目描述

145 is a curious number, as 1! + 4! + 5! = 1 + 24 + 120 = 145.

Find the sum of all numbers which are equal to the sum of the factorial of their digits.

Note: As 1! = 1 and 2! = 2 are not sums they are not included.

### 题解

```python
import math

def isCur(k):
    ans = 0
    for i in str(k):
        ans += math.factorial(int(i))
    if ans == k:
        return True
    return False

ans = 0
for i in range(10, 1000000):
    if isCur(i):
        ans += i

print(ans) 
```

## [Circular primes](https://projecteuler.net/problem=35)

### 题目描述

The number, 197, is called a circular prime because all rotations of the digits: 197, 971, and 719, are themselves prime.

There are thirteen such primes below 100: 2, 3, 5, 7, 11, 13, 17, 31, 37, 71, 73, 79, and 97.

How many circular primes are there below one million?

### 题解

```python
import math

def isprime(num):
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True


def getroll(num):
    num = str(num)
    length = len(num)
    for i in range(length):
        tmp = int(num[i:] + num[:i])
        if not isprime(tmp):
            return False
    return True
idx = 0
for i in range(2, 100 * 10000):
    if getroll(i):
        idx+=1
    if i % 1000 == 0:
        print(i)
print(idx)

print(getroll(197))
```

## [Double-base palindromes](https://projecteuler.net/problem=36)

### 题目描述

The decimal number, 585 = $1001001001_2$ (binary), is palindromic in both bases.

Find the sum of all numbers, less than one million, which are palindromic in base 10 and base 2.

(Please note that the palindromic number, in either base, may not include leading zeros.)

### 题解

二进制和十进制下均为回文数的和

```python

def ispal(nums):
    b_num = str(bin(nums)[2:])
    if b_num == b_num[::-1]:
        return True
    return False

ans = 0
for i in range(100 * 10000):
    if ispal(i) and str(i) == str(i)[::-1]:
        ans += i
    if i % 1000 == 0:
        print(i)
print(ans)
```

## [Truncatable primes](https://projecteuler.net/problem=37)

### 题目描述

The number 3797 has an interesting property. Being prime itself, it is possible to continuously remove digits from left to right, and remain prime at each stage: 3797, 797, 97, and 7. Similarly we can work from right to left: 3797, 379, 37, and 3.

Find the sum of the only eleven primes that are both truncatable from left to right and right to left.

NOTE: 2, 3, 5, and 7 are not considered to be truncatable primes.

### 题解

```python
import math
def isprime(nums):
    if nums == 1:
        return False
    for i in range(2, int(math.sqrt(nums)) + 1):
        if nums % i == 0:
            return False
    return True

def jieduan(nums):
    result = set()
    for i in range(len(str(nums))):
        try:
            result.add(int(str(nums)[i:]))
            result.add(int(str(nums)[:-1 * i - 1]))
        except:
            pass
    return result

ans = 0
idx = 0
for i in range(10, 1000000):
    for j in jieduan(i):
        if not isprime(j):
            break
    else:
        ans += i
        idx += 1
    
print(ans, idx)
```

## [Pandigital multiples](https://projecteuler.net/problem=38)

### 题目描述

Take the number 192 and multiply it by each of 1, 2, and 3:

> 192 × 1 = 192
> 192 × 2 = 384
> 192 × 3 = 576

By concatenating each product we get the 1 to 9 pandigital, 192384576. We will call 192384576 the concatenated product of 192 and (1,2,3)

The same can be achieved by starting with 9 and multiplying by 1, 2, 3, 4, and 5, giving the pandigital, 918273645, which is the concatenated product of 9 and (1,2,3,4,5).

What is the largest 1 to 9 pandigital 9-digit number that can be formed as the concatenated product of an integer with (1,2, ... , n) where n > 1?

### 题解

1. 不能有重复数字
2. 不能有0
3. 长度等于9

```python
def pandigital(num):
    result = ''
    for i in range(1, 10):
        tmp = str(num * i)
        for i in tmp:
            if i in result:
                return 0
        if len(set(tmp)) != len(tmp):
            break
        if '0' in tmp:
            break
        result += tmp
        if len(result) == 9:
            return int(result)
    return 0
max = 0
idx = 0
for i in range(10001):
    tmp = pandigital(i)
    if tmp > max:
        max = tmp
        idx = i
        print(max, idx)
print(max, idx)
```

## [Integer right triangles](https://projecteuler.net/problem=39)

### 题目描述

If *p* is the perimeter of a right angle triangle with integral length sides, {*a*,*b*,*c*}, there are exactly three solutions for *p* = 120.

{20,48,52}, {24,45,51}, {30,40,50}

For which value of *p* ≤ 1000, is the number of solutions maximised?

### 题解

暴力

```python
def sovle(p):
    ans = 0
    for a in range(p//2):
        for b in range(a, p//2):
            if a**2 + b**2 == (p-a-b)**2:
                ans += 1
    return ans
max = 0
idx = 0 
for i in range(1000):
    tmp = sovle(i)
    if tmp > max:
        max = tmp
        idx = i
        print(max, i)
print(max)
```

## [Champernowne's constant](https://projecteuler.net/problem=40)

### 题目描述

An irrational decimal fraction is created by concatenating the positive integers:

0.12345678910**1**112131415161718192021...

It can be seen that the $12^{th}$ digit of the fractional part is 1.

If $d_n$ represents the $n^{th}$ digit of the fractional part, find the value of the following expression.

$d_1 \times d_{10} \times d_{100} \times d_{1000} \times d_{10000} \times d_{100000} \times d_{1000000}$

### 题解

```python
ans = ''
i = 0
while len(ans) < 1000001:
    ans += str(i)
    i += 1
j = 1
result = 1
while j < 1000001:
    result *= int(ans[j])
    j *= 10
print(result)
```

