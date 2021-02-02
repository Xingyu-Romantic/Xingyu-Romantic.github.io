---
author: "xingyu"
author_link: ""
title: "Project_Euler_20 30"
date: 2021-01-23T11:11:46+08:00
lastmod: 2021-01-23T11:11:46+08:00
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

## [ Non-abundant sums](https://projecteuler.net/problem=23)

### 题目描述

A perfect number is a number for which the sum of its proper divisors is exactly equal to the number. For example, the sum of the proper divisors of 28 would be 1 + 2 + 4 + 7 + 14 = 28, which means that 28 is a perfect number.

A number n is called deficient if the sum of its proper divisors is less than n and it is called abundant if this sum exceeds n.

As 12 is the smallest abundant number, 1 + 2 + 3 + 4 + 6 = 16, the smallest number that can be written as the sum of two abundant numbers is 24. By mathematical analysis, it can be shown that all integers greater than 28123 can be written as the sum of two abundant numbers. However, this upper limit cannot be reduced any further by analysis even though it is known that the greatest number that cannot be expressed as the sum of two abundant numbers is less than this limit.

Find the sum of all the positive integers which cannot be written as the sum of two abundant numbers.

### 题解

自己写了一个，不仅速度慢，而且最后结果也不知道为什么不对。

找到一个网站里面的题解，学到很多。

```python
import math


def abun(N):
    Q = dict.fromkeys(range(1, N+1), 0)
    for q in Q:
        for k in [q * n for n in range(1, int(N / q) +1)]:
            if q != k : Q[k] += q
    return [q for q in Q if Q[q] > q]

N = 28123
A = abun(N)
possible = set()

for a in A:
    for b in A:
        if a+b < N:possible.add(a+b)
        else:break
print(sum([p for p in range(N) if p not in possible]))
```

##　[Lexicographic permutations](https://projecteuler.net/problem=24)

### 题目描述

A permutation is an ordered arrangement of objects. For example, 3124 is one possible permutation of the digits 1, 2, 3 and 4. If all of the permutations are listed numerically or alphabetically, we call it lexicographic order. The lexicographic permutations of 0, 1 and 2 are:

012  021  102  120  201  210

What is the millionth lexicographic permutation of the digits 0, 1, 2, 3, 4, 5, 6, 7, 8 and 9?

###　题解

第一位数是0,　其他任意，　共有$9! = 362880$

第一位数是１，其他任意，　共有$9! = 362880$

即，　$100 * 10000 / 362880 = 2 ... 274240$

那么第一位是2，第二位是１的有，$8! = 40320$,   

$274240 / 40320 = 6...32320$，即第二位为7（2不取），那么为 27 xxxxxxx

第三位为１的有 $7! =5040$ ,  

$32320 / 5040 = 6...2080$, 即第三位为 8（2,7不取），278xxxxxxx

第四位　$6!=720$

$2080 / 720 = 2...640$，即第四位为3,  2783xxxxxx

第五位　$5!= 120$

$640 / 120 =5...40$,  即第五位为9,　　27839xxxx

第六位 $4! = 24$

$40 / 24 = 1...16$, 即第六位为1,　　278391xxx

第七位$3!=6$

$16 / 6 =2...4$， 即第七位为5,   2783915xxx

第八位$2!=2$,

$4 / 2=2$, 此时就出现问题了，余数为0, 那么接下来的三位数就是0,4,6 就是第一位为4的字典序最小的。

...., 然后手动其实就可以求出来，写个程序吧

```python
import math
nums = set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
result = 0
N = 100 * 10000
yu = 100 * 10000
for i in range(9, -1, -1):
    idx =yu // math.factorial(i)
    yu = yu % math.factorial(i)
    if yu == 0:
        idx -= 1
    print(nums)
    result += list(nums)[idx] * 10**(i)
    nums.remove(list(nums)[idx])
print(result)
```

## [1000-digit Fibonacci number](https://projecteuler.net/problem=25)

### 题目描述

The Fibonacci sequence is defined by the recurrence relation:

> F*n* = F*n*−1 + F*n*−2, where F1 = 1 and F2 = 1.

Hence the first 12 terms will be:

> F1 = 1
> F2 = 1
> F3 = 2
> F4 = 3
> F5 = 5
> F6 = 8
> F7 = 13
> F8 = 21
> F9 = 34
> F10 = 55
> F11 = 89
> F12 = 144

The 12th term, F12, is the first term to contain three digits.

What is the index of the first term in the Fibonacci sequence to contain 1000 digits?

###　题解

```python

def f(n):
    F = [0, 1, 1]
    if n < 2:
        return len(str(F[n]))
    else:
        for i in range(n - 2):
            F.append(F[-1] + F[-2])
            del F[0]
        return len(str(F[-1]))

i = 100
while f(i) < 1000:
    if i % 100 == 0:
        print(i)
    i += 1
    
print(f(4782))
```

## [Reciprocal cycles](https://projecteuler.net/problem=26)

### 题目描述

A unit fraction contains 1 in the numerator. The decimal representation of the unit fractions with denominators 2 to 10 are given:

> | 1/2  | =    | 0.5        |
> | ---- | ---- | ---------- |
> | 1/3  | =    | 0.(3)      |
> | 1/4  | =    | 0.25       |
> | 1/5  | =    | 0.2        |
> | 1/6  | =    | 0.1(6)     |
> | 1/7  | =    | 0.(142857) |
> | 1/8  | =    | 0.125      |
> | 1/9  | =    | 0.(1)      |
> | 1/10 | =    | 0.1        |

Where 0.1(6) means 0.166666..., and has a 1-digit recurring cycle. It can be seen that 1/7 has a 6-digit recurring cycle.

Find the value of *d* < 1000 for which 1/*d* contains the longest recurring cycle in its decimal fraction part.

### 题解

一个结论，`余数`在一个循环序列中不会重复出现，故取10 % n, 100 % n，计算出每次的余数，出现相同的就直接返回两个位置差即可。

```python
def Count_Cycle(n):
    yu_dict = {}
    #yu_dict[0] = 1  # 存放余数， 余数：idx
    numiteration = 10
    yu = numiteration % n
    idx = 1
    yu_dict[1] = 0
    while yu != 0 and yu not in yu_dict:
        yu_dict[yu] = idx
        numiteration *= 10
        yu = numiteration % n 
        idx += 1
    if yu == 0:
        return 0
    return idx - yu_dict[yu]

max = 0
idx = 0
for _ in range(1, 1000):
    tmp = Count_Cycle(_)
    if tmp > max:
        max = tmp
        idx = _
print(idx, max)

```

## [Quadratic primes](https://projecteuler.net/problem=27)

### 题目描述

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210126122424.png)

### 题解

```python
import math

def isprime(n):
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


max_ = 0
result = 0
for a in range(-999, 1000):
    for b in range(-999, 1000):
        k = 0
        tmp = k**2 + a * k + b 
        print(a, b, tmp)
        while tmp > 1 and isprime(tmp) :
            k += 1
            tmp = k**2 + a * k + b
        if k > max_:
            max_ = k
            result = (a,b)
print(max_, result, result[0] * result[1])
```

## [Number spiral diagonals](https://projecteuler.net/problem=28)

### 题目描述

Starting with the number 1 and moving to the right in a clockwise direction a 5 by 5 spiral is formed as follows:

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210126131016.png)

It can be verified that the sum of the numbers on the diagonals is 101.

What is the sum of the numbers on the diagonals in a 1001 by 1001 spiral formed in the same way?

### 题解

第一层 1个数

第二层 8个数      = $3^2-1^2 = 2\times4$   

第三层 16个数    = $5^2 - 3^2 = 2\times8$

第四层 24个数       = $7^2 - 5^2 = 2\times12$

即第n层  = $2 *  (2n-1 + 2n-3) = 2\times4(n-1) = 8(n - 1)$

对角线的位置为

| 43   | 44   | 45   | 46   | 47   | 48   | 49   |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| 42   | 21   | 22   | 23   | 24   | 25   | 26   |
| 41   | 20   | 7    | 8    | 9    | 10   | 27   |
| 40   | 19   | 6    | 1    | 2    | 11   | 28   |
| 39   | 18   | 5    | 4    | 3    | 12   | 29   |
| 38   | 17   | 16   | 15   | 14   | 13   | 30   |
| 37   | 36   | 35   | 34   | 33   | 32   | 31   |

找规律的题目，第二个函数是自己写的

每一圈对角线上的数求和 

```python
def spiral_diag_sum(n):
    if n < 1: return None
    elif n == 1: return 1
    elif n % 2 == 0: return None
    else:
        numbers = [1]
        while len(numbers) < (2*n - 1):
            increment = int(len(numbers) * 0.5 + 1.5)
            for p in range(4):
                numbers.append(numbers[-1] + increment)
             
    return sum(numbers)

def diag_sum(n):
    if n < 1: return None
    elif n == 1: return 1
    elif n % 2 == 0: return None
    else:

        init_ = [3, 5, 7, 9]
        sum_ = sum(init_) + 1
        for i in range(n // 2 - 1):
            for j in range(len(init_)):
                init_[j] = init_[(j - 1) % 4]  + 2 * (i + 2 )
            sum_ += sum(init_)
        return sum_

        
print(diag_sum(1001))
print(spiral_diag_sum(1001))  

```

## [Distinct powers](https://projecteuler.net/problem=29)

### 题目描述

Consider all integer combinations of *a**b* for 2 ≤ *a* ≤ 5 and 2 ≤ *b* ≤ 5:

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210127150719.png)

If they are then placed in numerical order, with any repeats removed, we get the following sequence of 15 distinct terms:

4, 8, 9, 16, 25, 27, 32, 64, 81, 125, 243, 256, 625, 1024, 3125

How many distinct terms are in the sequence generated by *a**b* for 2 ≤ *a* ≤ 100 and 2 ≤ *b* ≤ 100?

### 题解

可能这就是Python吧

```python
result = set()
for a in range(2, 101):
    for b in range(2, 101):
        result.add(a**b)
print(len(result))
```

## [Digit fifth powers](https://projecteuler.net/problem=30)

### 题目描述

Surprisingly there are only three numbers that can be written as the sum of fourth powers of their digits:

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210127150744.png)

As 1 = 14 is not a sum it is not included.

The sum of these numbers is 1634 + 8208 + 9474 = 19316.

Find the sum of all the numbers that can be written as the sum of fifth powers of their digits.

### 题解

可能这就是Python吧

```python
def get_5_powers_sum(n):
    tmp = str(n)
    result = 0
    for i in tmp:
        result += int(i)**5
    return result

ans = 0
for i in range(2, 1000000):
    if i == get_5_powers_sum(i):
        ans += i
        print(i)

print('ans:', ans)
```

