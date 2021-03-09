---
author: "xingyu"
author_link: ""
title: "Bit_operation"
date: 2021-03-03T10:15:59+08:00
lastmod: 2021-03-03T10:15:59+08:00
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

<!--more-->

```
n & 1  # 取最低位， 判断奇偶数

n & (n - 1)
```

**判断奇偶数** ： x & 1 取最低位

**获取二进制位是1还是0**

**交换两个整数变量的值：** 

```python
a = a ^ b
b = a ^ b
a = a ^ b
```

异或，可以理解为不进位加法

```python
例题：找出数组中唯一重复的数字
1 2 2 3
# input : nums = [1,2,2,3]
nums.extend(set(nums))
tmp = nums[0]
for i in range(1, length):
	tmp = tmp ^ nums[i]
print(tmp)
```

**二进制中1的个数**

```python
class Solution:
    def hammingWeight(self, n: int) -> int:
        if n == 0:
            return 0
        res = 0
        while n:
            res += n & 1
            n = n >> 1
        return res
```

**k个k进制数不进位相加为0**

