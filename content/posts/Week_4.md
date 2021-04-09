---
author: "xingyu"
author_link: ""
title: "Week_4"
date: 2021-04-02T09:46:32+08:00
lastmod: 2021-04-02T09:46:32+08:00
draft: true
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["每周总结"]
categories: ["每周总结"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

4.2 - 

<!--more-->

## 4.2

**LeetCode:**

* 每日打卡： [面试题 17.21. 直方图的水量](https://leetcode-cn.com/problems/volume-of-histogram-lcci/)  知识点：单调栈

**竞赛：**

* 调试行人识别代码、
* Sky Hacthon 

## 4.7

**LeetCode:**

* 每日打卡 [81. 搜索旋转排序数组 II](https://leetcode-cn.com/problems/search-in-rotated-sorted-array-ii/)

* 蓝桥杯竞赛模拟题刷题   滑动窗口最大值，利用双端队列

  ```python
  class Solution:
      def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
          result = []
          slow = 0
          deq = collections.deque()
          for fast in range(len(nums)):
              while deq and nums[fast] >= nums[deq[-1]]:
                  deq.pop()
              deq.append(fast)
              if deq[0] <= fast - k:                   
                  deq.popleft()
              if fast + 1 >= k:
                  result.append(nums[deq[0]])
          return result
  ```

**Englih：**

* 百词斩单词打卡
* 墨墨单词打卡

