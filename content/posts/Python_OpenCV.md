---
author: "xingyu"
author_link: ""
title: "Python_OpenCV"
date: 2021-02-12T18:28:10+08:00
lastmod: 2021-02-12T18:28:10+08:00
draft: true
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: []
categories: []

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

记录一下相关的API， 方便查找

<!--more-->

## basic

### imshow

#### img

```python
import cv2

img = cv2.imread('Path')
cv2.imshow('Cat', img)

cv2.waitKey(0)  #等待特定延迟
```

#### video

```python
capture = cv2.VideoCapture('Path')
while True:
	isTrue, frame = capture.read()
	cv2.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.relase()
cv2.destoryAllWindows()
```



### resize

