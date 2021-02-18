---
author: "xingyu"
author_link: ""
title: "Python_OpenCV"
date: 2021-02-12T18:28:10+08:00
lastmod: 2021-02-12T18:28:10+08:00
draft: false
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["cv", "数字图像处理"]
categories: ["数字图像处理"]

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

### img

```python
import cv2

img = cv2.imread('Path')
cv2.imshow('Cat', img)

cv2.waitKey(0)  #等待特定延迟
```

### video

```python
capture = cv2.VideoCapture('Path')
# 摄像头 capture = cv2.VideoCapture(0) 第1个摄像头
while True:
	isTrue, frame = capture.read()
	cv2.imshow('Video', frame)
    if cv.waitKey(20) & 0xFF == ord('d'):
        break
capture.relase()
cv2.destoryAllWindows()
```

### resize & rescale

#### method 1  适用于图像和视频

```python
import cv2 

def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

capture = cv2.VideoCapture('dde-introduction.mp4')

while True:
    isTrue, frame = capture.read()

    fream_resized = rescaleFrame(frame)
    cv2.imshow('Video', frame)
    cv2.imshow('Video Resized', fream_resized)
    if cv2.waitKey(20) & 0xFF == ord('d'):
        break
       
capture.release()
cv2.destroyAllWindows()
```

#### method 2 仅使用与视频

```python
def changeRes(width, height):  #更改视频图像分辨率
    capture.set(3, width)
    capture.set(4, height)
```

### Drawing Shapes & Putting Text

```python
# 1. Paing the Image a certain colour
blank[200:300, 100:200] = 0,255,255
cv2.imshow('Green', blank)
# 2. Draw a Rectangle
cv2.rectangle(blank, (0,0), (250,250), (0,255,0), thickness=cv2.FILLED)
cv2.imshow('Rectangle', blank)
# 3. Draw a circle 
cv2.circle(blank, (40,40), 40, (0,0,255), thickness=3)
cv2.imshow('Circle', blank)
# 4. Draw a line
cv2.line(blank, (0,0), (40, 40), (255,0,0), thickness=3)
cv2.imshow('line', blank)
# 5. Write text
cv2.putText(blank, "Hello, World!", (255,255), cv2.FONT_HERSHEY_TRIPLEX, 1.0, (0,255,255), 2)
cv2.imshow('text', blank)
```

### Essential Functions

```python
import cv2

img = cv2.imread('1.jpg')
cv2.imshow('BGR', img)

# Converting to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

# Blur -> 轻微的模糊，减少噪音
blur = cv2.GaussianBlur(img, (7,7), cv2.BORDER_DEFAULT)
cv2.imshow('Blur', blur)

#  Edge Cascade -> 应用模糊， 消除无用边缘
canny = cv2.Canny(blur, 125, 175)
cv2.imshow('Canny Edge', canny)

# Dilating the image  -> 扩展图像, 边缘增强
dilated = cv2.dilate(canny, (7,7), iterations=3)
cv2.imshow('Dilated', dilated)

# Frading # 图像腐蚀 消除Dialte
eroded = cv2.erode(dilated, (7,7), iterations=3)
cv2.imshow('Eroded', eroded)

# Resize
resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_CUBIC)
cv2.imshow('Resized', resized)

# Cropping
cropped = img[50:200, 200:400]
cv2.imshow('Croped', cropped)


cv2.waitKey(0)

```

### Image Transformations

```python
import cv2
import numpy as np

img = cv2.imread('1.jpg')
cv2. imshow('Img', img)

# Translation 平移 
def translate(img, x, y):
    transMat = np.float32([[1,0,x], [0,1,y]])
    dimensions = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, transMat, dimensions)

translated = translate(img, -100, -100)
cv2.imshow('Translated', translated)

# Rotation 旋转
def rotate(img, angle, rotPoint=None):
    (height, width) = img.shape[:2]
    if rotPoint is None:
        rotPoint = (width // 2, height // 2)
    rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)
    return cv2.warpAffine(img, rotMat, dimensions)
rotated = rotate(img, -45)
cv2.imshow('Rotated', rotated)

# Resizing
resized = cv2.resize(img, (500,500), interpolation=cv2.INTER_CUBIC)
cv2.imshow('Resized', resized)

# Flipping 翻转
flip = cv2.flip(img, 1) # 0: 垂直翻转 1:水平翻转 -1:垂直和水平翻转
cv2.imshow('Flip', flip)

# Cropping
cropped = img[200:400, 300:400]
cv2.imshow('Cropped', cropped)

cv2.waitKey(0)
```

### Contour Detection

```python
import cv2
import numpy as np

img = cv2.imread('1.jpg')

cv2.imshow('Img', img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

blur = cv2.GaussianBlur(gray, (5,5), cv2.BORDER_DEFAULT)
cv2.imshow('Blur',  blur)

# method1: cv2.Canny
canny = cv2.Canny(blur, 125, 175)
cv2.imshow('Canny', canny)
contours, hierarchies = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(len(contours))

cv2.waitKey(0)
```

```python
# method2: 二值化图像
blank = np.zeros(img.shape, dtype='uint8')
cv2.imshow('Blank', blank)

ret, thresh = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)
cv2.imshow('Thresh', thresh)

contours, hierarchies = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
print(len(contours))

cv2.drawContours(blank, contours, -1, (0,255,255), thickness=1)
cv2.imshow("Contours Drawn", blank)
```

## advanced

### Color Spaces

```python
# BGR to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray', gray)

# BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV', hsv)

# BGR to lab
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
cv2.imshow('LAB', lab)

# BGR to RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow('RGB', rgb)

cv2.imshow('Img', img)  # bgr
plt.imshow(img)   # rgb
plt.show()
```



