---
author: "xingyu"
author_link: ""
title: "DeepSORT_源码解读_01"
date: 2021-03-20T13:43:25+08:00
lastmod: 2021-03-20T13:43:25+08:00
draft: false
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["目标检测", "计算机视觉", "Pytorch", "深度学习"]
categories: ["深度学习"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

涉及到相关知识, 需要对代码进行修改.

<!--more-->

## track.py

### 导入模型 输入参数

```
model = torch.load(weights, map_location=device)[
        'model'].float()  # load to FP32
# weights = '.weights/yolov5s.pt'

```

**FP32  FP16  BF16**

FP32 是单精度浮点数，用8bit 表示指数，23bit 表示小数；

FP16半精度浮点数，用5bit 表示指数，10bit 表示小数；

BF16是对FP32单精度浮点数截断数据，即用8bit 表示指数，7bit 表示小数。

![img](https://pic4.zhimg.com/v2-fdb82e0762b0d6af8d9890d29bb0d05f_r.jpg)

### 加载数据

```python
vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        view_img = True
        save_img = True
        dataset = LoadImages(source, img_size=imgsz)
```

采用 yolov5 中的 `LOadImages()`

```python
class LoadImages:  # for inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if x.split('.')[-1].lower() in img_formats]
        videos = [x for x in files if x.split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])  # new video
        else:
            self.cap = None
        assert self.nf > 0, f'No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret_val, img0 = self.cap.read()
            if not ret_val:
                self.count += 1
                self.cap.release()
                if self.count == self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end='')

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path)  # BGR
            assert img0 is not None, 'Image Not Found ' + path
            print(f'image {self.count}/{self.nf} {path}: ', end='')

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf  # number of files
```

### 逐帧识别

将每一帧图片放到`gpu`上

```python
for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
```

>Tensor.ndimension()，返回tensor的维度（整数）

```python
pred = model(img, augment=opt.augment)[0]
# Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
```

>关于 NMS 在这里留个坑  等以后再来填上
>
>NMS算法一般是为了去掉模型预测后的多余框，其一般设有一个nms_threshold=0.5

pred[..., 0:4]为预测框坐标        

预测框坐标为xywh(中心点+宽长)格式        

pred[..., 4]为objectness置信度        

pred[..., 5:-1]为分类结果

处理每一个监测框

```python
for i, det in enumerate(pred):  # detections per image
    if webcam:  # batch_size >= 1
        p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
    else:
        p, s, im0 = path, '', im0s

    s += '%gx%g ' % img.shape[2:]  # print string
    save_path = str(Path(out) / Path(p).name)
```

```python
if det is not None and len(det):
    # Rescale boxes from img_size to im0 size
    det[:, :4] = scale_coords(
        img.shape[2:], det[:, :4], im0.shape).round()
    # Print results
    for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        s += '%g %ss, ' % (n, names[int(c)])  # add to string

    bbox_xywh = []
    confs = []
    
    # Adapt detections to deep sort input format
    for *xyxy, conf, cls in det:
        x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
        obj = [x_c, y_c, bbox_w, bbox_h]
        bbox_xywh.append(obj)
        confs.append([conf.item()])

    xywhs = torch.Tensor(bbox_xywh)
    confss = torch.Tensor(confs)

    # Pass detections to deepsort
    outputs = deepsort.update(xywhs, confess, im0)
```

调整尺寸, 输出预测结果

`outputs = deepsort.update(xywhs, confess, im0)`

```python
if len(outputs) > 0:
	bbox_xyxy = outputs[:, :4]
	identities = outputs[:, -1]
	draw_boxes(im0, bbox_xyxy, identities)
	# 绘制路径 By involute 
	draw_trace(im0, bbox_xyxy, identities)
	# End
```

```python
def draw_boxes(img, bbox, identities=None, offset=(0, 0)):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(
            img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 +
                                 t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
    return img

def draw_trace(img, bbox, identities=None):
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(id)
        if id in trace.keys():
            trace[id].append((x, y))
            temp = trace[id]
            for i in range(1, len(temp)):
                cv2.line(img, (temp[i-1][0], temp[i-1][1]), (temp[i][0], temp[i][1]), color, 3)
                if i > 20:
                    temp.pop(0)
        else:
            trace[id] = [(x, y)]
```

`draw_bbox` 函数简单, 不在此过多赘述.

`draw_trace` 自己添加, 把运动轨迹描绘出来.

* 采用 trace 字典 记录每个 id 的 运动坐标, 然后 绘制.



