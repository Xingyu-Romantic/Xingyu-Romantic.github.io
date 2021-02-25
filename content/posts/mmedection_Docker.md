---
author: "xingyu"
author_link: ""
title: "关于天池比赛，冠军方案复现"
date: 2021-02-22T14:25:19+08:00
lastmod: 2021-02-22T14:25:19+08:00
draft: false
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["Pytorch", "数字图像处理"]
categories: ["数字图像处理"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

继上一篇博客， 寻找到相关比赛冠军方案。

构建docker 运行。

文章地址：https://tianchi.aliyun.com/notebook-ai/detail?postId=74264

mmedection gitee地址：https://gitee.com/Xingyu-Romantic/mmdetection.git

github 太慢了。。:)

<!--more-->

# mmedection

由于采用的是 mmedection， 

### 换源

```shell
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["https://adfx5pa6.mirror.aliyuncs.com"]
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

通过dockerfile构建镜像

```shell
docker build -t mmedetection:1.0 .
```

### Failed to fetch  。。。。

```dockerfile
RUN echo 'nameserver 223.5.5.5' > /etc/resolv.conf
RUN echo 'nameserver 223.6.6.6' > /etc/resolv.conf
```
### mmcv

关于mmcv 安装 https://www.jianshu.com/p/12a142941da3

### 最终 

一堆错误都基本解决
Dockerfile 如下

```dockerfile
ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list
RUN apt-get clean
RUN apt-get update
RUN pip config set global.index-url http://mirrors.aliyun.com/pypi/simple
RUN pip config set install.trusted-host mirrors.aliyun.com
RUN pip install -U pip
RUN echo 'nameserver 223.5.5.5' > /etc/resolv.conf
RUN echo 'nameserver 223.6.6.6' > /etc/resolv.conf
RUN apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV
RUN pip install  pytest-runner -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install mmcv
# RUN pip install mmcv-full==latest+torch1.6.0+cu101 -f https://download.openmmlab.com/mmcv/dist/index.html

# Install MMDetection
RUN conda clean --all
RUN git clone https://gitee.com/Xingyu-Romantic/mmdetection.git /mmdetection
WORKDIR /mmdetection
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
```

### 更改代码内容

json 文件只有这几个类，13个

```python
pd.read_json('anno_train.json')['defect_name'].unique()
# array(['虫粘', '色差', '沾污', '缝头', '花毛', '织疵', '缝头印', '褶子', '漏印', '错花', '网折','破洞', '水印'], dtype=object)
```

`/mmdet/datasets/coco.py`  

```python
CLASSES = ('虫粘', '色差',  '沾污', '缝头', '花毛', '织疵', '缝头印', '褶子',
    '漏印', '错花', '网折', '破洞', '水印')
```

`/mmdet/datasets/voc.py`

```python
CLASSES = ('虫粘', '色差',  '沾污', '缝头', '花毛', '织疵', '缝头印', '褶子',
    '漏印', '错花', '网折', '破洞', '水印')
```

/mmdet/core/evaluation/class_names.py`

```python
def coco_classes():
    return [
        '虫粘', '色差',  '沾污', '缝头', '花毛', '织疵', '缝头印', '褶子',
    '漏印', '错花', '网折', '破洞', '水印'
    ]
```

搭载进去训练

```shell
docker run --gpus all -v ~/Desktop/机器学习/Computer_Vision/mmdetection_new/data/coco/:/mmdetection/data/coco \
-it 7fd91b5345bf \
/bin/bash
```

测试

```shell
docker run --gpus all \
-v ~/Desktop/机器学习/Computer_Vision/mmdetection_new/data/coco/val2017/:/tcdata/guangdong1_round2_testB_20191024/201908262_de5bf60c1d3b79ad0201908262154188OK \
-it e6732164f22d /bin/bash
```

结果惨不忍睹，，不知道哪里出问题了。。。

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210225104517.png)

## 网络配置

```python
# model settings
model = dict(
    type='CascadeRCNN',
    #num_stages=4,
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        dcn=dict(   #在最后一个阶段加入可变形卷积 改进点1
            deformable_groups=1, fallback_on_stride=False, type='DCN'),
        stage_with_dcn=(False, False, False, True),
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
         anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.02, 0.05, 0.1, 0.5, 1.0, 2.0, 10.0, 20.0, 50.0], #根据样本瑕疵尺寸分布，修改anchor的长宽比。 点2
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),   #此处可替换成focalloss
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=13,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=13,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=13,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    train_cfg = dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_num=2000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    ignore_iof_thr=-1),
                sampler=dict(   #默认使用的是随机采样RandomSampler，这里替换成OHEM采样，即每个级联层引入在线难样本学习，改进点3
                    type='OHEMSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    ignore_iof_thr=-1),
                sampler=dict( 
                    type='OHEMSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='OHEMSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ],
        stage_loss_weights=[1, 0.5, 0.25]),
    test_cfg = dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_num=1000,
            nms_thr=0.7,
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05, nms=dict(type='nms', iou_thr=0.5), max_per_img=100),
        keep_all_stages=False)
    )
# model training and testing settings

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
CLASSES = ('虫粘', '色差', '沾污', '缝头', '花毛', '织疵', '缝头印', '褶子', '漏印', '错花', '网折', '破洞', '水印')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 320), keep_ratio=True), #考虑算力原有限，修改图像尺寸为半图，可修改为全图训练
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1223, 500), 
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=1,  #每张gpu训练多少张图片  batch_size = gpu_num(训练使用gpu数量) * imgs_per_gpu
    workers_per_gpu=1,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json', #修改成自己的训练集标注文件路径
        img_prefix=data_root + 'train2017', #训练图片路径
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json', #修改成自己的验证集标注文件路径
        img_prefix=data_root + 'val2017', #验证图片路径
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json', #修改成自己的验证集标注文件路径
        img_prefix=data_root + 'val2017', #验证图片路径
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0025)  #学习率的设置尤为关键：lr = 0.00125*batch_size
optimizer_config = dict(grad_clip=None)
#optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None#'./checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth' #采用coco预训练模型 ,需要对权重类别数进行处理
resume_from = None
workflow = [('train', 1)]
total_epochs = 12
'''
checkpoint_config = dict(interval=1)
total_epochs = 12  
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './work_dirs/cascade_rcnn_r50_fpn_1x' #训练的权重和日志保存路径
load_from = './checkpoints/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth' #采用coco预训练模型 ,需要对权重类别数进行处理
resume_from = None
workflow = [('train', 1)]'''
```

