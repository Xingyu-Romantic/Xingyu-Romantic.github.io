---
author: "xingyu"
author_link: ""
title: "Hgame_accuracy_writeup"
date: 2021-02-15T23:19:11+08:00
lastmod: 2021-02-15T23:19:11+08:00
draft: false
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["CTF", "Pytorch"]
categories: ["CTF"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

把相关的代码记录一下， 方便以后使用

<!--more-->

## accuracy

石碑上的文字,究竟隐藏着怎样的秘密……

### Write Up

题目给出两个文件， chars.csv    dataset.csv， 

简单查看之后，dataset是数据集，  chars是测试集

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210215232337.png)

数据集是由784个pixel， 即 28 * 28

```python
df.iloc[:,0].unique()
#array([10, 11, 12, 13, 14, 15,  1,  0,  4,  7,  3,  5,  8,  9,  2,  6])
```

查看标签，共有16个， 结合数据集比对，应该是一串十六进制的数

接下来就是训练 + 预测

```python
import torch
import pytorch_lightning as pl
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F
from PIL import Image, ImageChops, ImageFilter, ImageEnhance
import os
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer
from torch.optim import Adam
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import cv2
import numpy as np
```
#### 网络

```python
class LitMNIST(LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)
    def __init__(self, input_num, hidden_num, output_num):
        super().__init__()
        self.fc1 = nn.Linear(input_num, hidden_num)
        self.fc2 = nn.Linear(hidden_num, output_num)
        self.relu = nn.ReLU()
    def forward(self, x):
        #batch_size, channels, width, height = x.size()
        # (b, 1, 28, 28) -> (b, 1*28*28)
        #x = x.view(batch_size, -1)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu(x)
        y = self.fc2(x)
        return F.log_softmax(y, dim=1)
```

#### 加载数据

```python
transform=transforms.Compose([   
                                 transforms.ToTensor()
                            ])
class MySet(Dataset):
	# 读取数据
    def __init__(self, df):
        self.df = df
	# 根据索引返回数据
    def __getitem__(self, index):
        image = Image.fromarray(np.uint8(self.df.iloc[index, 1:]).reshape(28,28))
        label = self.df.iloc[index,0]
        return transform(image),label
	# 返回数据集总长度
    def __len__(self):
        return self.df.shape[0]
```

```python
train_data = MySet(df)
train = DataLoader(train_data, batch_size=64, shuffle=True)
```

#### 训练

```python
input_num = 784
hidden_num = 500
output_num = 16
#optimizer = Adam(LitMNIST(input_num, hidden_num, output_num).parameters(), lr=1e-2)
model = LitMNIST(input_num, hidden_num, output_num)
trainer = Trainer(gpus=0, max_epochs = )
trainer.fit(model, train)
```

#### 预测

```python
import numpy as np
ceshi = Image.open('./chars/87.png').convert('L')
ceshi = ImageChops.invert(ceshi)
#ceshi = Image.fromarray(np.uint8(df.iloc[1150, 1:]).reshape(28,28))
img = transform(ceshi)
img = img.unsqueeze(0)  #图片扩展多一维,因为输入到保存的模型中是4维的[batch_size,通道,长，宽]，而普通图片只有三维，[通道,长，宽]
    #扩展后，为[1，1，28，28]
print(img.shape)
output = model(img).detach().numpy()
pred = np.argmax(output) #选出概率最大的一个
print(pred.item())
```

```python
file_dir = './chars/'
filenames = os.listdir(file_dir)
filenames = sorted(filenames, key = lambda x: int(x.strip('.png')))
dict_ = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9, 10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F'}
def predict(filename):
    ceshi = Image.open(file_dir + filename).convert('L')
    ceshi = ImageChops.invert(ceshi)
    img = transform(ceshi)
    img = img.unsqueeze(0)
    output = model(img).detach().numpy()
    pred = np.argmax(output) #选出概率最大的一个
    return dict_[pred.item()]
result = ''
for filename in filenames:
    result += str(predict(filename))
print(result)
```

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210215232805.png)

传入网站，得到flag

```flag
hgame{deep_learn1ng^and&AI*1s$amazing#r1ght?}
```



