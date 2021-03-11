---
author: "xingyu"
author_link: ""
title: "LHY_ML_HW1_Regression"
date: 2021-03-11T21:50:13+08:00
lastmod: 2021-03-11T21:50:13+08:00
draft: false
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["机器学习", "Pytorch"]
categories: ["机器学习"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

李宏毅机器学习 2021 

HW1  Regression

之前都是用Pytorch 训练图片数据， 这次作业是一个直接的csv数据，记录一下相关的操作

<!--more-->

## 读入数据

```python
train_data = pd.read_csv('./ml2021spring-hw1/covid.train.csv')
test_data = pd.read_csv('./ml2021spring-hw1/covid.test.csv')
train_data.shape, test_data.shape
# ((2700, 95), (893, 94))
```

### 数据维度说明

>States (40, encoded to one-hot vectors)
>
>○ e.g. AL, AK, AZ, ...
>
>● COVID-like illness (4)
>
>○ e.g. cli,ili (influenza-like illness), ...
>
>● Behavior Indicators (8)
>
>○ e.g. wearing_mask, travel_outside_state, ...
>
>● Mental Health Indicators (5)
>
>○ e.g. anxious, depressed, ...
>
>● Tested Positive Cases (1)
>
>○ tested_positive (this is what we want to predict)

## 定义网络

```python
class Net(LightningModule):
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = torch.mean((logits - y) ** 2)
        self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-2)
    def __init__(self, n_feature):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feature, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        #self.fc3 = nn.Linear(64, n_output)
        self.criterion = nn.MSELoss(reduction='mean')
    def forward(self, x):
        #batch_size, channels, width, height = x.size()
        # (b, 1, 28, 28) -> (b, 1*28*28)
        #x = x.view(batch_size, -1)
        return self.net(x).squeeze(1)
```

**踩坑**

也是自己蠢了， 直接套用了之前进行十六进制数字识别的网络结构，最后一层是sigmoid，而这次是一个回归问题，最后是一个数值，而不是分类问题。导致Loss 一直在60左右，不降，没想到是这里出问题了。。。

## DataLoader

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
        X = torch.tensor(np.array(self.df.iloc[index, :94]), dtype=torch.float)
        y = torch.tensor(np.array(self.df.iloc[index,94]), dtype=torch.float)
        return X, y
	# 返回数据集总长度
    def __len__(self):
        return self.df.shape[0]

```

```python
train_data = MySet(train_data)
train = DataLoader(train_data, batch_size=128, shuffle=True)
```

## 训练

```python
#optimizer = Adam(LitMNIST(input_num, hidden_num, output_num).parameters(), lr=1e-2)
model = Net(94)
trainer = Trainer(gpus=0, max_epochs = 500)
trainer.fit(model, train)
```

## 预测

```python
X = torch.tensor(np.array(test_data.iloc[:, :94]), dtype=torch.float)
result = model(X).detach().numpy().reshape(-1)
sub = pd.DataFrame(columns=['id', 'tested_positive'])
sub['id'] = list(range(test_data.shape[0]))
sub['tested_positive'] = result
sub.to_csv('result.csv', index=None)
```

## 结果

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210311220054.png)

第一名0.86536 还有很大提升空间



