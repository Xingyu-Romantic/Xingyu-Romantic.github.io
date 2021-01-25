---
author: "xingyu"
author_link: ""
title: "LightGBM_Learning"
date: 2021-01-25T23:20:56+08:00
lastmod: 2021-01-25T23:20:56+08:00
draft: false
description: ""
show_in_homepage: true
description_as_summary: false
license: ""

tags: ["机器学习"]
categories: ["机器学习"]

featured_image: ""
featured_image_preview: ""

comment: true
toc: true
auto_collapse_toc: true
math: true
---

观看阿水大神的直播，记录相关学习内容

LightGBM

<!--more-->

##　LightGBM 原理

LightGBM 由微软提出，主要解决 GDBT 在海量数据中遇到的问题，可以更好更快地用于工业实践中。

 ### LightGBM的贡献

* 单边梯度抽样算法；
* 直方图算法；
* 互斥特征捆绑算法；
* 深度限制的 Leaf-wise 算法；
* 类别特征最优分割；
* 特征并行和数据并行；
* 缓存优化；

##  LightGBM超参数解析

https://lightgbm.readthedocs.io/en/latest/Parameters.html

![](https://blog-1254266736.cos.ap-nanjing.myqcloud.com/img/20210125232425.png)

```python
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import GridSearchCV # 进行交叉验证
def auc2(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict(train)),
                            metrics.roc_auc_score(y_test,m.predict(test)))

lg = lgb.LGBMClassifier(silent=False)  #sklearn接口
param_dist = {"max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [200]
             }
grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 3, scoring="roc_auc", verbose=5)
grid_search.fit(train,y_train)
grid_search.best_estimator_


d_train = lgb.Dataset(train, label=y_train, free_raw_data=False) #原生接口
params = {"max_depth": 3, "learning_rate" : 0.1, "num_leaves": 900,  "n_estimators": 20}

## 以下训练两个模型，
# Without Categorical Features
model2 = lgb.train(params, d_train)
print(auc2(model2, train, test))

#With Catgeorical Features
cate_features_name = ["MONTH","DAY","DAY_OF_WEEK","AIRLINE","DESTINATION_AIRPORT",
                 "ORIGIN_AIRPORT"]
model2 = lgb.train(params, d_train, categorical_feature = cate_features_name)
print(auc2(model2, train, test))
```

### 训练保存模型

```python
# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# generate feature names
feature_name = ['feature_' + str(col) for col in range(num_feature)]

print('Starting training...')
# feature_name and categorical_feature
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                valid_sets=lgb_eval,  # eval training data
                feature_name=feature_name,
                categorical_feature=[21])

print('Finished first 10 rounds...')
# check feature name
print('7th feature name is:', lgb_train.feature_name[6])

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Dumping model to JSON...')
# dump model to JSON (and save to file)
model_json = gbm.dump_model()

with open('model.json', 'w+') as f:
    json.dump(model_json, f, indent=4)
```

### 特征重要性计算

```python
# feature names
print('Feature names:', gbm.feature_name())

# feature importances
print('Feature importances:', list(gbm.feature_importance()))
```

### 加载模型

```python
print('Loading model to predict...')
# load model to predict
bst = lgb.Booster(model_file='model.txt')

# can only predict with the best iteration (or the saving iteration)
y_pred = bst.predict(X_test)

# eval with loaded model
print("The rmse of loaded model's prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)

print('Dumping and loading model with pickle...')
# dump model with pickle
with open('model.pkl', 'wb') as fout:
    pickle.dump(gbm, fout)
# load model with pickle to predict
with open('model.pkl', 'rb') as fin:
    pkl_bst = pickle.load(fin)
# can predict with any iteration when loaded in pickle way
y_pred = pkl_bst.predict(X_test, num_iteration=7)
# eval with loaded model
print("The rmse of pickled model's prediction is:", mean_squared_error(y_test, y_pred) ** 0.5)
```

### 继续训练

```python
# continue training
# init_model accepts:
# 1. model file name
# 2. Booster()
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model='model.txt',
                valid_sets=lgb_eval)

print('Finished 10 - 20 rounds with model file...')
```

### 修改超参数

```python
# decay learning rates
# learning_rates accepts:
# 1. list/tuple with length = num_boost_round
# 2. function(curr_iter)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                learning_rates=lambda iter: 0.05 * (0.99 ** iter),
                valid_sets=lgb_eval)

print('Finished 20 - 30 rounds with decay learning rates...')

# change other parameters during training
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                valid_sets=lgb_eval,
                callbacks=[lgb.reset_parameter(bagging_fraction=[0.7] * 5 + [0.6] * 5)])

print('Finished 30 - 40 rounds with changing bagging_fraction...')
```

### 自定义损失函数

```python
# self-defined objective function
# f(preds: array, train_data: Dataset) -> grad: array, hess: array
# log likelihood loss
def loglikelihood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess


# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
# binary error
# NOTE: when you do customized loss function, the default prediction value is margin
# This may make built-in evalution metric calculate wrong results
# For example, we are doing log likelihood loss, the prediction is score before logistic transformation
# Keep this in mind when you use the customization
def binary_error(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    return 'error', np.mean(labels != (preds > 0.5)), False


gbm = lgb.train(params,
                lgb_train,
                num_boost_round=10,
                init_model=gbm,
                fobj=loglikelihood,
                feval=binary_error,
                valid_sets=lgb_eval)

print('Finished 40 - 50 rounds with self-defined objective function and eval metric...')
```

## 模型调参方法

### For Faster Speed

* 设置`num_threads`
* 用GPU版本的LightGBM
* 减小`max_depth` 和 `num_leaves`
* 增加`min_data_in_leaf `和` min_sum_hessian_in_leaf`
* 减小 `num_iterations`
* 使用`Early Stopping`
* 减小 `max_bin` 和 `max_bin_by_feature`， 增加 `min_data_in_bin`
* 增加采样， 特征采样`feature_fraction`， 样本采样`bagging_fraction`

### For Better Accuracy

* Use large `max_bin` (may be slower)
* Use small learning_rate with large `num_iterations`
* Use large `num_leaves` (may cause over-fitting)
* Use bigger `training` data
* Try `dart`

### Deal with over-fitting

* Use small `max_bin`
* Use small `num_leaves`
* Use `min_data_in_leaf` and `min_sum_hessian_in_leaf`
* Use bagging by set `bagging_fraction` and `bagging_freq`
* Use feature sub-sampling by set `feature_fraction`
* Use bigger training data
* Try `lambda_l1`, `lambda_l2` and `min_gain_to_split` for regularization
* Try `max_depth` to avoid growing deep tree
* Try `extra_trees`
* Try increasing `path_smooth`



人工调参效果会比自动调参好一点。

```python
d_train = lgb.Dataset(train, label=y_train)
params = {"max_depth": 4, "learning_rate" : 0.05, "num_leaves": 250, 'n_estimators': 600}

data = lgb.cv(params, d_train, num_boost_round=350, nfold=5, metrics='auc')
print(pd.DataFrame(data))
```

### 自动调参

```python
lg = lgb.LGBMClassifier(silent=False)
param_dist = {"max_depth": [4,5, 7],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [50, 100, 150]
             }
grid_search = GridSearchCV(lg, n_jobs=-1, param_grid=param_dist, cv = 5, scoring="roc_auc", verbose=5)
## n_jobs 线程个数  cv 每次分组个数
grid_search.fit(train,y_train)
grid_search.best_estimator_, grid_search.best_score_
```

### BayesianOptimization

https://github.com/fmfn/BayesianOptimization

```python
import warnings
import time
warnings.filterwarnings("ignore")
from bayes_opt import BayesianOptimization

def lgb_eval(max_depth, learning_rate, num_leaves, n_estimators):
    params = {
             "metric" : 'auc'
        }
    params['max_depth'] = int(max(max_depth, 1))
    params['learning_rate'] = np.clip(0, 1, learning_rate)
    params['num_leaves'] = int(max(num_leaves, 1))
    params['n_estimators'] = int(max(n_estimators, 1))
    cv_result = lgb.cv(params, d_train, nfold=5, seed=0, verbose_eval =200,stratified=False)
    return 1.0 * np.array(cv_result['auc-mean']).max()


lgbBO = BayesianOptimization(lgb_eval, {'max_depth': (4, 8),
                                            'learning_rate': (0.05, 0.2),
                                            'num_leaves' : (20,1500),
                                            'n_estimators': (5, 200)}, random_state=0)

lgbBO.maximize(init_points=5, n_iter=50,acq='ei')
print(lgbBO.max)
```

## LightGBM 模型部署

### Python 和 Spark环境下直接封装为HTTP接口

```python
import tornado.ioloop
import tornado.web
import json
import joblib

import numpy as np

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, 666")
        
    def post(self):
        data = self.request.body.decode('utf-8')
        data = json.loads(data)
        
        data = np.array(data['data']).reshape(1, 4)
        predict_lbl = clf.predict(data)[0]
        
        msg = {
            'label' : int(predict_lbl),
            'code' : 200,
        }
        self.write(json.dumps(msg))

application = tornado.web.Application([
    (r"/", MainHandler),
])


if __name__ == "__main__":
    clf = joblib.load('data1.pkl')
    
    application.listen(9999)
    tornado.ioloop.IOLoop.instance().start()
```

```python
import urllib, requests, json
post_data = { 'data': [5.9, 3. , 5.1, 1.8] } 
requests.post("http://192.168.0.106:9999", data = json.dumps(post_data)).text
```



1. 使用TreeLite打包为so文件

....