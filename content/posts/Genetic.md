---
title: "Genetic"
date: 2020-11-06T17:25:24+08:00
description: ""
---

# 遗传算法

通过编码、设置种群、设置适应度函数、遗传操作、解码产生需要的解。f(x)=x*sin(x)+1，x[0,2],求f(x)的最大值、最小值。

![](https://xingyublog.netlify.app/static/Gen.png)



## 定义函数 （即 适应度函数）


```python
import math
import numpy as np
import matplotlib.pyplot as plt
import random
```


```python
def f(x):
    return x * np.sin(x) + 1 
```

### 绘制函数图像


```python
plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'green', 'ytick.color':'green', 'figure.facecolor':'white'})
x = np.arange(0,2,0.1)
plt.plot(x,f(x))
plt.show()
```


![png](https://xingyublog.netlify.app/static/output_7_0.png)


## 随机产生种群


```python
def product_origin(population_size, length):
    """
    :population_size: 种群大小,
    :length: DNA长度
    """
    population = np.ones((population_size, length))
    #二维列表，包含染色体和基因
    for i in range(population_size):
        temporary=[]
        #染色体暂存器
        for j in range(length):
            population[i][j]= random.randint(0,1)
            #随机产生一个染色体,由二进制数组成
            #将染色体添加到种群中
    return population
            # 将种群返回，种群是个二维数组，个体和染色体两维
```

## 二进制转十进制


```python
"""
def translation(num):
    return int(str(num),2)
"""
def translation(population,chromosome_length):
    temporary=[]
    for i in range(len(population)):
        total=0
        for j in range(chromosome_length):
            total+=population[i][j]*(math.pow(2,j-4))
        temporary.append(total)
    return temporary

```

## 选择



```python
def cumsum(fitness1):
    for i in range(len(fitness1)-2,-1,-1):
    # range(start,stop,[step])
    # 倒计数
        total=0
        j=0
        while(j<=i):
            total+=fitness1[j]
            j+=1
        fitness1[i]=total
        fitness1[len(fitness1)-1]=1
```


```python
def selection(population,fitness):
        new_fitness=[]
        total = np.sum(fitness)
        for i in range(len(fitness)):
            new_fitness.append(fitness[i]/ total )
        cumsum(new_fitness)
        ms=[]
    #存活的种群
        population_length=pop_len=len(population)
    #求出种群长度
    #根据随机数确定哪几个能存活
        for i in range(pop_len):
            ms.append(random.random())
    # 产生种群个数的随机值
    # ms.sort()
    # 存活的种群排序
        fitin=0
        newin=0
        new_population=new_pop=population
     #轮盘赌方式
        while newin<pop_len:
            if(ms[newin]<new_fitness[fitin]):
                    new_pop[newin]=population[fitin]
                    newin+=1
            else:
                 fitin+=1
        population=new_pop
    
```

## 交叉



```python
def crossover(population, pc):
#pc是概率阈值，选择单点交叉还是多点交叉，生成新的交叉个体，这里没用
    pop_len=len(population)
    for i in range(pop_len-1):
        if(random.random()< pc):
            cpoint=random.randint(0,len(population[0]))
           #在种群个数内随机生成单点交叉点
            temporary1=[]
            temporary2=[]
            temporary1.extend(population[i][0:cpoint])
            temporary1.extend(population[i+1][cpoint:len(population[i])])
           #将tmporary1作为暂存器，暂时存放第i个染色体中的前0到cpoint个基因，
           #然后再把第i+1个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
            temporary2.extend(population[i+1][0:cpoint])
            temporary2.extend(population[i][cpoint:len(population[i])])
        # 将tmporary2作为暂存器，暂时存放第i+1个染色体中的前0到cpoint个基因，
        # 然后再把第i个染色体中的后cpoint到第i个染色体中的基因个数，补充到temporary2后面
            population[i]=temporary1
            population[i+1]=temporary2
        # 第i个染色体和第i+1个染色体基因重组/交叉完成
```

## 变异


```python
def mutation(population, pm):
     # pm是概率阈值
    px=len(population)
    # 求出种群中所有种群/个体的个数
    py=len(population[0])
    # 染色体/个体基因的个数
    for i in range(px):
        if(random.random()<pm):
            mpoint=random.randint(0,py-1)
            #
            if(population[i][mpoint]==1):
               #将mpoint个基因进行单点随机变异，变为0或者1
                population[i][mpoint]=0
            else:
                population[i][mpoint]=1
 
```

## 挑选最好


```python
def best(population, x_10, fitness):
    maxidx = np.argmax(fitness)
    return x_10[maxidx], fitness[maxidx]
```

## 绘图


```python
def plot(results, iteration):
    plt.plot(np.arange(1,iteration+1), [i[1] for i in results])
    plt.show()
```

## 遗传算法


```python
population_size = 100  #种群大小
length = 5 # DNA长度
iteration = 200 # 迭代次数
pm = 0.01 # 变异概率
pc = 0.6
```


```python

# 1 . 初始化种群
population = product_origin(population_size, length)
results = []
for i in range(iteration):
    x_10 = translation(population, length)
    fitness = f(x_10)
    best_individual, best_fitness = best(population,x_10, fitness)
    results.append([best_fitness, best_individual])
    selection(population, fitness)
    crossover(population, pc)
    mutation(population,pm)

results.sort()

plot(results,iteration)
```


![png](https://xingyublog.netlify.app/static/output_25_0.png)



```python

```


```python

```

