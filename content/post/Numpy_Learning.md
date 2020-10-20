---
title: "Numpy_Learning"
date: 2020-10-20T18:48:08+08:00
description: ""
---

# Numpy 学习

[TOC]



## 常量

| 函数       | 含义及其备注                 |
| ---------- | ---------------------------- |
| `np.nan`   | 空值，`np.nan`!=`np.nan`     |
| `np.isnan` | 返回同样维度的 boolean array |
| `np.inf`   | 正无穷大                     |
| `np.pi`    | 圆周率                       |
| `np.e`     | 自然常数                     |

## 数据类型

### 常见数据类型

类型 | 备注 | 说明 
---|---|---
bool_ = bool8 | 8位 | 布尔类型
int8 = byte | 8位 | 整型
int16 = short |	16位| 整型
int32 = intc | 32位| 整型
int_ = int64 = long = int0 = intp | 64位| 整型
uint8 = ubyte |8位 | 无符号整型
uint16 = ushort|16位| 无符号整型
uint32 = uintc|32位| 无符号整型
uint64 = uintp = uint0 = uint| 64位| 无符号整型
float16 = half|16位 | 浮点型
float32 = single| 32位| 浮点型
float_ = float64 = double| 64位| 浮点型
str_ = unicode_ = str0 = unicode| |Unicode 字符串
datetime64| |日期时间类型
timedelta64| |表示两个时间之间的间隔

### 创建数据类型

每个内建类型都有一个唯一定义它的字符代码，如下：

字符 | 	对应类型|备注
---|---|---
b	|boolean | 'b1'
i	|signed integer | 'i1', 'i2', 'i4', 'i8'
u	|unsigned integer | 'u1', 'u2' ,'u4' ,'u8'
f	|floating-point | 'f2', 'f4', 'f8'
c	|complex floating-point |
m	|timedelta64 |表示两个时间之间的间隔
M	|datetime64 |日期时间类型
O	|object |
S	|(byte-)string | S3表示长度为3的字符串
U	|Unicode | Unicode 字符串
V	|void

```python
a = np.dtype('b1')
print(a.type)  # <class 'numpy.bool_'>
print(a.itemsize)  # 1

a = np.dtype('f2')
print(a.type)  # <class 'numpy.float16'>
print(a.itemsize)  # 2
```

## 时间日期和时间增量

### datetime64

在 numpy 中，我们很方便的将字符串转换成时间日期类型 `datetime64`（`datetime` 已被 python 包含的日期时间库所占用）。

`datatime64`是带单位的日期时间类型，其单位如下：

| 日期单位 | 代码含义 | 时间单位 | 代码含义 |
| :------: | :------: | :------: | :------: |
|    Y     |    年    |    h     |   小时   |
|    M     |    月    |    m     |   分钟   |
|    W     |    周    |    s     |    秒    |
|    D     |    天    |    ms    |   毫秒   |
|    -     |    -     |    us    |   微秒   |
|    -     |    -     |    ns    |   纳秒   |
|    -     |    -     |    ps    |   皮秒   |
|    -     |    -     |    fs    |   飞秒   |
|    -     |    -     |    as    |  阿托秒  |

注意：

- 1秒 = 1000 毫秒（milliseconds）
- 1毫秒 = 1000 微秒（microseconds）

```python
import numpy as np

a = np.datetime64('2020-03-01')
print(a, a.dtype)  # 2020-03-01 datetime64[D]

a = np.datetime64('2020-03')
print(a, a.dtype)  # 2020-03 datetime64[M]

a = np.datetime64('2020-03-08 20:00:05')
print(a, a.dtype)  # 2020-03-08T20:00:05 datetime64[s]

a = np.datetime64('2020-03-08 20:00')
print(a, a.dtype)  # 2020-03-08T20:00 datetime64[m]

a = np.datetime64('2020-03-08 20')
print(a, a.dtype)  # 2020-03-08T20 datetime64[h]
```

从字符串创建 datetime64 类型时，可以强制指定使用的单位。

```python
import numpy as np

a = np.datetime64('2020-03', 'D')
print(a, a.dtype)  # 2020-03-01 datetime64[D]

a = np.datetime64('2020-03', 'Y')
print(a, a.dtype)  # 2020 datetime64[Y]

print(np.datetime64('2020-03') == np.datetime64('2020-03-01'))  # True
print(np.datetime64('2020-03') == np.datetime64('2020-03-02'))  #False
```

从字符串创建 datetime64 数组时，如果单位不统一，则一律转化成其中最小的单位。

```python
import numpy as np

a = np.array(['2020-03', '2020-03-08', '2020-03-08 20:00'], dtype='datetime64')
print(a, a.dtype)
# ['2020-03-01T00:00' '2020-03-08T00:00' '2020-03-08T20:00'] datetime64[m]
```

使用`arange()`创建 datetime64 数组，用于生成日期范围。

```python
import numpy as np

a = np.arange('2020-08-01', '2020-08-10', dtype=np.datetime64)
print(a)
# ['2020-08-01' '2020-08-02' '2020-08-03' '2020-08-04' '2020-08-05'
#  '2020-08-06' '2020-08-07' '2020-08-08' '2020-08-09']
print(a.dtype)  # datetime64[D]
```

### datetime64 和 tmedelta64 运算

**timedelta64 表示两个 datetime64 之间的差**。timedelta64 也是带单位的，并且和相减运算中的两个 datetime64 中的较小的单位保持一致。

```python
import numpy as np

a = np.datetime64('2020-03-08') - np.datetime64('2020-03-07')
b = np.datetime64('2020-03-08') - np.datetime64('202-03-07 08:00')
c = np.datetime64('2020-03-08') - np.datetime64('2020-03-07 23:00', 'D')

print(a, a.dtype)  # 1 days timedelta64[D]
print(b, b.dtype)  # 956178240 minutes timedelta64[m]
print(c, c.dtype)  # 1 days timedelta64[D]

a = np.datetime64('2020-03') + np.timedelta64(20, 'D')
b = np.datetime64('2020-06-15 00:00') + np.timedelta64(12, 'h')
print(a, a.dtype)  # 2020-03-21 datetime64[D]
print(b, b.dtype)  # 2020-06-15T12:00 datetime64[m]
```

生成 timedelta64时，要注意年（'Y'）和月（'M'）这两个单位无法和其它单位进行运算（一年有几天？一个月有几个小时？这些都是不确定的）。

```python
import numpy as np

a = np.timedelta64(1, 'Y')
b = np.timedelta64(a, 'M')
print(a)  # 1 years
print(b)  # 12 months

c = np.timedelta64(1, 'h')
d = np.timedelta64(c, 'm')
print(c)  # 1 hours
print(d)  # 60 minutes

print(np.timedelta64(a, 'D'))
# TypeError: Cannot cast NumPy timedelta64 scalar from metadata [Y] to [D] according to the rule 'same_kind'

print(np.timedelta64(b, 'D'))
# TypeError: Cannot cast NumPy timedelta64 scalar from metadata [M] to [D] according to the rule 'same_kind'
```

timedelta64 的运算

```python
import numpy as np

a = np.timedelta64(1, 'Y')
b = np.timedelta64(6, 'M')
c = np.timedelta64(1, 'W')
d = np.timedelta64(1, 'D')
e = np.timedelta64(10, 'D')

print(a)  # 1 years
print(b)  # 6 months
print(a + b)  # 18 months
print(a - b)  # 6 months
print(2 * a)  # 2 years
print(a / b)  # 2.0
print(c / d)  # 7.0
print(c % e)  # 7 days
```

numpy.datetime64 与 datetime.datetime 相互转换

```python
import numpy as np
import datetime

dt = datetime.datetime(year=2020, month=6, day=1, hour=20, minute=5, second=30)
dt64 = np.datetime64(dt, 's')
print(dt64, dt64.dtype)
# 2020-06-01T20:05:30 datetime64[s]

dt2 = dt64.astype(datetime.datetime)
print(dt2, type(dt2))
# 2020-06-01 20:05:30 <class 'datetime.datetime'>
```

### datetime64的应用

为了允许在只有一周中某些日子有效的上下文中使用日期时间，NumPy包含一组“busday”（工作日）功能。

- `numpy.busday_offset(dates, offsets, roll='raise', weekmask='1111100', holidays=None, busdaycal=None, out=None)` First adjusts the date to fall on a valid day according to the roll rule, then applies offsets to the given dates counted in valid days.

参数`roll`：{'raise', 'nat', 'forward', 'following', 'backward', 'preceding', 'modifiedfollowing', 'modifiedpreceding'}

- 'raise' means to raise an exception for an invalid day.
- 'nat' means to return a NaT (not-a-time) for an invalid day.
- 'forward' and 'following' mean to take the first valid day later in time.
- 'backward' and 'preceding' mean to take the first valid day earlier in time.

【例】将指定的偏移量应用于工作日，单位天（'D'）。计算下一个工作日，如果当前日期为非工作日，默认报错。可以指定 `forward` 或 `backward` 规则来避免报错。（一个是向前取第一个有效的工作日，一个是向后取第一个有效的工作日）

```python
import numpy as np

# 2020-07-10 星期五
a = np.busday_offset('2020-07-10', offsets=1)
print(a)  # 2020-07-13

a = np.busday_offset('2020-07-11', offsets=1)
print(a)
# ValueError: Non-business day date in busday_offset

a = np.busday_offset('2020-07-11', offsets=0, roll='forward')
b = np.busday_offset('2020-07-11', offsets=0, roll='backward')
print(a)  # 2020-07-13
print(b)  # 2020-07-10

a = np.busday_offset('2020-07-11', offsets=1, roll='forward')
b = np.busday_offset('2020-07-11', offsets=1, roll='backward')
print(a)  # 2020-07-14
print(b)  # 2020-07-13
```

`numpy.is_busday(dates, weekmask='1111100', holidays=None, busdaycal=None, out=None)` Calculates which of the given dates are valid days, and which are not.

```python
import numpy as np

# 2020-07-10 星期五
a = np.is_busday('2020-07-10')
b = np.is_busday('2020-07-11')
print(a)  # True
print(b)  # False
```

统计一个 `datetime64[D]` 数组中的工作日天数。

```python
import numpy as np

# 2020-07-10 星期五
begindates = np.datetime64('2020-07-10')
enddates = np.datetime64('2020-07-20')
a = np.arange(begindates, enddates, dtype='datetime64')
b = np.count_nonzero(np.is_busday(a))
print(a)
# ['2020-07-10' '2020-07-11' '2020-07-12' '2020-07-13' '2020-07-14'
#  '2020-07-15' '2020-07-16' '2020-07-17' '2020-07-18' '2020-07-19']
print(b)  # 6
```

自定义周掩码值，即指定一周中哪些星期是工作日。

```python
import numpy as np

# 2020-07-10 星期五
a = np.is_busday('2020-07-10', weekmask=[1, 1, 1, 1, 1, 0, 0])
b = np.is_busday('2020-07-10', weekmask=[1, 1, 1, 1, 0, 0, 1])
print(a)  # True
print(b)  # False
```

- `numpy.busday_count(begindates, enddates, weekmask='1111100', holidays=[], busdaycal=None, out=None)`Counts the number of valid days between `begindates` and `enddates`, not including the day of `enddates`.

返回两个日期之间的工作日数量。

```python
import numpy as np

# 2020-07-10 星期五
begindates = np.datetime64('2020-07-10')
enddates = np.datetime64('2020-07-20')
a = np.busday_count(begindates, enddates)
b = np.busday_count(enddates, begindates)
print(a)  # 6
print(b)  # -6
```

## 数组的创建

### 1. 依据现有数据来创建 ndarray

#### a) 通过array() 函数进行创建

```python
def array(p_object, dtype=None, copy=True, order='K', subok=False, ndmin=0): 
```

```python
# 创建三维数组
d = np.array([[(1.5, 2, 3), (4, 5, 6)],
              [(3, 2, 1), (4, 5, 6)]])
print(d, type(d))
```

#### b）通过asarray()函数进行创建

`array()`和`asarray()`都可以将结构数据转化为 ndarray，但是`array()`和`asarray()`主要区别就是当数据源是**ndarray** 时，`array()`仍然会 copy 出一个副本，占用新的内存，但不改变 dtype 时 `asarray()`不会。

```python
def asarray(a, dtype=None, order=None):
    return array(a, dtype, copy=False, order=order)
```

```python
print(z,type(z),z.dtype)
# [[1 1 1]
#  [1 1 2]
#  [1 1 1]] <class 'numpy.ndarray'> int32
```

#### c）通过fromfunction()函数进行创建

给函数绘图的时候可能会用到`fromfunction()`，该函数可从函数中创建数组。

```python
def fromfunction(function, shape, **kwargs):
```

通过在每个坐标上执行一个函数来构造数组。

```python
x = np.fromfunction(lambda i, j: i + j, (3, 3), dtype=int)
print(x)
# [[0 1 2]
#  [1 2 3]
#  [2 3 4]]
```

### 2. 依据 ones 和 zeros 填充方式

#### a) 零数组

- `zeros()`函数：返回给定形状和类型的零数组。
- `zeros_like()`函数：返回与给定数组形状和类型相同的零数组。

```python
def zeros(shape, dtype=None, order='C'):
def zeros_like(a, dtype=None, order='K', subok=True, shape=None):
```

```python
import numpy as np

x = np.zeros(5)
print(x)  # [0. 0. 0. 0. 0.]
x = np.zeros([2, 3])
print(x)
# [[0. 0. 0.]
#  [0. 0. 0.]]

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.zeros_like(x)
print(y)
# [[0 0 0]
#  [0 0 0]]
```

#### b) 1数组

- `ones()`函数：返回给定形状和类型的1数组。
- `ones_like()`函数：返回与给定数组形状和类型相同的1数组。

```python
def ones(shape, dtype=None, order='C'):
def ones_like(a, dtype=None, order='K', subok=True, shape=None):
```

```python
import numpy as np

x = np.ones(5)
print(x)  # [1. 1. 1. 1. 1.]
x = np.ones([2, 3])
print(x)
# [[1. 1. 1.]
#  [1. 1. 1.]]

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.ones_like(x)
print(y)
# [[1 1 1]
#  [1 1 1]]
```

#### c)空数组

- `empty()`函数：返回一个空数组，数组元素为随机数。
- `empty_like`函数：返回与给定数组具有相同形状和类型的新数组。

```python
def empty(shape, dtype=None, order='C'): 
def empty_like(prototype, dtype=None, order='K', subok=True, shape=None):
```

```python
import numpy as np

x = np.empty(5)
print(x)
# [1.95821574e-306 1.60219035e-306 1.37961506e-306 
#  9.34609790e-307 1.24610383e-306]

x = np.empty((3, 2))
print(x)
# [[1.60220393e-306 9.34587382e-307]
#  [8.45599367e-307 7.56598449e-307]
#  [1.33509389e-306 3.59412896e-317]]

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.empty_like(x)
print(y)
# [[  7209029   6422625   6619244]
#  [      100 707539280       504]]
```

#### d)单位数组

- `eye()`函数：返回一个对角线上为1，其它地方为零的单位数组。
- `identity()`函数：返回一个方的单位数组。

```python
def eye(N, M=None, k=0, dtype=float, order='C'):
def identity(n, dtype=None):
```

```python
import numpy as np

x = np.eye(4)
print(x)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]

x = np.eye(2, 3)
print(x)
# [[1. 0. 0.]
#  [0. 1. 0.]]

x = np.identity(4)
print(x)
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]]
```

#### e)对角数组

- `diag()`函数：提取对角线或构造对角数组。

```python
def diag(v, k=0):
```

```python
import numpy as np

x = np.arange(9).reshape((3, 3))
print(x)
# [[0 1 2]
#  [3 4 5]
#  [6 7 8]]
print(np.diag(x))  # [0 4 8]
print(np.diag(x, k=1))  # [1 5]
print(np.diag(x, k=-1))  # [3 7]

v = [1, 3, 5, 7]
x = np.diag(v)
print(x)
# [[1 0 0 0]
#  [0 3 0 0]
#  [0 0 5 0]
#  [0 0 0 7]]
```

#### f）常数数组

- `full()`函数：返回一个常数数组。
- `full_like()`函数：返回与给定数组具有相同形状和类型的常数数组。

```python
def full(shape, fill_value, dtype=None, order='C'):
def full_like(a, fill_value, dtype=None, order='K', subok=True, shape=None):
```

```python
import numpy as np

x = np.full((2,), 7)
print(x)
# [7 7]

x = np.full(2, 7)
print(x)
# [7 7]

x = np.full((2, 7), 7)
print(x)
# [[7 7 7 7 7 7 7]
#  [7 7 7 7 7 7 7]]

x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.full_like(x, 7)
print(y)
# [[7 7 7]
#  [7 7 7]]
```

### 3. 利用数值范围来创建ndarray

- `arange()`函数：返回给定间隔内的均匀间隔的值。
- `linspace()`函数：返回指定间隔内的等间隔数字。
- `logspace()`函数：返回数以对数刻度均匀分布。
- `numpy.random.rand()` 返回一个由[0,1)内的随机数组成的数组。

```python
def arange([start,] stop[, step,], dtype=None): 
def linspace(start, stop, num=50, endpoint=True, retstep=False, 
             dtype=None, axis=0):
def logspace(start, stop, num=50, endpoint=True, base=10.0, 
             dtype=None, axis=0):
def rand(d0, d1, ..., dn): 
```

```python
import numpy as np

x = np.arange(5)
print(x)  # [0 1 2 3 4]

x = np.arange(3, 7, 2)
print(x)  # [3 5]

x = np.linspace(start=0, stop=2, num=9)
print(x)  
# [0.   0.25 0.5  0.75 1.   1.25 1.5  1.75 2.  ]

x = np.logspace(0, 1, 5)
print(np.around(x, 2))
# [ 1.    1.78  3.16  5.62 10.  ]            
                                    #np.around 返回四舍五入后的值，可指定精度。
                                   # around(a, decimals=0, out=None)
                                   # a 输入数组
                                   # decimals 要舍入的小数位数。 默认值为0。 如果为负，整数将四舍五入到小数点左侧的位置


x = np.linspace(start=0, stop=1, num=5)
x = [10 ** i for i in x]
print(np.around(x, 2))
# [ 1.    1.78  3.16  5.62 10.  ]

x = np.random.random(5)
print(x)
# [0.41768753 0.16315577 0.80167915 0.99690199 0.11812291]

x = np.random.random([2, 3])
print(x)
# [[0.41151858 0.93785153 0.57031309]
#  [0.13482333 0.20583516 0.45429181]]
```

### 4.结构数组的创建

结构数组，首先需要定义结构，然后利用`np.array()`来创建数组，其参数`dtype`为定义的结构。

#### a)利用字典来定义结构

```python
import numpy as np

personType = np.dtype({
    'names': ['name', 'age', 'weight'],
    'formats': ['U30', 'i8', 'f8']})

a = np.array([('Liming', 24, 63.9), ('Mike', 15, 67.), ('Jan', 34, 45.8)],
             dtype=personType)
print(a, type(a))
# [('Liming', 24, 63.9) ('Mike', 15, 67. ) ('Jan', 34, 45.8)]
# <class 'numpy.ndarray'>
```

#### b）利用包含多个元组的列表来定义结构

```python
import numpy as np

personType = np.dtype([('name', 'U30'), ('age', 'i8'), ('weight', 'f8')])
a = np.array([('Liming', 24, 63.9), ('Mike', 15, 67.), ('Jan', 34, 45.8)],
             dtype=personType)
print(a, type(a))
# [('Liming', 24, 63.9) ('Mike', 15, 67. ) ('Jan', 34, 45.8)]
# <class 'numpy.ndarray'>

# 结构数组的取值方式和一般数组差不多，可以通过下标取得元素：
print(a[0])
# ('Liming', 24, 63.9)

print(a[-2:])
# [('Mike', 15, 67. ) ('Jan', 34, 45.8)]

# 我们可以使用字段名作为下标获取对应的值
print(a['name'])
# ['Liming' 'Mike' 'Jan']
print(a['age'])
# [24 15 34]
print(a['weight'])
# [63.9 67.  45.8]
```

## 数组的属性

在使用 numpy 时，你会想知道数组的某些信息。很幸运，在这个包里边包含了很多便捷的方法，可以给你想要的信息。

- `numpy.ndarray.ndim`用于返回数组的维数（轴的个数）也称为秩，一维数组的秩为 1，二维数组的秩为 2，以此类推。
- `numpy.ndarray.shape`表示数组的维度，返回一个元组，这个元组的长度就是维度的数目，即 `ndim` 属性(秩)。
- `numpy.ndarray.size`数组中所有元素的总量，相当于数组的`shape`中所有元素的乘积，例如矩阵的元素总量为行与列的乘积。
- `numpy.ndarray.dtype` `ndarray` 对象的元素类型。
- `numpy.ndarray.itemsize`以字节的形式返回数组中每一个元素的大小。

```python
class ndarray(object):
    shape = property(lambda self: object(), lambda self, v: None, lambda self: None)
    dtype = property(lambda self: object(), lambda self, v: None, lambda self: None)
    size = property(lambda self: object(), lambda self, v: None, lambda self: None)
    ndim = property(lambda self: object(), lambda self, v: None, lambda self: None)
    itemsize = property(lambda self: object(), lambda self, v: None, lambda self: None)
```

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a.shape)  # (5,)
print(a.dtype)  # int32
print(a.size)  # 5
print(a.ndim)  # 1
print(a.itemsize)  # 4

b = np.array([[1, 2, 3], [4, 5, 6.0]])
print(b.shape)  # (2, 3)
print(b.dtype)  # float64
print(b.size)  # 6
print(b.ndim)  # 2
print(b.itemsize)  # 8
```

在`ndarray`中所有元素必须是同一类型，否则会自动向下转换，`int->float->str`。

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a)  # [1 2 3 4 5]
b = np.array([1, 2, 3, 4, '5'])
print(b)  # ['1' '2' '3' '4' '5']
c = np.array([1, 2, 3, 4, 5.0])
print(c)  # [1. 2. 3. 4. 5.]
```