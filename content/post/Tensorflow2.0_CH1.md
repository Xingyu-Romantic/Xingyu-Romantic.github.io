---
title: "Tensorflow2_CH1"
date: 2020-08-24T13:04:48+08:00
draft: false
---

# Tensorflow 2.x 

>B站视频： BV1zE411T7nb

###### Hello, World!

```python
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)

print(model.predict([10.0]))
```

###### Example 1: recognize shoes and t-shirts  

```python
# 导入相关包
import keras
import tensorflow as tf
import numpy as np
#写回调类
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss')< 0.4):
            print('\n Loss is low so cancelling training!')
            self.model.stop_training = True
#调用回调函数，取消训练
callbacks = myCallback()
# 加载数据集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# 处理数据
train_images = train_images / 255.0
test_images = test_images / 255.0
# 定义模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
# 设置模型
model.compile(optimizer = tf.optimizers.Adam(),
             loss = 'sparse_categorical_crossentropy',
             metrics=['accuracy'])
# 训练模型
model.fit(train_images, train_labels, epochs=5, callbacks=[callbacks])
# 评估模型
model.evaluate(test_images, test_labels)
```

##### 应用卷积神经网络

```python
model = tf.keras.models.Sequential([
	#卷积层1   生成64个过滤器  最后维度   (28-3+1)*(28-3+1)*64 = 43264
	tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
	#池化层1 默认s=(2,2)   ((26-2)/2+1)*((26-2)/2+1)*64 = 10816
	tf.keras.layers.MaxPooling2D(2,2),
	#卷积层2    (13-3+1)*(13-3+1)*64 =  7744
	tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
	#池化层2    ((11-2)/2+1)*((11-2)/2+1)*64 = 1600
 	tf.keras.layers.MaxPooling2D(2,2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(10, activation='softmax')
])
```

###### tf.keras.layers.Conv2D

```python
keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
```

>- **filters**: 整数，输出空间的维度 （即卷积中滤波器的输出数量）。
>- **kernel_size**: 一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 可以是一个整数，为所有空间维度指定相同的值。
>- **strides**: 一个整数，或者 2 个整数表示的元组或列表， 指明卷积沿宽度和高度方向的步长。 可以是一个整数，为所有空间维度指定相同的值。 指定任何 stride 值 != 1 与指定 `dilation_rate` 值 != 1 两者不兼容。
>- **padding**: `"valid"` 或 `"same"` (大小写敏感)。
>- **data_format**: 字符串， `channels_last` (默认) 或 `channels_first` 之一，表示输入中维度的顺序。 `channels_last` 对应输入尺寸为 `(batch, height, width, channels)`， `channels_first` 对应输入尺寸为 `(batch, channels, height, width)`。 它默认为从 Keras 配置文件 `~/.keras/keras.json` 中 找到的 `image_data_format` 值。 如果你从未设置它，将使用 `channels_last`。
>- **dilation_rate**: 一个整数或 2 个整数的元组或列表， 指定膨胀卷积的膨胀率。 可以是一个整数，为所有空间维度指定相同的值。 当前，指定任何 `dilation_rate` 值 != 1 与 指定 stride 值 != 1 两者不兼容。
>- **activation**: 要使用的激活函数 (详见 [activations](https://keras.io/zh/activations/))。 如果你不指定，则不使用激活函数 (即线性激活： `a(x) = x`)。
>- **use_bias**: 布尔值，该层是否使用偏置向量。
>- **kernel_initializer**: `kernel` 权值矩阵的初始化器 (详见 [initializers](https://keras.io/zh/initializers/))。
>- **bias_initializer**: 偏置向量的初始化器 (详见 [initializers](https://keras.io/zh/initializers/))。
>- **kernel_regularizer**: 运用到 `kernel` 权值矩阵的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
>- **bias_regularizer**: 运用到偏置向量的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
>- **activity_regularizer**: 运用到层输出（它的激活值）的正则化函数 (详见 [regularizer](https://keras.io/zh/regularizers/))。
>- **kernel_constraint**: 运用到 `kernel` 权值矩阵的约束函数 (详见 [constraints](https://keras.io/zh/constraints/))。
>- **bias_constraint**: 运用到偏置向量的约束函数 (详见 [constraints](https://keras.io/zh/constraints/))。

###### keras.layers.MaxPooling2D

```python
keras.layers.MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid', data_format=None)
```

>- **pool_size**: 整数，或者 2 个整数表示的元组， 沿（垂直，水平）方向缩小比例的因数。 （2，2）会把输入张量的两个维度都缩小一半。 如果只使用一个整数，那么两个维度都会使用同样的窗口长度。
>- **strides**: 整数，2 个整数表示的元组，或者是 `None`。 表示步长值。 如果是 `None`，那么默认值是 `pool_size`。
>- **padding**: `"valid"` 或者 `"same"` （区分大小写）。
>- **data_format**: 字符串，`channels_last` (默认)或 `channels_first` 之一。 表示输入各维度的顺序。 `channels_last` 代表尺寸是 `(batch, height, width, channels)` 的输入张量， 而 `channels_first` 代表尺寸是 `(batch, channels, height, width)` 的输入张量。 默认值根据 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值来设置。 如果还没有设置过，那么默认值就是 "channels_last"。

##### 图片数据生成器

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescalse=1./255)
train_generator = train_datagen.flow_from_directory(
			train_dir,
    		target_size = (300,300),
    		batch_size = 128,
    		class_mode = 'binary'
			)
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
			validation_dir,
    		target_size = (300,300),
			batch_size = 32,
			class_mode = 'binary'
			)

```

##### ImageDataGenerator 类

```python
keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,  
                                             samplewise_center=False, 
                                             featurewise_std_normalization=False, 
                                             samplewise_std_normalization=False, 
                                             zca_whitening=False, 
                                             zca_epsilon=1e-06, 
                                             rotation_range=0, 
                                             width_shift_range=0.0, 
                                             height_shift_range=0.0, 
                                             brightness_range=None, 
                                             shear_range=0.0, 
                                             zoom_range=0.0, 
                                             channel_shift_range=0.0, 
                                             fill_mode='nearest', 
                                             cval=0.0, 
                                             horizontal_flip=False, 
                                             vertical_flip=False, 
                                             rescale=None, 
                                             preprocessing_function=None, 
                                             data_format=None, 
                                             validation_split=0.0, 
                                             dtype=None)
```

>- **featurewise_center**: 布尔值。将输入数据的均值设置为 0，逐特征进行。
>
>- **samplewise_center**: 布尔值。将每个样本的均值设置为 0。
>
>- **featurewise_std_normalization**: Boolean. 布尔值。将输入除以数据标准差，逐特征进行。
>
>- **samplewise_std_normalization**: 布尔值。将每个输入除以其标准差。
>
>- **zca_epsilon**: ZCA 白化的 epsilon 值，默认为 1e-6。
>
>- **zca_whitening**: 布尔值。是否应用 ZCA 白化。
>
>- **rotation_range**: 整数。随机旋转的度数范围。
>
>- width_shift_range
>
>  : 浮点数、一维数组或整数
>
>  - float: 如果 <1，则是除以总宽度的值，或者如果 >=1，则为像素值。
>  - 1-D 数组: 数组中的随机元素。
>  - int: 来自间隔 `(-width_shift_range, +width_shift_range)` 之间的整数个像素。
>  - `width_shift_range=2` 时，可能值是整数 `[-1, 0, +1]`，与 `width_shift_range=[-1, 0, +1]` 相同；而 `width_shift_range=1.0` 时，可能值是 `[-1.0, +1.0)` 之间的浮点数。
>
>- height_shift_range
>
>  : 浮点数、一维数组或整数
>
>  - float: 如果 <1，则是除以总宽度的值，或者如果 >=1，则为像素值。
>  - 1-D array-like: 数组中的随机元素。
>  - int: 来自间隔 `(-height_shift_range, +height_shift_range)` 之间的整数个像素。
>  - `height_shift_range=2` 时，可能值是整数 `[-1, 0, +1]`，与 `height_shift_range=[-1, 0, +1]` 相同；而 `height_shift_range=1.0` 时，可能值是 `[-1.0, +1.0)` 之间的浮点数。
>
>- **shear_range**: 浮点数。剪切强度（以弧度逆时针方向剪切角度）。
>
>- **zoom_range**: 浮点数 或 `[lower, upper]`。随机缩放范围。如果是浮点数，`[lower, upper] = [1-zoom_range, 1+zoom_range]`。
>
>- **channel_shift_range**: 浮点数。随机通道转换的范围。
>
>- fill_mode
>
>  : {"constant", "nearest", "reflect" or "wrap"} 之一。默认为 'nearest'。输入边界以外的点根据给定的模式填充：
>
>  - 'constant': kkkkkkkk|abcd|kkkkkkkk (cval=k)
>  - 'nearest': aaaaaaaa|abcd|dddddddd
>  - 'reflect': abcddcba|abcd|dcbaabcd
>  - 'wrap': abcdabcd|abcd|abcdabcd
>
>- **cval**: 浮点数或整数。用于边界之外的点的值，当 `fill_mode = "constant"` 时。
>
>- **horizontal_flip**: 布尔值。随机水平翻转。
>
>- **vertical_flip**: 布尔值。随机垂直翻转。
>
>- **rescale**: 重缩放因子。默认为 None。如果是 None 或 0，不进行缩放，否则将数据乘以所提供的值（在应用任何其他转换之前）。
>
>- **preprocessing_function**: 应用于每个输入的函数。这个函数会在任何其他改变之前运行。这个函数需要一个参数：一张图像（秩为 3 的 Numpy 张量），并且应该输出一个同尺寸的 Numpy 张量。
>
>- **data_format**: 图像数据格式，{"channels_first", "channels_last"} 之一。"channels_last" 模式表示图像输入尺寸应该为 `(samples, height, width, channels)`，"channels_first" 模式表示输入尺寸应该为 `(samples, channels, height, width)`。默认为 在 Keras 配置文件 `~/.keras/keras.json` 中的 `image_data_format` 值。如果你从未设置它，那它就是 "channels_last"。
>
>- **validation_split**: 浮点数。Float. 保留用于验证的图像的比例（严格在0和1之间）。
>
>- **dtype**: 生成数组使用的数据类型。

##### ImageDataGenerator 类方法

###### apply_transform

```python
apply_transform(x, transform_parameters)
```

>将数据生成器用于某些示例数据。
>
>它基于一组样本数据，计算与数据转换相关的内部数据统计。
>
>当且仅当 `featurewise_center` 或 `featurewise_std_normalization` 或 `zca_whitening` 设置为 True 时才需要。
>
>- **x**: 3D 张量，单张图像。
>
>- transform_parameters
>
>  : 字符串 - 参数 对表示的字典，用于描述转换。目前，使用字典中的以下参数：
>
>  - 'theta': 浮点数。旋转角度（度）。
>  - 'tx': 浮点数。在 x 方向上移动。
>  - 'ty': 浮点数。在 y 方向上移动。
>  - shear': 浮点数。剪切角度（度）。
>  - 'zx': 浮点数。放大 x 方向。
>  - 'zy': 浮点数。放大 y 方向。
>  - 'flip_horizontal': 布尔 值。水平翻转。
>  - 'flip_vertical': 布尔值。垂直翻转。
>  - 'channel_shift_intencity': 浮点数。频道转换强度。
>  - 'brightness': 浮点数。亮度转换强度。

###### fit

```python
fit(x, augment=False, rounds=1, seed=None)
```

>将数据生成器用于某些示例数据。
>
>它基于一组样本数据，计算与数据转换相关的内部数据统计。
>
>当且仅当 `featurewise_center` 或 `featurewise_std_normalization` 或 `zca_whitening` 设置为 True 时才需要。
>
>- **x**: 样本数据。秩应该为 4。对于灰度数据，通道轴的值应该为 1；对于 RGB 数据，值应该为 3。
>- **augment**: 布尔值（默认为 False）。是否使用随机样本扩张。
>- **rounds**: 整数（默认为 1）。如果数据数据增强（augment=True），表明在数据上进行多少次增强。
>- **seed**: 整数（默认 None）。随机种子。

###### flow

```python
flow(x, y=None, batch_size=32, shuffle=True, sample_weight=None, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None)
```

>采集数据和标签数组，生成批量增强数据。
>
>**参数**
>
>- **x**: 输入数据。秩为 4 的 Numpy 矩阵或元组。如果是元组，第一个元素应该包含图像，第二个元素是另一个 Numpy 数组或一列 Numpy 数组，它们不经过任何修改就传递给输出。可用于将模型杂项数据与图像一起输入。对于灰度数据，图像数组的通道轴的值应该为 1，而对于 RGB 数据，其值应该为 3。
>- **y**: 标签。
>- **batch_size**: 整数 (默认为 32)。
>- **shuffle**: 布尔值 (默认为 True)。
>- **sample_weight**: 样本权重。
>- **seed**: 整数（默认为 None）。
>- **save_to_dir**: None 或 字符串（默认为 None）。这使您可以选择指定要保存的正在生成的增强图片的目录（用于可视化您正在执行的操作）。
>- **save_prefix**: 字符串（默认 `''`）。保存图片的文件名前缀（仅当 `save_to_dir` 设置时可用）。
>- **save_format**: "png", "jpeg" 之一（仅当 `save_to_dir` 设置时可用）。默认："png"。
>- **subset**: 数据子集 ("training" 或 "validation")，如果 在 `ImageDataGenerator` 中设置了 `validation_split`。
>
>**返回**
>
>一个生成元组 `(x, y)` 的 `Iterator`，其中 `x` 是图像数据的 Numpy 数组（在单张图像输入时），或 Numpy 数组列表（在额外多个输入时），`y` 是对应的标签的 Numpy 数组。如果 'sample_weight' 不是 None，生成的元组形式为 `(x, y, sample_weight)`。如果 `y` 是 None, 只有 Numpy 数组 `x` 被返回。

###### flow_from_dataframe

```python
flow_from_dataframe(dataframe, directory, x_col='filename', y_col='class', has_ext=True, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', subset=None, interpolation='nearest')
```

>输入 dataframe 和目录的路径，并生成批量的增强/标准化的数据。
>
>这里有一个简单的教程： http://bit.ly/keras_flow_from_dataframe
>
>**参数**
>
>- **dataframe**: Pandas dataframe，一列为图像的文件名，另一列为图像的类别， 或者是可以作为原始目标数据多个列。
>
>- **directory**: 字符串，目标目录的路径，其中包含在 dataframe 中映射的所有图像。
>
>- **x_col**: 字符串，dataframe 中包含目标图像文件夹的目录的列。
>
>- **y_col**: 字符串或字符串列表，dataframe 中将作为目标数据的列。
>
>- **has_ext**: 布尔值，如果 dataframe[x_col] 中的文件名具有扩展名则为 True，否则为 False。
>
>- **target_size**: 整数元组 `(height, width)`，默认为 `(256, 256)`。 所有找到的图都会调整到这个维度。
>
>- **color_mode**: "grayscale", "rbg" 之一。默认："rgb"。 图像是否转换为 1 个或 3 个颜色通道。
>
>- **classes**: 可选的类别列表 (例如， `['dogs', 'cats']`)。默认：None。 如未提供，类比列表将自动从 y_col 中推理出来，y_col 将会被映射为类别索引）。 包含从类名到类索引的映射的字典可以通过属性 `class_indices` 获得。
>
>- class_mode
>
>  : "categorical", "binary", "sparse", "input", "other" or None 之一。 默认："categorical"。决定返回标签数组的类型：
>
>  - `"categorical"` 将是 2D one-hot 编码标签，
>  - `"binary"` 将是 1D 二进制标签，
>  - `"sparse"` 将是 1D 整数标签，
>  - `"input"` 将是与输入图像相同的图像（主要用于与自动编码器一起使用），
>  - `"other"` 将是 y_col 数据的 numpy 数组，
>  - None, 不返回任何标签（生成器只会产生批量的图像数据，这对使用 `model.predict_generator()`, `model.evaluate_generator()` 等很有用）。
>
>- **batch_size**: 批量数据的尺寸（默认：32）。
>
>- **shuffle**: 是否混洗数据（默认：True）
>
>- **seed**: 可选的混洗和转换的随即种子。
>
>- **save_to_dir**: None 或 str (默认: None). 这允许你可选地指定要保存正在生成的增强图片的目录（用于可视化您正在执行的操作）。
>
>- **save_prefix**: 字符串。保存图片的文件名前缀（仅当 `save_to_dir` 设置时可用）。
>
>- **save_format**: "png", "jpeg" 之一（仅当 `save_to_dir` 设置时可用）。默认："png"。
>
>- **follow_links**: 是否跟随类子目录中的符号链接（默认：False）。
>
>- **subset**: 数据子集 (`"training"` 或 `"validation"`)，如果在 `ImageDataGenerator` 中设置了 `validation_split`。
>
>- **interpolation**: 在目标大小与加载图像的大小不同时，用于重新采样图像的插值方法。 支持的方法有 `"nearest"`, `"bilinear"`, and `"bicubic"`。 如果安装了 1.1.3 以上版本的 PIL 的话，同样支持 `"lanczos"`。 如果安装了 3.4.0 以上版本的 PIL 的话，同样支持 `"box"` 和 `"hamming"`。 默认情况下，使用 `"nearest"`。
>
>**Returns**
>
>一个生成 `(x, y)` 元组的 DataFrameIterator， 其中 `x` 是一个包含一批尺寸为 `(batch_size, *target_size, channels)` 的图像样本的 numpy 数组，`y` 是对应的标签的 numpy 数组。

###### flow_from_directory

```python
flow_from_directory(directory, target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest')
```

>**参数**
>
>- **directory**: 目标目录的路径。每个类应该包含一个子目录。任何在子目录树下的 PNG, JPG, BMP, PPM 或 TIF 图像，都将被包含在生成器中。更多细节，详见 [此脚本](https://gist.github.com/fchollet/        0830affa1f7f19fd47b06d4cf89ed44d)。
>
>- **target_size**: 整数元组 `(height, width)`，默认：`(256, 256)`。所有的图像将被调整到的尺寸。
>
>- **color_mode**: "grayscale", "rbg" 之一。默认："rgb"。图像是否被转换成 1 或 3 个颜色通道。
>
>- **classes**: 可选的类的子目录列表（例如 `['dogs', 'cats']`）。默认：None。如果未提供，类的列表将自动从 `directory` 下的 子目录名称/结构 中推断出来，其中每个子目录都将被作为不同的类（类名将按字典序映射到标签的索引）。包含从类名到类索引的映射的字典可以通过 `class_indices` 属性获得。
>
>- class_mode
>
>  : "categorical", "binary", "sparse", "input" 或 None 之一。默认："categorical"。决定返回的标签数组的类型：
>
>  - "categorical" 将是 2D one-hot 编码标签，
>  - "binary" 将是 1D 二进制标签，"sparse" 将是 1D 整数标签，
>  - "input" 将是与输入图像相同的图像（主要用于自动编码器）。
>  - 如果为 None，不返回标签（生成器将只产生批量的图像数据，对于 `model.predict_generator()`, `model.evaluate_generator()` 等很有用）。请注意，如果 `class_mode` 为 None，那么数据仍然需要驻留在 `directory` 的子目录中才能正常工作。
>
>- **batch_size**: 一批数据的大小（默认 32）。
>
>- **shuffle**: 是否混洗数据（默认 True）。
>
>- **seed**: 可选随机种子，用于混洗和转换。
>
>- **save_to_dir**: None 或 字符串（默认 None）。这使你可以最佳地指定正在生成的增强图片要保存的目录（用于可视化你在做什么）。
>
>- **save_prefix**: 字符串。 保存图片的文件名前缀（仅当 `save_to_dir` 设置时可用）。
>
>- **save_format**: "png", "jpeg" 之一（仅当 `save_to_dir` 设置时可用）。默认："png"。
>
>- **follow_links**: 是否跟踪类子目录中的符号链接（默认为 False）。
>
>- **subset**: 数据子集 ("training" 或 "validation")，如果 在 `ImageDataGenerator` 中设置了 `validation_split`。
>
>- **interpolation**: 在目标大小与加载图像的大小不同时，用于重新采样图像的插值方法。 支持的方法有 `"nearest"`, `"bilinear"`, and `"bicubic"`。 如果安装了 1.1.3 以上版本的 PIL 的话，同样支持 `"lanczos"`。 如果安装了 3.4.0 以上版本的 PIL 的话，同样支持 `"box"` 和 `"hamming"`。 默认情况下，使用 `"nearest"`。
>
>**返回**
>
>一个生成 `(x, y)` 元组的 `DirectoryIterator`，其中 `x` 是一个包含一批尺寸为 `(batch_size, *target_size, channels)`的图像的 Numpy 数组，`y` 是对应标签的 Numpy 数组。

###### get_random_transform

```python
get_random_transform(img_shape, seed=None)
```

>为转换生成随机参数。
>
>**参数**
>
>- **seed**: 随机种子
>- **img_shape**: 整数元组。被转换的图像的尺寸。
>
>**返回**
>
>包含随机选择的描述变换的参数的字典。

###### random_transform

```python
random_transform(x, seed=None)
```

>将随机变换应用于图像。
>
>**参数**
>
>- **x**: 3D 张量，单张图像。
>- **seed**: 随机种子。
>
>**返回**
>
>输入的随机转换版本（相同形状）。

###### standardize

```python
standardize(x)
```

>将标准化配置应用于一批输入。
>
>**参数**
>
>- **x**: 需要标准化的一批输入。
>
>**返回**
>
>标准化后的输入。

