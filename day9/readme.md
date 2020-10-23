### 复杂数据密度建模

#### 隐变量

隐变量可以是离散或连续的，个数与数据*x* 个数相同（既然属于变量，那么数据的个数理所应当相同）。

在之前，数据*x* 得分布与参数*θ* 有关，引入隐变量后，这个分布首先是*x* 和隐变量*h* 的联合分布关于关于*h* 的边缘分布，即：

![](E:\ProgramData\Anaconda3\envs\imageprocessing\projects\images\Snipaste_2020-10-23_15-55-27.png)

那么，现在的参数选择方法是：

![](E:\ProgramData\Anaconda3\envs\imageprocessing\projects\images\Snipaste_2020-10-23_15-56-24.png)

这个方法称为期望最大方法。