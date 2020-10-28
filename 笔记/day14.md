Table Of Contents:

+ [今日工作](#今日工作)
+ [非线性回归](#非线性回归)
  + [最大似然法](#最大似然法)
  + [贝叶斯非线性回归](#贝叶斯非线性回归)
+ [贝叶斯线性回归拟合算法](#贝叶斯线性回归拟合算法)



### 今日工作

1、阅读《计算机视觉 模型、学习和推理》第8章第3节——非线性回归。

2、实现贝叶斯线性回归的拟合算法。



### 非线性回归

之前讨论了线性模型，即状态*ω* 与变量*x* 的关系是线性的。

但是很大情况下，是不符合线性关系的。



如果将*x* 的维度增加，维度增加后的变量为*z* ，用*z* 作为*ω* 概率密度分布的参数，显然*ω* 与*x* 的关系就不是线性了。

例如：*z_i* = *[1,  x_i,  x_i^2, x_i^3]* 。

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-28_20-11-30.png?raw=true)

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-28_20-11-07.png?raw=true)

主要是通过*z_i = f(x_i)* 这个非线性变换，建立*ω* 与*x* 的非线性关系。

#### 最大似然法

同样，与第一节线性回归相似，极值点处的参数(权重)可以由以下公式直接计算：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-28_20-14-03.png?raw=true)

#### 贝叶斯非线性回归

对比贝叶斯线性回归，贝叶斯非线性回归也只是把*x* 换成了*z* 。

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-28_20-47-41.png?raw=true)

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-28_20-48-01.png?raw=true)





### 贝叶斯线性回归拟合算法

完成了先验分布的计算，注意的点是，先验分布的均值为0，协方差为球形，先验分布的协方差为较大的值，反映先验较弱的事实，这个值由我们自己设置。

而后验分布参数的计算，也是拟合算法的关键暂时还没有完全实现。

先验分布：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-28_22-13-51.png?raw=true)

