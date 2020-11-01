Table Of Contents:

+ [今日工作](#今日工作)
+ [分类模型](#分类模型)
  + [逻辑回归](#逻辑回归)



### 今日工作

1、阅读《计算机视觉 模型、学习和推理》第八章——分类模型，第1节：逻辑回归。



### 分类模型

分类模型中，全局状态*ω* 取值是离散的：*ω∈{1，……，k}* 。

对后验概率*Pr(ω|x)* 建立模型。

本章基于的问题是：性别分类，因此本章中*ω* 是二值的。

#### 逻辑回归

因为*ω* 是二值离散的，因此我们选择伯努利分布。

将伯努利分布的参数*λ* 与*x* 建立关系，那么*λ=Φ_0+Φ^T \* x* ，又*λ∈[0,1]* ，对这个值再使用*sigmoid* 函数进行输出。

其中，*x* 的维度是*D\*1* ，*Φ* 的维度是*D\*1* 。

与第八章拓展维度的方式相同，我们通过以下方法可以简化表达式：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-01_21-42-45.png?raw=true)， ![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-01_21-43-12.png?raw=true)

那么，后验分布可以写成：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-01_21-43-48.png?raw=true)

学习（最大似然估计）：

假设各样本独立，最大似然估计就是要让我们训练数据的概率密度乘积最大：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-01_21-55-16.png?raw=true)

*X* 的维度是*D\*I* ，*ω_i* 的取值是0或1。

由于是乘积，我们可以对数化再求极值点：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-01_21-57-28.png?raw=true)

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-01_21-57-40.png?raw=true)

这个式子无法闭式求解，因此需要设置一个*Φ* 的初始值，再对其进行迭代。



同样，针对逻辑回归模型存在的问题，我们有相对应的解决方案：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-01_22-02-50.png?raw=true)

