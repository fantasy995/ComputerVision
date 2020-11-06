Table Of Contents:

+ [今日工作](#今日工作)
+ [增量模型和boosting](#增量模型和boosting)
+ [分类树](#分类树)
+ [多分类逻辑回归](#多分类逻辑回归)




### 今日工作

1、阅读《计算机视觉 模型、学习和推理》第9章（分类模型），9.7节（增量模型和boosting），9.8节（分类树），9.9节（多分类逻辑回归）。



### 增量模型和boosting

在对*x* 的非线性变换中，我们改变了数据的维度，那么几维才合适呢。

在我看来，增量模型就是为了解决这个问题的。

两个主要的公式为：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-06_20-39-57.png?raw=true)

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-06_20-40-14.png?raw=true)

核函数：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-06_20-41-30.png?raw=true)

在*a_i* 的表达式中，*φ* 的个数逐渐增加。当新添加*φ* 后，前面的*φ* 不需要进行修正，就停止添加*φ* 。

增加*φ* 的过程可以理解为不断引入*x* 的维度。



### 分类树

在之前，我们为了得到非线性决策边界，对*x* 进行了非线性变换。

这里提出了另一种方法：将数据空间分割成不同区域，在每个区域应用不同的分类器。



### 多分类逻辑回归

之前讨论的都是二分类维度，即*ω* 是二元的，概率密度为伯努利分布。

现在讨论多分类模型，则*ω* 是多元的，为分类分布。

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-06_21-17-15.png?raw=true)

*λ* 的和为1，并且属于[0,1]。

通过*softmax* 函数可以实现上述要求。

之前二分类中，我们关于*x* 有一个函数：![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-06_21-21-13.png?raw=true)

再将*a* 作为*sigmoid* 函数的输入得到[0-1]之间的值。

这里，我们同样先对*x* 进行计算以获得观测数据的特征，再将得到的数据作为*softmax* 函数的输入。

对于每个类别：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-06_21-24-10.png?raw=true)

*n* 为类别数目。

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-06_21-24-43.png?raw=true)

