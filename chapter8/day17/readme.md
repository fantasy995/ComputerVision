Table Of Contents:

+ [今日工作](#今日工作)
+ [二元线性回归](#二元线性回归)
  + [二元模型](#二元模型)
  + [最大似然解 ](#最大似然解 )
  + [贝叶斯解](#贝叶斯解)
+ [相关向量回归](#相关向量回归)
+ [多变量数据回归](#多变量数据回归)



### 今日工作

1、阅读《计算机视觉 模型、学习和推理》第8章（回归模型）的第7，8，9节——二元线性回归、相关向量回归、多变量数据回归。



### 二元线性回归

在标准线性回归模型中，参数向量 *Φ* 包含 *D*  个元素，分别对应于输人数据(可能是变 换过的)的 *D*  个维度。在二元规划中，我们重新将模型以包含 *I*  个元素的向量*ψ*  来参数 化，其中 *I* 是训练样本的个数。这在训练中输入数据的维度很高但是样本个数很小 *(I<D)*  的时候更有效，并且可以产生其他有趣的模型，例如相关向量回归。

#### 二元模型

在二元模型中，保持预测*ω* 对输入数据*x* 的原始依赖，使得：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-31_21-33-10.png?raw=true)

其中，*Φ = Xψ* ，*X* 维度为*D\*I* ，*I* 指训练样本数量，*D* 是*x* 的维度， *ψ* 的维度为*I\*1* 。所以*Φ* 的维度为*D\*1* 。

那么，可以得到：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-31_21-37-54.png?raw=true)

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-31_21-38-08.png?raw=true)

模型的参数为*θ={ψ, σ^2}* 。与之前类似，有最大似然法和贝叶斯法求模型的参数。

#### 最大似然解 

最大似然解是

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-31_21-38-08.png?raw=true)

取得最大值时所对应的参数。

#### 贝叶斯解

先验分布为 ![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-31_21-41-31.png?raw=true)，

后验分布：![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-31_21-42-24.png?raw=true)

最终得到表达式：![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-31_21-43-14.png?raw=true)

其中：![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-31_21-43-43.png?raw=true)

预测：![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-31_21-45-28.png?raw=true)



### 相关向量回归

相关向量回归结合二元变量与稀疏线性模型的知识，将二元参数的正态先验分布替换为一维*t* 分布的乘积：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-31_21-56-29.png?raw=true)



### 多变量数据回归

之前讨论的状态*ω* 是标量，实际上很多问题的*ω* 是有多个维度的，我们只需要对每个维度分布建立前面叙述的模型即可。