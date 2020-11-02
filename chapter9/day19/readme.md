Table Of Contents:

+ [今日工作](#今日工作)
+ [贝叶斯逻辑回归](#贝叶斯逻辑回归)
+ [总结](#总结)



### 今日工作

1、阅读《计算机视觉 模型、学习与推理》第9章（分类模型）的第2节——贝叶斯逻辑回归。



### 贝叶斯逻辑回归

与之前一样，提到贝叶斯就需要考虑先验分布，与线性回归相同，都是参数*Φ* 的先验分布。在普通逻辑回归中，我们直接拟合参数*Φ* 。

由于*Φ* 是多元连续的，因此使用均值为0，球形协方差的多元正态分布为先验分布。

同样，*Φ* 后验概率的计算需要使用贝叶斯公式得到：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-02_20-37-39.png?raw=true)

由于似然概率密度*Pr(ω|X,Φ)* 是伯努利分布，先验分布*Pr(Φ)* 为多元正态分布，两者没有共轭关系，因此后验分布无法闭式求解。

因此引入了新的拟合方法：拉普拉斯近似。

使用一个峰值相同的多元正态分布，正态方差使得峰值处的二阶段数与原始概率密度函数的二阶导数相同。

峰值处的*Φ* 根据MAP估计得到，即一个使![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-02_21-10-34.png?raw=true)

取最大值的*Φ* 。

峰值处的二阶导数：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-02_21-11-39.png?raw=true)

这样，我们就都得到了这个近似的正态分布的所以参数：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-02_21-12-13.png?raw=true)

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-02_21-12-24.png?raw=true)

推理时，同样需要使用无限加权和，考虑所有*Φ* ，根据权重*Pr(Φ|X, ω)* 决定每个*Φ* 的影响。

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-02_21-18-22.png?raw=true)



### 总结

贝叶斯线性回归相比最大似然模型起预测更加平缓，可以视为不那么极端，最大似然在远离中心的地方概率变为0或1。

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-11-02_21-21-16.png?raw=true)

