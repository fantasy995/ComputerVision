Table Of Contents:

+ [今日工作](#今日工作)
+ [线性回归](#线性回归)
+ [贝叶斯线性回归](#贝叶斯线性回归)
+ [后验与先验](#后验与先验)
+ [线性回归拟合算法](#线性回归拟合算法)



### 今日工作

1、阅读《计算机视觉 模型、学习和推理》第八章1、2节——线性回归和贝叶斯线性回归。

2、加深对后验分布与先验分布的理解。

3、编程实现书中的线性回归拟合算法。

### 线性回归

待求的模型：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_18-08-22.png?raw=true)

*X* 是数据集合，维度为N * D，N是数据的个数，D是数据*x* 的维度。

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_18-13-25.png?raw=true)通过矩阵乘法，*Φ_i* 可以理解为第*i* 维的的梯度。

通过变换：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_18-15-54.png?raw=true)和![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_18-16-13.png?raw=true)

原分布可以写成:

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_18-16-54.png?raw=true)

例如1维的*X* :

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_18-17-42.png?raw=true)

变成：(50, 2)

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_18-18-15.png?raw=true)

*Ф* 为：(50, 1)

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_18-26-34.png?raw=true)



参数：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_18-36-01.png?raw=true)

对数化不改变极值点位置。由于函数是线性的，可以通过求偏导求得极值点位置:

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_18-36-48.png?raw=true)



### 贝叶斯线性回归

这个方法中引入了*Φ* 的先验*Pr(Φ)* 。

*Φ* 是多元连续的。

将*Pr(Φ)* 建模为均值为0，协方差为球形的多元正态分布。

根据训练样本，*Φ* 的后验分布为：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_21-17-27.png?raw=true)

似然函数为前一节*线性回归* 的：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_21-21-59.png?raw=true)

根据正态分布的运算特性，后验分布可以化为以下式子：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_21-37-15.png?raw=true)

其中：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_21-37-34.png?raw=true)



最终，我们对*ω* 的预测为:

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_21-40-08.png?raw=true)

### 后验与先验

例子：

一个班有50个人，其中49个男的，1个女的。

在公园碰到一个同学，这个同学性别是男的概率为49/50。

如果知道这个同学穿裙子，这个同学性别是男的概率是 p(穿裙子|男生)* 49 / 50

这个同学是男生的**先验概率**是基于客观情况的估计即 P(男生)=49/50

这个同学是男生的**后验概率**是 P(男生|穿裙子) = p(穿裙子|男生)* 49 / 50，即观测到一定条件后之后，一个事件的概率，同时这个条件是否能够发生本身是有一个概率。



在先验分布*Pr(Φ)* 中，*Φ* 的分布本来是非常广的：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_20-39-56.png?raw=true)

但是由于存在一个观测到的事实：*x* ，那么*Φ* 就会变窄，因为在*Φ* 这个区域内时，我们观测到*x* 的概率是更高的。

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_20-41-52.png?raw=true)

### 线性回归拟合算法

这个算法比较简单，关键步骤是：

1、根据书中的偏导公式，可以直接计算极值点处的参数。

```python
def fit_lr(X, w):    
    phi = np.dot(np.dot(np.linalg.inv(np.dot(X, X.T)), X), w)    
    # 正态分布均值 : phi[1, :]    
    batch_size = X.shape[1]    
    temp = w - np.dot(X.T, phi)    
    sigma = np.dot(temp.T, temp) / batch_size    
    return phi, sigma
```

2、建立一个自定义的状态*ω* 与*x* 的线性关系。

```python
phi = np.array([[7], [-0.5]])
X_phi = np.dot(X, phi)
r = np.random.randn(batch_size, 1)
for i in range(batch_size):    
    mu = X_phi[i]    
    w[i] = mu + r[i]
```

第一维梯度是-0.5，7是*Φ_0* ，r[i]为干扰信息。

3、最后是验证，验证思路与之前的算法相同，从区域内均匀提取大量的点，计算这些点的概率密度。



结果分析：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-27_21-19-43.png?raw=true)

这是概率密度用热力图可视化以及数据的散点图可视化结果。

可以看出，数据集中的地方概率密度值是更大的。