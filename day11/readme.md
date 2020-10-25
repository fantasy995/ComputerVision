Table Of Contents：

+ [今日工作](#今日工作)
+ [混合高斯模型拟合算法](#混合高斯模型拟合算法)



### 今日工作

1、回顾《计算机视觉 模型、学习和推理》第七章。

2、编程实现混合高斯模型的拟合。

### 混合高斯模型拟合算法

混合模型主要是针对一个问题：某个状态*ω* 对应的观测数据的分布并不总是集中的。

集中的情况：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-25_21-00-44.png?raw=true)

不集中的情况：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-25_21-02-08.png?raw=true)

*1)* 首先，我们模拟不集中的数据，从两个二元正态分布中随机抽取一定比例的数据。

```python
x1 = np.random.multivariate_normal([1, 2], [[2, 0], [0, 0.5]], size=3500)
x2 = np.random.multivariate_normal([1, 5], [[1, 0], [0, 1]], size=1500)
x = np.append(x1, x2, axis=0)
```

*2)* 初始化*K* 个二元正态分布的参数，使用数据集的信息初始化，这样也许更接近*x* 的分布状态。

*3)* 执行*E* 步和参数更新的操作。

*4)* 统计新参数下整体的对数似然，计算与前一次的对数似然相减的绝对值，若绝对值足够小则认为达到了极值点，也就是局部最优解，停止拟合。