Table Of Contents:

+ [今日工作](#今日工作)
+ [视觉学习和推理](#视觉学习和推理)
  + [计算机视觉问题的解决方案的构成](#计算机视觉问题的解决方案的构成)
  + [判别模型介绍](#判别模型介绍)
  + [生成模型介绍](#生成模型介绍)
  + [二值分类的示例](#二值分类的示例)
    + [判别模型](#判别模型)
    + [生成模型](#生成模型)
  + [总结](#总结)

### 今日工作

1、阅读《计算机视觉 模型、学习和推理》第6章——*视觉学习和推理*。

2、编程练习。

### 视觉学习和推理

本章是书中的第二部分的第一章，第二部分的主题是**机器视觉的机器学习**。

本章阐述**测量图像**与**真实场景内容**相关的分类模型。

并且将模型分为两类：**生成模型**和**判别模型**。

我们的任务是通过观测的图像数据，判断内容的状态。

例如输入*x*是图片中汽车的像素个数，*ω*是汽车的尺寸。

我们建立模型，并且通过图片中汽车的像素个数预测汽车的尺寸。

#### 计算机视觉问题的解决方案的构成

**模型**：在数学上，将视觉数据*x*和全局状态*ω*关联起来。模型指定*x*与*ω*之间一系列可能的关系。

使用模型参数*θ*确定这一关系。

**学习算法**：用成对的训练样本 *{xi, ωi}* 来拟合参数 *θ* 。

**推理算法**：根据新的观测值*x*，利用模型来返回全局状态*ω*对应的后验概率*Pr(ω|x,θ)*。

（有可能返回MAP解或者从后验中抽样）

#### 判别模型介绍

建立在数据 *Pr(ω|x)* 上的全局状态可能模型。

要建立的模型：*Pr(ω|x)*。

首先，我们选择一个适合*ω*的分布*Pr(ω)*，分布的参数是*θ*。

*θ* 的取值是关于 *x* 的函数。

因此 *Pr(ω)* 返回的值与 *x* 和 *θ* 有关。将其写为 *Pr(ω|x, θ)* ，将其称为**后验分布**。

**学习的目标**：利用训练数据 *{xi，ωi}* 拟合参数 *θ* 。这可以通过最大似然、最大后验或贝叶斯方法得到。

**推理的目标**：对新的观测值 *x* 求出关于可能的全局状态 *ω* 的一个分布。

**推理的过程**：直接通过经过学习的模型 *Pr(ω|x,θ)* 得到。

#### 生成模型介绍

建立在全局状态 *Pr(x|ω)* 上的数据可能性模型。

要建立的模型：*Pr(x|ω)*。

首先，选择关于数据分布 *Pr(x)* 的形式，将分布参数 *θ* 设为 *ω* 的函数。

函数 *Pr(x)* 返回的值与 *ω* 和 *θ* 有关，将其记为 *Pr(x|ω, θ)* ，将其称为**似然函数**。

**学习的目标**：拟合参数 *θ* 。

**推理的目标**：计算**后验分布** *Pr(ω|x)*。

**推理的过程**：

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-21_17-47-12.png?raw=true)



![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-21_17-49-48.png?raw=true)

#### 二值分类的示例

条件：

观测值 *x* 连续。

状态 *ω* 二值：0，1。

##### 判别模型

定义全局状态*ω∈{0,1}* 的一个概率分布*Pr(ω)* 。

状态*ω* 表示成功的概率，*λ∈[0,1]* ，*Pr(ω=1)=λ* 。

*λ* 是关于*x* 的函数，建立*x* 的线性函数*Φ0 + Φ1x* 。

因为*λ∈[0,1]* ，使用`sigmoid`函数将*Φ0 + Φ1x *函数的值映射到*[0,1]* 。

##### 生成模型

选取数据*x* 的一个概率分布*Pr(x)* ，参数*θ *依据全局状态*ω* 而定。

因为*x* 是一元连续的，选取正态分布。

正态分布的参数均值、方差为*ω* 的函数。

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-21_18-09-15.png?raw=true)

通过对似然函数 *Pr(x|ω)* 建模来分类(生成模型)。

a) 选取一正态分布来表示数据 *x0*。

b) 令该正态分布的参数 *{μ, σ2}* 为全局状态*w* 的函数。在实际情况下，这意味着在全局状态 *ω=0*  时使用一组均值和方差，在 *ω=1*  时使用另一组均值和方差。学习算法根据训练样本对 ![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-21_18-13-42.png?raw=true) 来 拟合参数 

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-21_18-13-08.png?raw=true)。

c) 也将全局状态 *ω*  的先验概率建模为参数为 *λ*  的伯努利分布。

d) 在推理时，选取一新数据 *x* 并根据贝叶斯法则计算状态的后验 *Pr(ω|x)* 。

在实际情况下，这意味着当全局状态为 *ω=0* 时有一组参数![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-21_18-16-31.png?raw=true) ,当状态为 *ω=1*时有另一组不同的参数 ![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-21_18-17-40.png?raw=true) ，因此

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-21_18-05-55.png?raw=true)

因为它们对每种类别数据的概率密度进行了建模，将其称为**类条件密度函数**。

**在学习中**，分别针对*ω=0* 和*ω=1* 的条件，分别拟合两个分布函数*Pr(x|ω=0)* 和*Pr(x|ω=1)* 的参数。

通过训练全局状态*ω* 学习*Pr(w|x)* 的参数*λ* 。

**推理**，

![](https://github.com/fantasy995/ComputerVision/blob/main/images/Snipaste_2020-10-21_18-35-49.png?raw=true)

我们需要让*Pr(ω=0|x)* 与*Pr(ω=1|x)* 的和为1。

#### 总结

对比这两种模型，判别模型的推理明显更加简单。生成模型需要使用贝叶斯方法计算后验概率。

书中提到：

适用生成模型：

>1、对似然 *Pr(x|ω)* 建模反映了数据实际是怎样产生的 ; 全局状态通过某些物理过程产生了观测数据(通常光来自于光源，与物体相互作用并被相机捕获)。如果我们想建立关于模型中生成过程的信息，该方法更适合。例如，我们可以考虑诸如透视投影和遮挡等现象。使用其他方法很难发觉这种知识：本质上，我们需要从数据中重新学习这些现象。
>
>2、在一些情况下，训练或测试数据向量 *x* 中的某些部分可能丢失。在这里，首选生成模型。它们对在所有数据维度上的联合分布建模，并能有效地插人丢失的元素。
>
>3、生成模型的一个基本特性是它允许以先验的方式合并专家知识。在判别模型中很难以主要的方式施加先验知识。

>值得注意的是，生成模型在视觉应用中更加普遍。因此，本书剩下大部分章节中主要讨论的是生成模型。

在判别模型中，我们只需要选定*ω* 的分布，分布参数由*x* 决定。

而在生成模型中，我们需要选定*x* 的分布*Pr1*，*Pr1*的参数由*ω* 决定。

再选定*ω* 的分布*Pr2*，*Pr2*的参数通过拟合数据样本中*ω* 分布情况得到。

由于*Pr1(x)* 与*ω* 有关，因此记为*Pr(x|ω)* ，*Pr2(ω)* 可以直接记为*Pr(ω)*。

