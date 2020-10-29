### 今日工作

1、继续实现贝叶斯线性回归的拟合算法



### 贝叶斯线性回归拟合算法

1、基于PyTorch的多元正态分布概率密度计算函数。

```python
def multivariate_normal_pdf(X: torch, mu: torch.Tensor, sigma: torch.Tensor):    batch_size = X.shape[0]    
    D = X.shape[1]    
    p1 = 1. / (math.pow(2 * np.pi, D / 2) * math.sqrt(torch.det(sigma)))    
    p2 = -0.5 * torch.mm(torch.mm((X - mu).reshape(batch_size, D), torch.inverse(sigma)),                         (X - mu).reshape(D, batch_size))    
    return p1 * torch.exp(p2)
```

2、基于PyTorch框架进行先验分布的参数修正。

```python
def fit_blr_cost(var: torch.Tensor, X: torch.Tensor, W: torch.Tensor, var_prior):    last_grad = 10000    
    optimizer = torch.optim.SGD([var], lr=0.001)    
    while last_grad > 0.05:        
        # print('var:', var)        
        batch_size = X.shape[1]        
        covariance = var_prior * torch.mm(X.T, X) + torch.pow(torch.sqrt(var), 2) * torch.eye(batch_size)        
        # covariance.requires_grad_()        
        # covariance.retain_grad()        
        W = W.reshape(1, batch_size)        
        f = multivariate_normal_pdf(W, torch.zeros((batch_size), requires_grad=False), covariance)        
        f = -torch.log(f)        
        optimizer.zero_grad()        
        f.backward()        
        last_grad = math.fabs(var.grad)        
        # print(last_grad)        
        optimizer.step()    
 return var
```

3、后续参数的求取比较简单，根据公式计算即可。

4、可视化结果验证算法。

这一步还未完成，明天应该可以贝叶斯线性回归拟合算法的全部内容。