import torch
import numpy as np
import matplotlib.pyplot as plt


# x: 数据
# k: 正态分布的数量
# precision: 精度 什么时候停止拟合
def fix_mod(X, K, precision):
    D = X[0].shape[0]

    # w[i] 每个正态分布的权重
    w = np.zeros(K).reshape(K, 1)
    mean = np.random.rand(K).reshape(K, 1)
    sigma = [np.random.rand(D * D).reshape(D, D) for i in range(K)]



if __name__ == '__main__':
    x = np.random.multivariate_normal([1, 2], [[1, 0], [0, 1]], 5000)
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()
    fix_mod(x, 10, 0.01)
