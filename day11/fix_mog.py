import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
from PIL import Image

# x: 数据(N, D)
# k: 正态分布的数量
# precision: 精度 什么时候停止拟合
def fix_mod(X, K, precision):
    # 单个观察数据的维度
    D = X.shape[1]
    batch_size = X.shape[0]

    # w[i] 每个正态分布的权重
    w = np.zeros(K).reshape(K, 1)
    w[:, 0] = 1. / K

    # 随机选取K个位置，初始化每个D维正态分布的均值
    rand_idx = np.random.randint(0, batch_size, K)
    # print(rand_idx)
    mean = np.zeros(D * K).reshape(K, D)
    mean[0:K, :] = X[rand_idx, :]
    # print(mean)
    # print(X[rand_idx[0], 0], X[rand_idx[0], 1])
    # print(X[rand_idx[3], 0], X[rand_idx[3], 1])
    # print('******************end ini mu for every norm********************')

    # 初始K个D维正态分布的协方差矩阵为数据集的协方差矩阵
    data_set_mean = np.mean(X, axis=0)
    data_set_variance = np.zeros((D, D))
    for i in range(batch_size):
        mat = (X[i, :] - data_set_mean).reshape(1, D)
        mat = np.dot(mat.T, mat)
        data_set_variance = data_set_variance + mat
    data_set_variance = data_set_variance / batch_size
    sigma = np.array([data_set_variance for i in range(K)])
    # print('******************end ini sigma for every norm********************')

    # r = stats.multivariate_normal.pdf(X, mean[0], sigma[0])
    # print(r)
    # print(r.shape)

    # 迭代次数
    iteration = 0
    previous_L = 1000000

    # 迭代E步和M步拟合模型
    while True:
        l = np.array([w[k] * stats.multivariate_normal.pdf(X, mean[k], sigma[k]) for k in range(K)]).T
        s = np.sum(l, axis=1)
        # 标准化 求出r_ik 第i个数据对第k个分布的影响
        r = np.array([l[i, :] / s[i] for i in range(batch_size)])
        # E步结束

        r_sum_rows = np.sum(r, axis=0)
        # r_sum_rows[k]全部数据对第k个分布的影响
        r_sum_all = np.sum(r_sum_rows)

        # 更新每个正态分布的权重
        w = np.array([r_sum_rows[k] / r_sum_all for k in range(K)])

        # 更新每个正态分布的参数
        old_sigma = sigma.copy()
        for k in range(K):
            # 更新每个正态分布的mu
            new_mean = np.zeros((1, D))
            for i in range(batch_size):
                new_mean = new_mean + r[i, k] * X[i, :]
            mean[k, :] = new_mean / r_sum_rows[k]

            # 更新每个正态分布的sigma
            new_sigma = np.zeros((D, D))
            for i in range(batch_size):
                mat = (X[i] - mean[k]).reshape(1, 2)
                mat = r[i, k] * np.dot(mat.T, mat)
                new_sigma = new_sigma + mat
            sigma[k] = new_sigma / r_sum_rows[k]
        # 参数更新过程结束

        temp = np.array([w[k] * stats.multivariate_normal.pdf(X, mean[k], sigma[k]) for k in range(K)]).T
        temp = np.sum(temp, axis=1).reshape(batch_size, 1)
        temp = np.log(temp)
        L = sum(temp)

        if iteration+1 % 100 == 0:
            print('iteration:{}|L - previous_L:{}'.format(iteration, L - previous_L))

        iteration = iteration + 1
        # 达到极值点
        if math.fabs(L - previous_L) < precision:
            print(iteration)
            print(math.fabs(L - previous_L))
            break
        previous_L = L

    return w, mean, sigma


if __name__ == '__main__':
    x1 = np.random.multivariate_normal([1, 2], [[2, 0], [0, 0.5]], size=3500)
    x2 = np.random.multivariate_normal([1, 5], [[1, 0], [0, 1]], size=1500)

    x = np.append(x1, x2, axis=0)

    plt.plot(x[:, 0], x[:, 1], '.', color='green')
    plt.show()

    w, mean, sigma = fix_mod(x, 2, 0.0001)
    print('w:', w)
    print('mean:',mean)
    print('sigma:', sigma)
    # area = np.zeros((2,2))
    # area[0, :] = [np.min(x[:, 0]), np.max(x[:, 0])]
    # area[1, :] = [np.min(x[:, 1]), np.max(x[:, 1])]

    x1_min = np.min(x[:, 0])
    x1_max = np.max(x[:, 0])
    x2_min = np.min(x[:, 1])
    x2_max = np.max(x[:, 1])
    data_x1 = np.linspace(x1_min, x1_max, 100)
    data_x2 = np.linspace(x2_min, x2_max, 100)
    data = np.zeros(shape=(100 * 100, 2))
    for i in range(100):
        data[i * 100:(i + 1) * 100] = data_x1[i]
        data[i * 100:(i + 1) * 100] = data_x2.reshape(100, 1)
    pre = np.zeros((100,100))

    for i in range(2):
        temp = w[i] * stats.multivariate_normal.pdf(data, mean[i], sigma[i]).reshape(100, 100)
        pre = pre + temp
    pre = np.rot90(pre, 1)
    pre = pre * (255 / np.max(pre))
    im = Image.fromarray(pre)
    im = im.convert('L')
    im.show()