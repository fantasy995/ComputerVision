import numpy as np
import torch
from scipy import stats
import matplotlib.pyplot as plt
from PIL import Image


# w : Φ
# X: shape (N,D)
# 根据偏导公式，可以直接求出极值点对应的参数
def fit_lr(X, w):
    phi = np.dot(np.dot(np.linalg.inv(np.dot(X, X.T)), X), w)
    # 正态分布均值 : phi[1, :]

    batch_size = X.shape[1]
    temp = w - np.dot(X.T, phi)
    sigma = np.dot(temp.T, temp) / batch_size

    return phi, sigma


if __name__ == '__main__':
    batch_size = 50
    x = np.linspace(1, 10, batch_size).reshape(batch_size, 1)
    # 初始化参数
    phi = np.array([[7], [-0.5]])
    sigma = 0.6
    w = np.zeros((batch_size, 1))
    X = np.array([np.ones((batch_size, 1)), x]).T
    X = X.squeeze()
    X_phi = np.dot(X, phi)

    # 建立一个自定义的状态 ω 与 x 的线性关系
    r = np.random.randn(batch_size, 1)
    for i in range(batch_size):
        mu = X_phi[i]
        w[i] = mu + r[i]

    fit_phi, fit_sigma = fit_lr(X.T, w)
    print(fit_phi)
    print(fit_sigma)
    # 验证
    domain = np.linspace(0, 10, 1000).reshape(1000, 1)
    temp = np.array([np.ones((1000, 1)), domain]).T
    temp = temp.squeeze()
    domain_phi = np.dot(temp, fit_phi)

    # pre = stats.norm.pdf(X, X_phi[:, 1], fit_sigma)
    pre = np.zeros((1000, 1000))
    for j in range(1000):
        mu = domain_phi[j]
        for i in range(1000):
            domain_i = domain[i]
            pre[i, j] = stats.norm.pdf(domain_i, mu, fit_sigma)

    print(np.amax(pre))
    print(np.amin(pre))
    im_arr = pre * (255 / np.amax(pre))
    im_arr = np.rot90(im_arr, 1)
    im = Image.fromarray(im_arr)
    im = im.convert('L')
    im.show()

    X, Y = np.mgrid[0:10:1000j, 0:10:1000j]
    f = np.reshape(pre.T, X.shape)
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('x')
    ax.set_ylabel('w')
    ax.imshow(np.rot90(f), cmap=plt.cm.hot, extent=[0, 10, 0, 10])
    plt.scatter(x, w)

    plt.show()

