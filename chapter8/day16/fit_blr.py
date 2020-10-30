import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import torch
import math


# 似然函数分布
# X -> w
def multivariate_normal_pdf(X: torch, mu: torch.Tensor, sigma: torch.Tensor):
    batch_size = X.shape[0]
    D = X.shape[1]
    p1 = 1. / (math.pow(2 * np.pi, D / 2) * math.sqrt(torch.det(sigma)))
    p2 = -0.5 * torch.mm(torch.mm((X - mu).reshape(batch_size, D), torch.inverse(sigma)),
                         (X - mu).reshape(D, batch_size))
    return p1 * torch.exp(p2)


def fit_blr_cost(var: torch.Tensor, X: torch.Tensor, W: torch.Tensor, var_prior, var_world):
    last_grad = 10000
    last_var = var
    optimizer = torch.optim.SGD([var], lr=0.001)
    # 求似然球形方差参数
    while last_grad > 0.05:
        if last_var > var:
            last_var = var
        if var > var_world or var < 0:
            break
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


def fit_blr(X, W, var_prior, X_test):
    D = X.shape[0] - 1
    batch_size = X.shape[1]
    batch_size_test = X_test.shape[1]

    mu_world = np.sum(W) / batch_size
    var_wolrd = np.sum(np.power(W - mu_world, 2))
    # var = fit_blr_cost(torch.tensor(0.6817, requires_grad=True)
    #                    , torch.tensor([[1, 1, 1, 1, 1, 1], [-4, -1, -1, 0, 1, 3.5]])
    #                    , torch.tensor([[4.5], [3], [2], [2.5], [2.5], [0]])
    #                    , torch.tensor(6))

    var = var_wolrd / 2
    print(var_wolrd)
    var = fit_blr_cost(torch.tensor(var, requires_grad=True),
                       torch.as_tensor(X),
                       torch.as_tensor(W),
                       torch.as_tensor(var_prior),
                       var_wolrd)
    print(var)
    var = var.detach().numpy()
    # var = 0.37 # 测试用 拟合时间过长

    if D < batch_size:
        A_inv = np.linalg.inv((np.dot(X, X.T))) / var + np.identity(D+1) / var_prior
    else:
        A_inv = np.identity(D+1) - np.dot(X, np.linalg.inv(np.dot(X.T, X) + (var / var_prior) * np.identity(batch_size), X.T))
        A_inv = var_prior * A_inv

    temp = np.dot(X_test.T, A_inv) # 100 * 2
    mu_test = np.dot(np.dot(temp, X), w) / var

    var_test = np.tile(var, batch_size_test).reshape(batch_size_test, 1)
    for i in range(batch_size_test):
        var_test[i] = var_test[i] + np.dot(temp[i].reshape(1,-1), X_test[:, i].reshape(D+1,1))

    return mu_test, var_test, var, A_inv


if __name__ == '__main__':
    # X, Y = np.mgrid[1:4:4j, 1:3:3j]
    # x = X.reshape(12,1)
    # y = Y.reshape(12,1)
    # data = np.zeros((12,2))
    # data[:, 0] = x.reshape(12)
    # data[:, 1] = y.reshape(12)
    # print(data)

    granularity = 100
    start, top = -1, 1
    domain = np.linspace(start, top, granularity)

    X, Y = np.mgrid[start:top:100j, start:top:100j]
    x_y = np.zeros((granularity * granularity, 2))
    x_y[:, 0] = X.reshape(granularity * granularity)
    x_y[:, 1] = Y.reshape(granularity * granularity)

    # 构建先验分布模型
    mu_prior = [0, 0]
    var_prior = 1
    # 球形方差
    conv_prior = var_prior * np.identity(2)
    # 计算参数Φ的先验分布
    pdf_prior = stats.multivariate_normal.pdf(x_y, mu_prior, conv_prior)
    pdf_prior = pdf_prior.reshape(granularity, granularity)
    # 热力图可视化
    # plt.title("prior pdf")
    # plt.imshow(pdf_prior, extent=[start, top, start, top], cmap=plt.cm.hot)

    # 构造训练样本
    batch_size = 7
    X_train = np.zeros((batch_size, 2))
    X_train[:, 0] = 1
    X_train[:, 1] = np.array([-0.7, -0.5, -0.41, 0, 0.12, 0.3, 0.6]).reshape(batch_size)
    phi = np.array([[2], [-0.7]])
    X_train_phi = np.dot(X_train, phi)
    r = np.random.randn(batch_size, 1)
    w = np.zeros((batch_size, 1))
    for i in range(batch_size):
        w[i] = X_train_phi[i] + r[i]

    X_test = np.linspace(start, top, granularity).reshape(granularity, 1)
    X_test = np.array([np.ones((granularity, 1)), X_test]).reshape(2, granularity)

    # 获得模型参数
    mu_test, var_test, var, A_inv = fit_blr(X_train.T, w, var_prior, X_test)

    # 后验分布可视化
    mu_2 = (np.dot(np.dot(A_inv, X_train.T), w) / 0.37).T
    mu_2 = mu_2.reshape(2)
    print(mu_2)
    covariance_2 = A_inv
    mvpdf_2 = stats.multivariate_normal.pdf(x_y, mu_2, covariance_2)
    mvpdf_2 = mvpdf_2.reshape(granularity, granularity)
    plt.title("posterior pdf")
    plt.imshow(mvpdf_2, extent=[start, top, start, top], cmap=plt.cm.hot)

