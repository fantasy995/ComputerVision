import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def fit_blr(X, W, var_prior, X_test):
    pass

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
    x_y = np.zeros((granularity*granularity, 2))
    x_y[:, 0] = X.reshape(granularity*granularity)
    x_y[:, 1] = Y.reshape(granularity*granularity)

    # ��������ֲ�ģ��
    mu_prior = [0, 0]
    var_prior = 1
    # ���η���
    conv_prior = var_prior * np.identity(2)
    # ���������������ֲ�
    pdf_prior = stats.multivariate_normal.pdf(x_y, mu_prior, conv_prior)
    pdf_prior = pdf_prior.reshape(granularity, granularity)
    # ����ͼ���ӻ�
    plt.title("prior pdf")
    plt.imshow(pdf_prior, extent=[start, top, start, top], cmap=plt.cm.hot)

    # ����ѵ������
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

    # ���ģ�Ͳ���


