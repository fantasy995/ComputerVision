import numpy as np
import math
import matplotlib.pyplot as plt

def normal_distribution_2d(x, mean, sigma):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)

def fun_test(p1, p2):
    print(p1.shape)
    print(p2.shape)


def fun_test1(p1, p2):
    print(p1)
    print(p2)


def fun_test2(data):
    print(data[0])

sampleNo = 1000
mean = np.array([[1, 5]])
Sigma = np.array([[10, 5], [5, 5]])

np.random.seed(0)
s = np.random.multivariate_normal(mean[0], Sigma, sampleNo)
print(s.shape)  # (1000, 2) x1一列 x2一列
# fun_test(*s.T)
'''
(1000,)
(1000,)
'''
data_test = np.array([
    [1, 2],
    [1, 2],
    [1, 2]
])
print(data_test.shape)  # (3, 2)
fun_test1(*data_test.T)
'''
[1 1 1]
[2 2 2]
'''
fun_test2(data_test)  # [1 2]


# *s.T提取参数x和y
plt.plot(*s.T, '.')
plt.axis('scaled')
# plt.show()

pre = normal_distribution_2d(s, )


