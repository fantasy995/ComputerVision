import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image

# x[i] : (2,1)
# mean : (1,2)
# 二元正态分布 概率密度函数
def normal_distribution_2d(x, mean, sigma):
    return np.array([np.exp(-0.5 * np.dot(
        np.dot(x[i].T - mean, np.linalg.inv(sigma))
        , (x[i].T - mean).T)) / (
                    math.pow(2 * np.pi, 2 / 2) * math.sqrt(np.linalg.det(sigma))) for i in range(int(x.size / 2))])


size = 1000
mean = np.array([[1, 5]])
print(mean.shape)
sigma = np.array([[10, 5], [5, 5]])

np.random.seed(0)
s = np.random.multivariate_normal(mean[0], sigma, size)
print(s.shape)  # (1000, 2) x1一列 x2一列

data_test = np.array([
    [1, 2],
    [1, 2],
    [1, 2]
])

# *s.T提取参数x和y
# [‘solid’ | ‘dashed’, ‘dashdot’, ‘dotted’ | (offset, on-off-dash-seq)
# | '-' | '--' | '-.' | ':' | 'None' | ' ' | '']
plt.plot(*s.T, '.')
plt.axis('scaled')
# plt.show()

x1_min = np.min(s[:, 0])
x1_max = np.max(s[:, 0])
x2_min = np.min(s[:, 1])
x2_max = np.max(s[:, 1])
print('x1_min:', x1_min)
print('x1_max:', x1_max)
print('x2_min:', x2_min)
print('x2_max:', x2_max)

data_x1 = np.linspace(x1_min, x1_max, 100)
print(data_x1[0], data_x1[99])
data_x2 = np.linspace(x2_min, x2_max, 100)
print(data_x2[0], data_x2[99])
data = np.zeros(shape=(100 * 100, 2))
data = data[:, :, np.newaxis]
for i in range(100):
    data[i*100:(i+1)*100, 0] = data_x1[i]
    data[i*100:(i+1)*100, 1] = data_x2.reshape(100, 1)
pre = normal_distribution_2d(data, mean, sigma)
pre = pre.reshape(100, 100)
pre = np.rot90(pre, 1)

# 转为灰度图
pre = pre * (255 / np.max(pre))
print(np.max(pre), np.min(pre))
# im = Image.fromarray(pre)
# im = im.convert('L')
# im.show()

pre2 = normal_distribution_2d(data, [[9,7]], [[1,0],[0,1]])
pre2 = pre2.reshape(100,100)
pre2 = np.rot90(pre2,1)
pre2 = pre2 * (255 / np.max(pre2))
im = Image.fromarray((pre+pre2)/2)
im = im.convert('L')
im.show()


data_test=np.array([[[1]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]],[[8]],[[9]]])
data_test = data_test.reshape(3, 3)
data_test = np.rot90(data_test,1)
print(data_test)


