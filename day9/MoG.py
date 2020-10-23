import numpy as np
import matplotlib.pyplot as plt
import math

size = 1000
np.random.seed(0)
x = np.random.randn(size)
x = np.sort(x)

# x的分布情况
plt.scatter(x, x)
plt.show()

mean1, sigma1, w1 = 0, 1, 0.2
mean2, sigma2, w2 = 1, 2, 0.4
mean3, sigma3, w3 = 3, 1, 0.4


def normal_distribution(x, mean, sigma):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)

y1 = normal_distribution(x, mean1, sigma1)
y2 = normal_distribution(x, mean2, sigma2)
y3 = normal_distribution(x, mean3, sigma3)

plt.plot(x, y1, 'r', label='p1')
plt.plot(x, y2, 'g', label='p2')
plt.plot(x, y3, 'b', label='p3')
plt.legend()
plt.show()

y = np.array([ (normal_distribution(x[i], mean1, sigma1) * w1) +
      (normal_distribution(x[i], mean2, sigma2) * w2) +
      (normal_distribution(x[i], mean3, sigma3) * w3) for i in range(size)])
y = (normal_distribution(x, mean1, sigma1) * w1) + (normal_distribution(x, mean2, sigma2) * w2) +(normal_distribution(x, mean3, sigma3) * w3)
# 广播特性 两个方式得到的y相同

plt.title('MoG')
plt.plot(x, y)
plt.show()
