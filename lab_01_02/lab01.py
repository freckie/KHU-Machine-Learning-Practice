import math
import random
import numpy as np
import matplotlib.pyplot as plt


def trans(x, y):
    xp = x / (2 * x * y + 0.5)
    yp = y / (2 * x * y + 0.5)
    return xp, yp


repeat = 200

x11 = [random.uniform(0, 0.2) for i in range(repeat)]
x12 = [random.uniform(0, 0.2) for i in range(repeat)]
x11 += [random.uniform(0.4, 0.6) for i in range(repeat)]
x12 += [random.uniform(0.4, 0.6) for i in range(repeat)]

x21 = [random.uniform(0, 0.2) for i in range(repeat)]
x22 = [random.uniform(0.8, 1) for i in range(repeat)]
x21 += [random.uniform(0.8, 1) for i in range(repeat)]
x22 += [random.uniform(0, 0.2) for i in range(repeat)]


for i in range(len(x11)):
    x11[i], x12[i] = trans(x11[i], x12[i])
    x21[i], x22[i] = trans(x21[i], x22[i])

plt.plot(x11, x12, ".")
plt.plot(x21, x22, ".")
plt.plot([1.1, 0], [0, 1.1], "-")
plt.show()

