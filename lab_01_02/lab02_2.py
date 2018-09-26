import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from for_lab02 import *

target_cost = 0


def algorithm2(X, Y, target_cost):
    while True:
        weight = np.random.random((3, 1)) - 0.5
        if cost(X, Y, weight) <= target_cost:
            return weight


X, Y = prepare_data(x11, x12, x21, x22)
min_weight = algorithm2(X, Y, target_cost)

plt.plot(x11, x12, ".")
plt.plot(x21, x22, ".")
plt.xlabel('x1 axis')
plt.ylabel('x2 axis')

b, w1, w2 = min_weight
line_x_coords = [0, -b/w1]
line_y_coords = [-b/w2, 0]
print('y = {}x + {}'.format(-w1/w2, -b/w1))
plt.plot(line_x_coords, line_y_coords, linewidth=2)
plt.show()
