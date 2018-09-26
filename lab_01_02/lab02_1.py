import sys
import math
import random
import numpy as np
import matplotlib.pyplot as plt

from for_lab02 import *


def algorithm1(X, Y, r_weights):
    min = sys.maxsize
    min_weights = -1

    for i in range(random_number):
        temp_cost = cost(X, Y, r_weights[i])
        if temp_cost < min:
            min = temp_cost
            min_weights = r_weights[i]

    return min_weights


random_number = 1000
r_weights = [4*np.random.random((3, 1)) - 2 for i in range(random_number)]

X, Y = prepare_data(x11, x12, x21, x22)
min_weight = algorithm1(X, Y, r_weights)

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
