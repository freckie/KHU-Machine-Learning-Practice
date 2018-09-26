import sys
import math
import time
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt

from for_lab02 import *

target_cost = 0
learning_rate = 0.0001


def algorithm3(X, Y, learning_rate, target_cost):
    weight = np.random.random((3, 1))

    while True:
        weight = weight - learning_rate * delta_weight(X, Y, weight)/3
        co = cost(X, Y, weight)
        if co <= target_cost:
            return weight


X, Y = prepare_data(x11, x12, x21, x22)

start = datetime.datetime.now()
min_weight = algorithm3(X, Y, learning_rate, target_cost)
end = datetime.datetime.now()
print('Process Time :', end - start)

plt.plot(x11, x12, ".")
plt.plot(x21, x22, ".")
plt.xlabel('x1 axis')
plt.ylabel('x2 axis')

b, w1, w2 = min_weight
line_x_coords = [0, -b/w1]
line_y_coords = [-b/w2, 0]
print(min_weight, line_x_coords, line_y_coords)
print('y = {}x + {}'.format(-w1/w2, -b/w1))
plt.plot(line_x_coords, line_y_coords, linewidth=2)
plt.show()
