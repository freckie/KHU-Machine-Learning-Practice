import sys
import math
import time
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt

from for_lab02 import *

batch_size = 30
target_cost = 0
learning_rate = 0.0001


def algorithm4(X, Y, learning_rate, target_cost, batch_size):
    weight = np.random.random((3, 1))

    while True:
        idx = [i for i in range(len(X))]
        random.shuffle(idx)

        for j in range(len(X)):
            batch_X = np.array([X[k] for k in idx[:batch_size]])
            batch_Y = np.array([Y[l] for l in idx[:batch_size]])
            weight = weight - learning_rate * delta_weight(batch_X, batch_Y, weight)/3

        if cost(X, Y, weight) <= target_cost:
            return weight


X, Y = prepare_data(x11, x12, x21, x22)

start = datetime.datetime.now()
min_weight = algorithm4(X, Y, batch_size, target_cost, batch_size)
end = datetime.datetime.now()
print('Process Time :', end - start)

plt.plot(x11, x12, ".")
plt.plot(x21, x22, ".")
plt.xlabel('x1 axis')
plt.ylabel('x2 axis')

b, w1, w2 = min_weight
line_x_coords = [0, (-b/w1)]
line_y_coords = [(-b/w2), 0]
print('min_weight : ', min_weight)
print('y = {}x + {}'.format(-w1/w2, -b/w1))
print('coords : ({}, {}), ({}, {})'.format(line_x_coords[0], line_x_coords[1], line_y_coords[0], line_y_coords[1]))
plt.plot(line_x_coords, line_y_coords, linewidth=2)
plt.show()
