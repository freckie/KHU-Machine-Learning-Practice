import random
import numpy as np


def trans(x, y):
    xp = x / (2 * x * y + 0.5)
    yp = y / (2 * x * y + 0.5)
    return xp, yp


def loss(X, Y, weight):
    return np.where(X @ weight >= 0, 1, 0) - Y


def cost(X, Y, weight):
    loss_val = loss(X, Y, weight)
    return np.sum(loss_val.T @ loss_val) / X.shape[0]


def delta_weight(X, Y, weight):
    return X.T @ loss(X, Y, weight)


def prepare_data(x11, x12, x21, x22):
    repeat = 200

    x1 = x11 + x21
    x2 = x12 + x22

    X = np.column_stack((x1, x2))
    bias = np.ones((X.shape[0], 1))
    X = np.concatenate((bias, X), axis=1)
    Y = np.zeros(shape=(X.shape[0], 1))
    Y[:repeat*2] = 0
    Y[repeat*2:] = 1

    return X, Y


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

print('--- data loaded ---')
