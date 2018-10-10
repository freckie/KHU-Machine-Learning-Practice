import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]
first_w = 1
w = 1  # a random guess: random value
lr = 0.01  # learning rate


# our model forward pass
def forward(x):
    return x * w


# Loss function
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# compute gradient
def gradient(x, y):  # d_loss/d_w
    return 2 * x * (x * w - y)


# Before training
print("predict (before training)", 4, forward(4))

# Training loop
w_list = list()
l_list = list()
for epoch in range(10):
    for x_val, y_val in zip(x_data, y_data):
        grad = gradient(x_val, y_val)
        w = w - lr * grad
        print("\tgrad: ", x_val, y_val, round(grad, 2))
        l = loss(x_val, y_val)
    w_list.append(round(w, 2))
    l_list.append(round(l, 2))
    print("progress:", epoch, "w=", round(w, 2), "loss=", round(l, 2))

# After training
print("predict (after training)", "4 hours", forward(4))

# epoch-w 그래프 출력
plt.plot(range(10), w_list)
plt.title('w initialized by ' + str(first_w))
plt.ylabel('w')
plt.xlabel('epoch')
plt.show()

# epoch-loss 그래프 출력
plt.plot(range(10), l_list)
plt.title('w initialized by ' + str(first_w))
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
