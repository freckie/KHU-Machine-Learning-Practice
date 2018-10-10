import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]


# our model for the forward pass
def forward(x):
    return x * w


# 비용 함수
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


w_list = []     # 가중치 리스트
mse_list = []   # MSE 리스트

# w : 0.0 ~ 4.0 까지 0.1 단위
for w in np.arange(0.0, 4.1, 0.1):
    print("w =", w)
    l_sum = 0
    # x, y의 값
    for x_val, y_val in zip(x_data, y_data):
        y_pred_val = forward(x_val)  # 예측 함수 값
        l = loss(x_val, y_val)  # loss 값
        l_sum += l
        print("\t", x_val, y_val, y_pred_val, l)
    print("MSE =", l_sum / 3)
    w_list.append(w)  # 계산한 w를 수집
    mse_list.append(l_sum / 3)  # 계산한 MSE를 수집

# 그래프 출력
plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')
plt.show()