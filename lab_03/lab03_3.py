import torch
from torch.autograd import Variable

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

# random value를 위한 torch의 tensor
w = Variable(torch.Tensor([1.0]), requires_grad=True)


# our model forward pass
def forward(x):
    return x * w


# 비용 함수
def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) * (y_pred - y)


# 학습 전
print("predict (before training)", 4, forward(4).data[0])

# 학습 루프
for epoch in range(10):
    # x, y 데이터
    for x_val, y_val in zip(x_data, y_data):
        l = loss(x_val, y_val)  # 비용 함수 계산
        l.backward()  # 역전파
        print("\tgrad: ", x_val, y_val, w.grad.data[0])

        w.data = w.data - 0.01 * w.grad.data  # w 갱신
        w.grad.data.zero_()  # w 갱신 후 gradient를 초기화
    print("progress:", epoch, l.data[0])

# After training
print("predict (after training)", 4, forward(4).data[0])