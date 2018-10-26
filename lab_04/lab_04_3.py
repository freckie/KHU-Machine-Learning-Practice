import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable

# 데이터 로딩 (csv 타입)
xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
# x 데이터와 y 데이터 로딩.
x_data = Variable(torch.from_numpy(xy[:, 0:-1]))
y_data = Variable(torch.from_numpy(xy[:, [-1]]))
print(x_data.data.shape)
print(y_data.data.shape)


class Model_1(torch.nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()

        # 6-6-6-4-4-4
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 6)
        self.l3 = torch.nn.Linear(6, 6)
        self.l4 = torch.nn.Linear(6, 4)
        self.l5 = torch.nn.Linear(4, 4)
        self.l6 = torch.nn.Linear(4, 4)
        self.l7 = torch.nn.Linear(4, 1)

        # 활성함수로 sigmoid 함수 설정.
        self.actv = torch.nn.Softmax()

    # 예측 함수.
    def forward(self, x):
        out1 = self.actv(self.l1(x))
        out2 = self.actv(self.l2(out1))
        out3 = self.actv(self.l3(out2))
        out4 = self.actv(self.l4(out3))
        out5 = self.actv(self.l5(out4))
        out6 = self.actv(self.l6(out5))
        y_pred = self.actv(self.l7(out6))
        return y_pred


class Model_2(torch.nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()

        # 6-6-4-4
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 6)
        self.l3 = torch.nn.Linear(6, 4)
        self.l4 = torch.nn.Linear(4, 4)
        self.l5 = torch.nn.Linear(4, 1)

        # 활성함수로 sigmoid 함수 설정.
        self.actv = torch.nn.Softmax()

    # 예측 함수.
    def forward(self, x):
        out1 = self.actv(self.l1(x))
        out2 = self.actv(self.l2(out1))
        out3 = self.actv(self.l3(out2))
        out4 = self.actv(self.l4(out3))
        y_pred = self.actv(self.l5(out4))
        return y_pred


class Model_3(torch.nn.Module):
    def __init__(self):
        super(Model_3, self).__init__()

        # 8-6
        self.l1 = torch.nn.Linear(8, 8)
        self.l2 = torch.nn.Linear(8, 6)
        self.l3 = torch.nn.Linear(6, 1)

        # 활성함수로 sigmoid 함수 설정.
        self.actv = torch.nn.Softmax()

    # 예측 함수.
    def forward(self, x):
        out1 = self.actv(self.l1(x))
        out2 = self.actv(self.l2(out1))
        y_pred = self.actv(self.l3(out2))
        return y_pred

class Model_4(torch.nn.Module):
    def __init__(self):
        super(Model_4, self).__init__()

        # 4-2
        self.l1 = torch.nn.Linear(8, 4)
        self.l2 = torch.nn.Linear(4, 2)
        self.l3 = torch.nn.Linear(2, 1)

        # 활성함수로 sigmoid 함수 설정.
        self.actv = torch.nn.Softmax()

    # 예측 함수.
    def forward(self, x):
        out1 = self.actv(self.l1(x))
        out2 = self.actv(self.l2(out1))
        y_pred = self.actv(self.l3(out2))
        return y_pred


class Model_5(torch.nn.Module):
    def __init__(self):
        super(Model_5, self).__init__()

        # 8 (single)
        self.l1 = torch.nn.Linear(8, 8)
        self.l2 = torch.nn.Linear(8, 1)

        # 활성함수로 sigmoid 함수 설정.
        self.actv = torch.nn.Softmax()

    # 예측 함수.
    def forward(self, x):
        out1 = self.actv(self.l1(x))
        y_pred = self.actv(self.l2(out1))
        return y_pred


class Model_6(torch.nn.Module):
    def __init__(self):
        super(Model_6, self).__init__()

        # 6 (single)
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 1)

        # 활성함수로 sigmoid 함수 설정.
        self.actv = torch.nn.Softmax()

    # 예측 함수.
    def forward(self, x):
        out1 = self.actv(self.l1(x))
        y_pred = self.actv(self.l2(out1))
        return y_pred


class Model_7(torch.nn.Module):
    def __init__(self):
        super(Model_7, self).__init__()

        # 4 (single)
        self.l1 = torch.nn.Linear(8, 4)
        self.l2 = torch.nn.Linear(4, 1)

        # 활성함수로 sigmoid 함수 설정.
        self.actv = torch.nn.Softmax()

    # 예측 함수.
    def forward(self, x):
        out1 = self.actv(self.l1(x))
        y_pred = self.actv(self.l2(out1))
        return y_pred


class Model_8(torch.nn.Module):
    def __init__(self):
        super(Model_8, self).__init__()

        # 2 (single)
        self.l1 = torch.nn.Linear(8, 2)
        self.l2 = torch.nn.Linear(2, 1)

        # 활성함수로 sigmoid 함수 설정.
        self.actv = torch.nn.Softmax()

    # 예측 함수.
    def forward(self, x):
        out1 = self.actv(self.l1(x))
        y_pred = self.actv(self.l2(out1))
        return y_pred

# model = Model()

# # loss로 BCE 사용.
# criterion = torch.nn.BCELoss(size_average=True)
# # 최적화 방식으로 SGD 사용, 학습률=0.1.
# optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# # 학습 메인 루프.
# loss_list = list()
# for epoch in range(100):
#     # x를 생성한 모델을 통해 예측.
#     y_pred = model(x_data)
#     # loss를 계산.
#     loss = criterion(y_pred, y_data)
#     loss_list.append(loss.data[0])
#     print(epoch, loss.data[0])
#     # gradient 초기화.
#     optimizer.zero_grad()
#     # 오류 역전파.
#     loss.backward()
#     # weight 갱신.
#     optimizer.step()

# plt.plot(range(100), loss_list)
# plt.xlabel('iteration')
# plt.ylabel('loss')
# plt.show()