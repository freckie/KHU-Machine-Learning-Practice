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


class Model(torch.nn.Module):
    def __init__(self):
        """
        모델 초기화 및 활성함수 설정.
        """
        super(Model, self).__init__()

        # 3개의 layer를 세팅. (8->6->1)
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)

        # 활성함수로 relu 함수 설정.
        self.actv = torch.nn.ReLU()

    # 예측 함수.
    def forward(self, x):
        out1 = self.actv(self.l1(x))
        out2 = self.actv(self.l2(out1))
        y_pred = self.actv(self.l3(out2))
        return y_pred


model = Model()

# loss로 BCE 사용.
criterion = torch.nn.BCELoss(size_average=True)
# 최적화 방식으로 SGD 사용, 학습률=0.1.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 학습 메인 루프.
loss_list = list()
for epoch in range(100):
    # x를 생성한 모델을 통해 예측.
    y_pred = model(x_data)
    # loss를 계산.
    loss = criterion(y_pred, y_data)
    loss_list.append(loss.data[0])
    print(epoch, loss.data[0])
    # gradient 초기화.
    optimizer.zero_grad()
    # 오류 역전파.
    loss.backward()
    # weight 갱신.
    optimizer.step()

plt.plot(range(100), loss_list)
plt.xlabel('iteration')
plt.ylabel('loss')
plt.show()