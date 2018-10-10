import torch
from torch.autograd import Variable

# x, y는 모두 tensor
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


# torch의 NN을 사용한 모델
class Model(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear module
        """
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(1, 1)  # 입력노드 : 1, 출력노드 : 1

    # 예측 함수
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        y_pred = self.linear(x)
        return y_pred


# our model
model = Model()

# 비용 함수
criterion = torch.nn.MSELoss(size_average=False)
# 학습률을 0.01로 지정한 SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 학습 루프
for epoch in range(500):
    # 주어진 x를 예측 함수에 전달해 얻은 예측 값
    y_pred = model(x_data)

    # 비용 함수 계산
    loss = criterion(y_pred, y_data)
    print(epoch, loss.data[0])

    optimizer.zero_grad()  # gradient 초기화
    loss.backward()  # 역전파
    optimizer.step()  # 가중치 갱신

# After training
hour_var = Variable(torch.Tensor([[4.0]]))
y_pred = model(hour_var)
print("predict (after training)", 4, model(hour_var).data[0][0])