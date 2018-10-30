# References
# https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/01-basics/pytorch_basics/main.py
# http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#dataset-class

import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader


# 데이터셋
class DiabetesDataset(Dataset):
    def __init__(self):
        # diabete 데이터 로드.
        xy = np.loadtxt('./data/diabetes.csv.gz', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# 모델
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 6-4 모델
        self.l1 = torch.nn.Linear(8, 6)
        self.l2 = torch.nn.Linear(6, 4)
        self.l3 = torch.nn.Linear(4, 1)
        # 활성 함수로 Sigmoid 설정
        self.sigmoid = torch.nn.Sigmoid()

    # 예측 함수
    def forward(self, x):
        # y = o(o(x))
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred


def main():
    # 데이터셋 로드
    dataset = DiabetesDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

    # 모델 인스턴스화
    model = Model()

    # loss로 BCE 사용
    criterion = torch.nn.BCELoss(size_average=True)
    # 초기화로 SGD 사용, 학습률 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # 학습 메인 루프
    for epoch in range(2):
        for i, data in enumerate(train_loader, 0):
            # x, y 가져와서 Variable로 만들기
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels)

            # y 예측
            y_pred = model(inputs)

            # loss 계산 및 출력
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.data[0])

            # gradient 초기화
            optimizer.zero_grad()
            # 오류 역전파
            loss.backward()
            # 가중치 갱신
            optimizer.step()


if __name__ == '__main__':
    main()