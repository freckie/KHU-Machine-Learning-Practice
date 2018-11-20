import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. conv 2개, dense 1개
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # convolusion
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # maxpool
        self.mp = nn.MaxPool2d(2)
        # dense layer (fully connected)
        self.fc = nn.Linear(320, 10)

    # 예측 함수
    def forward(self, x):
        in_size = x.size(0)
        # 컨볼루젼 1, 2 적용, 활성함수로 ReLU 사용
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1) # flatten the tensor
        x = self.fc(x)
        # log softmax 사용하여 리턴.
        return F.log_softmax(x)


# 2. conv 3개, dense 1개
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # convolusion
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3)
        # maxpool
        self.mp = nn.MaxPool2d(2)
        # dense layer (fully connected)
        self.fc = nn.Linear(30, 10)

    # 예측 함수
    def forward(self, x):
        in_size = x.size(0)
        # 컨볼루젼 1, 2 적용, 활성함수로 ReLU 사용
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))
        x = x.view(in_size, -1) # flatten the tensor
        x = self.fc(x)
        # log softmax 사용하여 리턴.
        return F.log_softmax(x)


# 2. conv 2개, dense 2개
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        # convolusion
        self.conv1 = nn.Conv2d(1, 10, kernel_size=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=1)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=1)
        self.conv4 = nn.Conv2d(30, 40, kernel_size=1)
        # maxpool
        self.mp = nn.MaxPool2d(2)
        # dense layer (fully connected)
        self.fc = nn.Linear(40, 10)

    # 예측 함수
    def forward(self, x):
        in_size = x.size(0)
        # 컨볼루젼 1, 2 적용, 활성함수로 ReLU 사용
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))
        x = F.relu(self.mp(self.conv4(x)))
        x = x.view(in_size, -1) # flatten the tensor
        x = self.fc(x)
        # log softmax 사용하여 리턴.
        return F.log_softmax(x)


# 2. conv 2개, dense 2개
class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        # convolusion
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # maxpool
        self.mp = nn.MaxPool2d(2)
        # dense layer (fully connected)
        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, 10)

    # 예측 함수
    def forward(self, x):
        in_size = x.size(0)
        # 컨볼루젼 1, 2 적용, 활성함수로 ReLU 사용
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1) # flatten the tensor
        x = self.fc2(self.fc1(x))
        # log softmax 사용하여 리턴.
        return F.log_softmax(x)


# 2. conv 2개, dense 2개
class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        # convolusion
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3)
        # maxpool
        self.mp = nn.MaxPool2d(2)
        # dense layer (fully connected)
        self.fc1 = nn.Linear(30, 20)
        self.fc2 = nn.Linear(20, 10)

    # 예측 함수
    def forward(self, x):
        in_size = x.size(0)
        # 컨볼루젼 1, 2 적용, 활성함수로 ReLU 사용
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))
        x = x.view(in_size, -1) # flatten the tensor
        x = self.fc2(self.fc1(x))
        # log softmax 사용하여 리턴.
        return F.log_softmax(x)


# 2. conv 2개, dense 2개
class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        # convolusion
        self.conv1 = nn.Conv2d(1, 10, kernel_size=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=1)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=1)
        self.conv4 = nn.Conv2d(30, 40, kernel_size=1)
        # maxpool
        self.mp = nn.MaxPool2d(2)
        # dense layer (fully connected)
        self.fc1 = nn.Linear(40, 20)
        self.fc2 = nn.Linear(20, 10)

    # 예측 함수
    def forward(self, x):
        in_size = x.size(0)
        # 컨볼루젼 1, 2 적용, 활성함수로 ReLU 사용
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))
        x = F.relu(self.mp(self.conv4(x)))
        x = x.view(in_size, -1) # flatten the tensor
        x = self.fc2(self.fc1(x))
        # log softmax 사용하여 리턴.
        return F.log_softmax(x)


# 2. conv 2개, dense 2개
class Net7(nn.Module):
    def __init__(self):
        super(Net7, self).__init__()
        # convolusion
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # maxpool
        self.mp = nn.MaxPool2d(2)
        # dense layer (fully connected)
        self.fc1 = nn.Linear(320, 160)
        self.fc2 = nn.Linear(160, 80)
        self.fc3 = nn.Linear(80, 10)

    # 예측 함수
    def forward(self, x):
        in_size = x.size(0)
        # 컨볼루젼 1, 2 적용, 활성함수로 ReLU 사용
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = x.view(in_size, -1) # flatten the tensor
        x = self.fc3(self.fc2(self.fc1(x)))
        # log softmax 사용하여 리턴.
        return F.log_softmax(x)


# 2. conv 2개, dense 2개
class Net8(nn.Module):
    def __init__(self):
        super(Net8, self).__init__()
        # convolusion
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=3)
        # maxpool
        self.mp = nn.MaxPool2d(2)
        # dense layer (fully connected)
        self.fc1 = nn.Linear(30, 20)
        self.fc2 = nn.Linear(20, 15)
        self.fc3 = nn.Linear(15, 10)

    # 예측 함수
    def forward(self, x):
        in_size = x.size(0)
        # 컨볼루젼 1, 2 적용, 활성함수로 ReLU 사용
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))
        x = x.view(in_size, -1) # flatten the tensor
        x = self.fc3(self.fc2(self.fc1(x)))
        # log softmax 사용하여 리턴.
        return F.log_softmax(x)


# 2. conv 2개, dense 2개
class Net9(nn.Module):
    def __init__(self):
        super(Net9, self).__init__()
        # convolusion
        self.conv1 = nn.Conv2d(1, 10, kernel_size=1)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=1)
        self.conv3 = nn.Conv2d(20, 30, kernel_size=1)
        self.conv4 = nn.Conv2d(30, 40, kernel_size=1)
        # maxpool
        self.mp = nn.MaxPool2d(2)
        # dense layer (fully connected)
        self.fc1 = nn.Linear(40, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 10)

    # 예측 함수
    def forward(self, x):
        in_size = x.size(0)
        # 컨볼루젼 1, 2 적용, 활성함수로 ReLU 사용
        x = F.relu(self.mp(self.conv1(x)))
        x = F.relu(self.mp(self.conv2(x)))
        x = F.relu(self.mp(self.conv3(x)))
        x = F.relu(self.mp(self.conv4(x)))
        x = x.view(in_size, -1) # flatten the tensor
        x = self.fc3(self.fc2(self.fc1(x)))
        # log softmax 사용하여 리턴.
        return F.log_softmax(x)