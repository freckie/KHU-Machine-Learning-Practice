# https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt


def main(lr, momentum):
    # 학습 배치 사이즈 64
    batch_size = 64

    # DataLoader를 사용해 MNIST 데이터셋 로드
    train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 학습 네트워크
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            # layer 세팅
            self.l1 = nn.Linear(784, 520)
            self.l2 = nn.Linear(520, 320)
            self.l3 = nn.Linear(320, 240)
            self.l4 = nn.Linear(240, 120)
            self.l5 = nn.Linear(120, 10)

        # 예측 함수
        def forward(self, x):
            x = x.view(-1, 784) # Flatten the data (n, 1, 28, 28)-> (n, 784)
            x = F.relu(self.l1(x))
            x = F.relu(self.l2(x))
            x = F.relu(self.l3(x))
            x = F.relu(self.l4(x))
            return self.l5(x)


    model = Net()
    # loss로 cross entropy 사용
    criterion = nn.CrossEntropyLoss()
    # optimizer로 SGD 사용
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    #optimizer = optim.Adam(model.parameters(), lr=0.01)


    # 학습
    def train(epoch):
        loss_list = list()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            # gradient 초기화
            optimizer.zero_grad()
            # 예측
            output = model(data)
            # loss 계산
            loss = criterion(output, target)
            # 오류 역전파
            loss.backward()
            # 가중치 갱신
            optimizer.step()
            # 10의 배수마다 중간 결과 출력
            if batch_idx % 10 == 0:
                loss_list.append(loss.data[0])
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))
        return loss_list


    # 테스트
    def test():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            # 예측
            output = model(data)
            # 배치의 loss를 합산
            test_loss += criterion(output, target).data[0]
            # get the index of the max
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


    loss_list = list()
    for epoch in range(1, 10):
        loss_list.extend(train(epoch))
        test()

    plt.plot(range(len(loss_list)), loss_list, label='lr={} / momentum={}'.format(lr, momentum))


lr_list = [0.1, 0.01, 0.001]
momentum_list = [0.2, 0.5, 0.8]
for momentum in momentum_list:
    main(lr_list[1], momentum)

plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.show()