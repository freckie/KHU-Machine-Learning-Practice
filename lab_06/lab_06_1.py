# https://github.com/pytorch/examples/blob/master/mnist/main.py

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

import lab_06_1_net as my_net

models = [my_net.Net1(), my_net.Net2(), my_net.Net3(), my_net.Net4(),
            my_net.Net5(), my_net.Net6(), my_net.Net7(), my_net.Net8(), my_net.Net9()]
idx = 1
for model in models:
    test_loss_list = list()

    # Training settings
    batch_size = 64

    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # optimizer로 SGD 사용. (학습률:0.01, 모멘텀:0.5)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


    def train(epoch):
        # 학습
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            # gradient 초기화
            optimizer.zero_grad()
            output = model(data)
            # loss 계산
            loss = F.nll_loss(output, target)
            # 오류 역전파
            loss.backward()
            # 가중치 갱신
            optimizer.step()
            if batch_idx % 10 == 0:
                print('[Net{} Model] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    idx, epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data[0]))


    # 테스트 함수
    def test():
        model.eval()
        test_loss = 0
        correct = 0

        for data, target in test_loader:
            data, target = Variable(data, volatile=True), Variable(target)
            output = model(data)
            # 배치의 loss를 합산
            test_loss += F.nll_loss(output, target, size_average=False).data[0]
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
        
        return test_loss


    # 메인 루프.
    start_time = time.time()
    for epoch in range(1, 10):
        train(epoch)
        test_loss_list.append(test())

    print('[Net{} Model] Train Completed : {}'.format(idx, time.time() - start_time))
    plt.plot(range(1, 10), test_loss_list, label='Net{} Model'.format(idx))
    idx += 1

plt.ylabel('Test Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()