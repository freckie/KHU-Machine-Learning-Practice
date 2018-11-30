import math
import torch
import logging
import itertools
from tqdm import tqdm
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


# 생성기
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # 생성기 모델
        self.model = nn.Sequential(
            nn.Linear(3*100, 3*256),
            nn.ReLU(inplace=True),
            nn.Linear(3*256, 3*512),
            nn.ReLU(inplace=True),
            nn.Linear(3*512, 3*1024),
            nn.ReLU(inplace=True),
            nn.Linear(3*1024, 3 * 28 * 28),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), 3*100)
        return self.model(x)


# 분별기
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # 분별기 모델
        self.model = nn.Sequential(
            nn.Linear(3 * 28 * 28, 3*1024),
            nn.ReLU(inplace=True),
            nn.Linear(3*1024, 3*512),
            nn.ReLU(inplace=True),
            nn.Linear(3*512, 3*256),
            nn.ReLU(inplace=True),
            nn.Linear(3*256, 3*1),
            nn.ReLU(inplace=True),
            nn.Linear(3*1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x의 shape를 batch_size * (3 * 28 * 28)로 변경
        x = x.view(x.size(0), 3 * 28 * 28)
        out = self.model(x)
        return out.view(out.size(0), -1)


# 분별기(D) 트레이닝
def train_D(model, x, real_labels, fake_images, fake_labels):
    # grad 초기화
    model.zero_grad()

    # 실제 데이터로 loss 계산
    outputs = model(x)
    real_loss = criterion(outputs, real_labels)
    real_score = outputs
    
    # 생성된 데이터로 loss 계산
    outputs = model(fake_images)
    fake_loss = criterion(outputs, fake_labels)
    fake_score = outputs

    # 오류 합산 후 역전파, 가중치 갱신
    loss = real_loss + fake_loss
    loss.backward()
    optim_D.step()

    return loss, real_loss, fake_loss


# 생성기(G) 트레이닝
def train_G(model, output_D, real_labels):
    # grad 초기화
    model.zero_grad()

    # loss 계산 후 역전파 ,가중치 갱신
    loss = criterion(output_D, real_labels)
    loss.backward()
    optim_G.step()

    return loss


if __name__ == '__main__':
    # logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)

    # 학습 세팅
    batch_size = 100

    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    dataset_dir = './celeba_data/processed/'
    train_dataset = datasets.ImageFolder(dataset_dir, transform)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=100, shuffle=True)
    
    # 모델 생성
    model_G = Generator()
    model_D = Discriminator()

    # optimizer, criterion
    criterion = nn.BCELoss()
    optim_G = optim.Adam(model_G.parameters())
    optim_D = optim.Adam(model_D.parameters())

    # 학습
    epochs = 50
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data), Variable(target)
            real_labels = Variable(torch.ones(100))

            # 가짜 데이터 생성
            noise = Variable(torch.randn(data.size(0), 3*100))
            fake_images = model_G(noise)
            fake_labels = Variable(torch.zeros(data.size(0)))

            # 분별기(D) 학습
            loss_D, real_score, fake_score = train_D(model_D, data, real_labels, fake_images, fake_labels)

            # 생성기에서 샘플을 다시 추출, 분별기에서 결과 get
            noise = Variable(torch.randn(data.size(0), 3*100))
            fake_images = model_G(noise)
            outputs = model_D(fake_images)

            # 생성기(G) 학습
            loss_G = train_G(model_G, outputs, real_labels)

            # 진행 과정 출력
            if batch_idx % 10 == 0:
                logger.info('Epoch {} [{}/{} ({:.0f}%)] loss_D: {:.6f} / loss_G: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss_D.data[0], loss_G.data[0]))

    # 학습된 모델 저장
    torch.save(model_G.state_dict(), './gan_generator_celeba.pth')

    test_noise = Variable(torch.randn(1, 3*100))
    test_data = model_G(test_noise)
    new_data = test_data.view(3, 28, 28)
    tensor_to_img = transforms.ToPILImage()
    plt.imshow(transforms.functional.to_pil_image(new_data))
    plt.show()