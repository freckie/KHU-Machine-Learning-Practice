import torch
import logging
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms


# 생성기
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.map = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1 * 28 * 28),
            nn.Tanh()
        )

    def forward(self, z, y):
        try:
            return self.map(torch.cat([z, y], 1))
        except Exception as exc:
            print(z.type())
            print(z.shape)
            print(y.type())
            print(y.shape)
            print(exc)
            exit(1)


# 분별기
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1 * 28 * 28 + 10, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, y):
        # x의 shape를 batch_size * 28 * 28로 변경
        x = x.view(x.size(0), 28 * 28)
        out = self.model(torch.cat([x, y], 1))
        out = out.view(out.size(0), -1)
        return out


# 분별기(D) 트레이닝
def train_D(model, x, real_labels, fake_images, fake_labels, y):
    # grad 초기화
    model.zero_grad()

    # 실제 데이터로 loss 계산
    outputs = model(x, y)
    real_loss = criterion(outputs, real_labels)
    
    # 생성된 데이터로 loss 계산
    outputs = model(fake_images, y)
    fake_loss = criterion(outputs, fake_labels)

    # 오류 합산 후 역전파, 가중치 갱신
    loss = real_loss + fake_loss
    loss.backward()
    optim_D.step()

    return loss, real_loss, fake_loss


# 생성기(G) 트레이닝
def train_G(model, output_D, real_labels, y):
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
    batch_size = 64

    # MNIST Dataset
    train_dataset = datasets.MNIST(root='./data/', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data/', train=False, transform=transforms.ToTensor())

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 모델 생성
    model_G = Generator()
    model_D = Discriminator()

    # optimizer, criterion
    criterion = nn.BCELoss()
    optim_G = optim.Adam(model_G.parameters())
    optim_D = optim.Adam(model_D.parameters())

    # 학습
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        z = Variable(torch.randn(100, 100).long())

        # 가짜 데이터 생성
        fake_images = model_G(z, target)

        # 라벨 초기화
        real_labels = Variable(torch.ones(100))
        fake_labels = Variable(torch.zeros(100))

        # 분별기(D) 학습
        loss_D, real_score, fake_score = train_D(model_D, data, real_labels, fake_images, fake_labels, target)

        # batch_size * 100의 noise를 랜덤으로 추출
        z = Variable(torch.randn(100, 100).long())

        # 가짜 데이터 생성 후 분별
        fake_images = model_G(z, target)
        outputs = model_D(fake_images, target)

        # 생성기(G) 학습
        loss_G = train_G(model_G, outputs, real_labels, target)

        # 중간 결과 출력
        if batch_idx % 100 == 0:
            logger.info('[{}/{} ({:.0f}%)] loss_D: {} / loss_G: {}'.format(
                batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx * len(train_loader), loss_D.data[0], loss_G.data[0]))