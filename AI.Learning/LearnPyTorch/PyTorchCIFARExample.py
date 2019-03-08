import os
import torch as t
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

show = ToPILImage()

download_dir = os.getcwd()
transform = transforms.Compose([
    transforms.ToTensor(),  # 相当于pipeline，先转成Tensor
    transforms.Normalize((.5, .5, .5), (.5, .5, .5))  # 然后全部归一化为0.5均值和0.5方差
])
# 返回的是dataset对象
trainset = tv.datasets.CIFAR10(
    root=download_dir,
    train=True,
    download=True,
    transform=transform  # 进行transform pipeline
)

trainloader = t.utils.data.DataLoader(
    trainset,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

# 测试集
testset = tv.datasets.CIFAR10(
    download_dir,
    train=False,
    download=True,
    transform=transform)

testloader = t.utils.data.DataLoader(
    testset,
    batch_size=4,
    shuffle=False,
    num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

data, label = trainset[100]
print(classes[label])
print(data, data.shape)

a = show((data + 1) / 2).resize((100, 100))
print(a)


# a.show()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
print(net)
from torch import optim

criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

t.set_num_threads(8)
for epoch in range(2):

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):

        # 输入数据
        inputs, labels = data

        # 梯度清零
        optimizer.zero_grad()

        # forward + backward
        outputs = net(inputs)
        # 这里需要给出相应的输出tensor，得到loss的tensor
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数（在每个参数都有grad的基础上）
        optimizer.step()

        # 打印log信息
        # loss 是一个scalar,需要使用loss.item()来获取数值，不能使用loss[0]
        running_loss += loss.item()
        if i % 2000 == 1999:  # 每2000个batch打印一下训练状态
            print('[%d, %5d] loss: %.3f' \
                  % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
print('Finished Training')

correct = 0
total = 0
# 测试，不需要梯度就关闭梯度
with t.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = t.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

# 下面是在GPU上训练！
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

net.to(device)
# 需要把每个样本加载到GPU中
images = images.to(device)
labels = labels.to(device)
output = net(images)
loss= criterion(output,labels)