from torch import nn
import torch as t
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        super(ResBlock, self).__init__()
        # ResNet的左边部分，也就是卷积而非直接连接部分
        self.left = nn.Sequential(
            # stride后面的是padding
            nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),# 如果不是inplace，不会更改值本身，而是得到返回值
            nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
            nn.BatchNorm2d(outchannel))
        self.right = shortcut

    def forward(self, x):
        output = self.left(x)
        res = x if self.right is None else self.right(x)
        # 左右层相连
        output += res
        return F.relu(output)

class ResNet(nn.Module):
    def __init__(self,num_classes=1000):
        super(ResNet, self).__init__()

        self.pre = nn.Sequential(
            # 输入3通道，输出64通道，7x7的核，步长为2，padding为3
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )


        self.layer1 = self._make_layer(64,64,3)


        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(512, num_classes)

        #测试一下输出是多大的shape
        b = t.ones(2, 3, 64, 64)
        a = self.pre(b)

        print("shape after pre", a.shape)
        a = self.layer1(a)
        print("shape after layer1: ", a.shape)

    def _make_layer(self,inchannel,outchannel,block_num,stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
            nn.BatchNorm2d(outchannel))
        layers = []
        layers.append(ResBlock(inchannel,outchannel,stride,shortcut))
        for i in range(1,block_num):
            layers.append(ResBlock(outchannel,outchannel))
        return nn.Sequential(*layers)

    def forward(self, *input):
        x = input[0]
        x = self.pre(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)





resnet = ResNet()
input = t.randn(1,3,224,224)
output = resnet(input)
print(output.shape)

# 当然我们可以直接使用现成的resnet

from torchvision import models
model = models.resnet34()
print(model)
# 以及读取里面的参数
print(model.conv1.weight.data)
