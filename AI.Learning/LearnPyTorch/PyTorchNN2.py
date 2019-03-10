import torch as t
from torch import nn

class Linear(nn.Module):
    def __init__(self,in_features,out_features):
        super(Linear, self).__init__()
        # 注意这里并不是Tensor了，而是Parameter
        self.w = nn.Parameter(t.randn(in_features,out_features))
        self.b = nn.Parameter(t.randn(out_features))

    def forward(self, *input):
        x = input[0].mm(self.w)
        print(self.b)
        # 相当于进行广播
        print(self.b.expand_as(x))
        return x + self.b.expand_as(x)

layer = Linear(4,3)
input = t.randn(2,4)
output = layer(input)# 这里调用forward函数
print(output)

from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
to_tensor = ToTensor() # img -> tensor
to_pil = ToPILImage()
img = Image.open('timg.jpg')
# 增加一个维度，第一维度，用来表示样本数目
input = to_tensor(img).unsqueeze(0)
def conv_test():
    img.show()

    print(input.shape)

    # 锐化卷积核，3通道，3x3
    kernel = t.ones(3,3, 3)/-9.
    kernel[1][1] = 1
    conv = nn.Conv2d(1, 1, (3, 3), 1, bias=False)
    # 把卷积核更改为固定权重！！注意这里的shape！！！
    conv.weight.data = kernel.view(1, 3, 3, 3)

    out = conv(input)
    to_pil(out.data.squeeze(0)).show()

def pool_test():
    img.show()
    # 在每个3x3区域取最大值，步长也为3x3，没有滑动窗口重叠
    pool = nn.MaxPool2d(3,3)
    pool = nn.AvgPool2d(3, 3)

    out = pool(input)
    to_pil(out.data.squeeze(0)).show()

pool_test()

'''
torch和tf不同之处在于，每次的计算结果，不管是BN，
relu还是dropout，结果都是立即可见的！
'''

# 4 channel，初始化标准差为4，均值为0
bn = nn.BatchNorm1d(4)
bn.weight.data = t.ones(4) * 4
bn.bias.data = t.zeros(4)

bn_out = bn(h)
# 注意输出的均值和方差
# 方差是标准差的平方，计算无偏方差分母会减1
# 使用unbiased=False 分母不减1
bn_out.mean(0), bn_out.var(0, unbiased=False)

# 每个元素以0.5的概率舍弃
dropout = nn.Dropout(0.5)
o = dropout(bn_out)
print(o) # 打印出来可以看到有一半左右的数变为0

relu = nn.ReLU(inplace=True)
input = t.randn(2, 3)
print(input)
output = relu(input)
print(output) # 打印出来可以看到小于0的都被截断为0
# 等价于input.clamp(min=0)

# 使用类似keras的Sequential方法创建

# Sequential的三种写法
net1 = nn.Sequential()
net1.add_module('conv', nn.Conv2d(3, 3, 3))
net1.add_module('batchnorm', nn.BatchNorm2d(3))
net1.add_module('activation_layer', nn.ReLU())

net2 = nn.Sequential(
        nn.Conv2d(3, 3, 3),
        nn.BatchNorm2d(3),
        nn.ReLU()
        )

from collections import OrderedDict
net3= nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(3, 3, 3)),
          ('bn1', nn.BatchNorm2d(3)),
          ('relu1', nn.ReLU())
        ]))
print('net1:', net1)
print('net2:', net2)
print('net3:', net3)

# 可根据名字或序号取出子module，这非常重要，对于截断
# 全连接网络常常用到，需要非常灵活
net1.conv, net2[0], net3.conv1

input = t.rand(1, 3, 4, 4)

output = net1(input)
output = net2(input)
output = net3(input)
output = net3.relu1(net1.batchnorm(net1.conv(input)))

# 采用ModuleList，注意不能使用list
modellist = nn.ModuleList([nn.Linear(3,4), nn.ReLU(), nn.Linear(4,2)])
input = t.randn(1, 3)
for model in modellist:
    input = model(input)