from torchvision import transforms as T
from PIL import Image
from torch.utils import data
import os
import random


transform = T.Compose([
    T.Resize(224), # 缩放图片(Image)，保持长宽比不变，最短边为224像素
    T.CenterCrop(224), # 从图片中间切出224*224的图片
    T.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
    T.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
])
class ImageDataset(data.Dataset):

    def __init__(self,root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]


    def __getitem__(self, item):
        data = Image.open(self.imgs[item])
        if self.transforms:
            data = self.transforms(data)
        return data
#除了上述操作之外，transforms还可通过Lambda封装自定义的转换策略。
# 例如想对PIL Image进行随机旋转，则可写成这样
trans=T.Lambda(lambda img: img.rotate(random.random()*360))




#torchvision已经预先实现了常用的Dataset，包括前面使用过的CIFAR-10，
# 以及ImageNet、COCO、MNIST、LSUN等数据集，可通过诸如
# torchvision.datasets.CIFAR10来调用，具体使用方法请参看官方文档1。
# 在这里介绍一个会经常使用到的Dataset——ImageFolder，
# 它的实现和上述的DogCat很相似。ImageFolder假设所有的文件按文件夹保存，
# 每个文件夹下存储同一个类别的图片，文件夹名为类名，其构造函数如下：

# ImageFolder(root, transform=None, target_transform=None, loader=default_loader)


'''

torch支持Tensorfboard
安装tensorboardX：可通过pip install tensorboardX命令直接安装。
tensorboardX的使用非常简单。首先用如下命令启动tensorboard：

tensorboard --logdir <your/running/dir> --port <your_bind_port>
打开浏览器输入http://localhost:6006（其中6006应改成你的tensorboard所绑定的端口)
'''
from tensorboardX import SummaryWriter
# 构建logger对象，logdir用来指定log文件的保存路径
# flush_secs用来指定刷新同步间隔
logger = SummaryWriter(log_dir='experimient_cnn', flush_secs=2)

for ii in range(100):
    logger.add_scalar('data/loss', 10-ii**0.5)
    logger.add_scalar('data/accuracy', ii**0.5/10)

# Visdom1是Facebook专门为PyTorch开发的一款可视化工具，其开源于2017年3月。
# Visdom十分轻量级，但却支持非常丰富的功能，能胜任大多数的科学运算可视化任务。
# 其可视化界面如图3所示。
import torch as t

import visdom

# 新建一个连接客户端
# 指定env = u'test1'，默认端口为8097，host是‘localhost'
# vis = visdom.Visdom(env=u'test1',use_incoming_socket=False,port=12345)
#
# x = t.arange(1, 30, 0.01)
# y = t.sin(x)
# vis.line(X=x, Y=y, win='sinx', opts={'title': 'y=sin(x)'})

tensor = t.Tensor(3, 4)
# 返回一个新的tensor，保存在第1块GPU上，但原来的tensor并没有改变
tensor.cuda(0)
tensor.is_cuda # False


class VeryBigModule(nn.Module):
    def __init__(self):
        super(VeryBigModule, self).__init__()
        # 将parameters放入不同的GPU中！
        self.GiantParameter1 = t.nn.Parameter(t.randn(100000, 20000)).cuda(0)
        self.GiantParameter2 = t.nn.Parameter(t.randn(20000, 100000)).cuda(1)

    def forward(self, x):
        # 输入也是在不同的设备中！
        x = self.GiantParameter1.mm(x.cuda(0))
        x = self.GiantParameter2.mm(x.cuda(1))
        return x

'''
GPU运算很快，但对于很小的运算量来说，并不能体现出它的优势，因此对于一些简单的操作可直接利用CPU完成
数据在CPU和GPU之间，以及GPU与GPU之间的传递会比较耗时，应当尽量避免
在进行低精度的计算时，可以考虑HalfTensor，它相比于FloatTensor能节省一半的显存，但需千万注意数值溢出的情况。
'''

# 如果未指定使用哪块GPU，默认使用GPU 0
x = t.cuda.FloatTensor(2, 3)
# x.get_device() == 0
y = t.FloatTensor(2, 3).cuda()
# y.get_device() == 0

# 指定默认使用GPU 1
with t.cuda.device(1):
    # 在GPU 1上构建tensor
    a = t.cuda.FloatTensor(2, 3)

    # 将tensor转移至GPU 1
    b = t.FloatTensor(2, 3).cuda()
    print(a.get_device() == b.get_device() == 1 )

    c = a + b
    print(c.get_device() == 1)

    z = x + y
    print(z.get_device() == 0)

    # 手动指定使用GPU 0
    d = t.randn(2, 3).cuda(0)
    print(d.get_device() == 2)


# 保存
a = t.Tensor(3, 4)
if t.cuda.is_available():
    a = a.cuda(1)  # 把a转为GPU1上的tensor,
    t.save(a, 'a.pth')

    # 加载为b, 存储于GPU1上(因为保存时tensor就在GPU1上)
    b = t.load('a.pth')

    # 加载为c, 存储于CPU
    c = t.load('a.pth', map_location=lambda storage, loc: storage)

    # 加载为d, 存储于GPU0上
    d = t.load('a.pth', map_location={'cuda:1': 'cuda:0'})