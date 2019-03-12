# coding:utf8
import torch
from torchvision.models import vgg16
from collections import namedtuple
import torch as t
import torchnet as tnt
from torch.utils import data
from torch.nn import functional as F
import tqdm
import os
import ipdb
import visdom
import time
import torchvision as tv
from torch import nn
import torch as t
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

'''
这里我们并不是在vgg中使用噪音x然后优化x，那一种方法是把噪音作为x，然后求取x和content以及style的loss，然后优化x，这样很慢因为转换过程是训练过程，
现在我们采用transformer输入图片输出风格化后的图片
            流程：
            x是content，通过transformer变成风格化的图片
            y是经过风格化后的图片
            也就是说我们训练的是transformer使其将所有输入图片变成风格化过后的图片
            
            然后x y提取特征得到特征向量，求出loss
            当然这个是content的loss
            style会在之后求，同样也是提取特征，不过y的特征和style的特征都会，获得格拉姆gram矩阵，然后以矩阵的范数loss作为loss来反向传播
            最终优化的是transformer
            
'''


def gram_matrix(y):
    """
    Gram Matrices格拉姆矩阵，这个矩阵可以捕获风格信息，具体请查看相应资料
    假设输入图像经过卷积后，得到的feature map为[b, ch, h, w]。
    我们经过flatten和矩阵转置操作，可以变形为[b, ch, h*w]和[b, h*w, ch]的矩阵。
    再对1，2维作矩阵内积得到[b, ch, ch]大小的矩阵，这就是我们所说的Gram Matrices。

    实际上Gram矩阵的每个值可以说是代表m通道的特征与n通道的特征的互相关程度。
    先把通道的feature map给flatten，然后两两相乘，得到互相关，feature大的部分feature放得更大


    输入 b,c,h,w
    输出 b,c,c
    """
    (b, ch, h, w) = y.size()
    # 首先flatten后面两个维度，变成一维表示的feature map，但是保留通道数目
    features = y.view(b, ch, w * h)
    # transpose是交换维度，也就是通道和一维表示的feature map交换
    features_t = features.transpose(1, 2)
    # bmm是batch matrix multiply，按照batch求内积
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


class Visualizer():
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env,use_incoming_socket=False,  **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def img(self, name, img_):
        """
        self.img('input_img',t.Tensor(64,64))
        """

        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(),
                       win=name,
                       opts=dict(title=name)
                       )

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        """
        一个batch的图片转成一个网格图，i.e. input（36，64，64）
        会变成 6*6 的网格图，每个格子大小64*64
        """
        self.img(name, tv.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0)))

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


def get_style_data(path):
    """
    加载风格图片，
    输入： path， 文件路径
    返回： 形状 1*c*h*w， 分布 -2~2
    """
    style_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    style_image = tv.datasets.folder.default_loader(path)
    style_tensor = style_transform(style_image)
    return style_tensor.unsqueeze(0)# 增加第一个维度作为样本数量


def normalize_batch(batch):
    """
    输入: b,ch,h,w  0~255
    输出: b,ch,h,w  -2~2
    """
    mean = batch.data.new(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = batch.data.new(IMAGENET_STD).view(1, -1, 1, 1)
    mean = (mean.expand_as(batch.data))# 这里相当于手动广播操作
    std = (std.expand_as(batch.data))
    return (batch / 255.0 - mean) / std


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained=True).features)[:23]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()# 相当于进入eval模式，具体看代码，有train=False

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):# 获取某一层的特征
            x = model(x)
            if ii in {3, 8, 15, 22}:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(*results)# 将这些特征建立索引


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()

        # 下卷积层
        self.initial_layers = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(True),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True),
        )

        # Residual layers(残差层)
        self.res_layers = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        # Upsampling Layers(上卷积层)
        self.upsample_layers = nn.Sequential(
            UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(True),
            ConvLayer(32, 3, kernel_size=9, stride=1)
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.res_layers(x)
        x = self.upsample_layers(x)
        return x


class ConvLayer(nn.Module):
    """
    设置反射填充而不是0补
    add ReflectionPad for Conv
    默认的卷积的padding操作是补0，这里使用边界反射填充
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    默认的卷积的padding操作是补0，这里使用边界反射填充
    先上采样，然后做一个卷积(Conv2d)，而不是采用ConvTranspose2d，这种效果更好，参见
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = t.nn.functional.interpolate(x_in, scale_factor=self.upsample)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out


class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

## main

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class Config(object):
    image_size = 256  # 图片大小
    batch_size = 8
    data_root = 'data/'  # 数据集存放路径：data/coco/a.jpg
    num_workers = 4  # 多线程加载数据
    use_gpu = True  # 使用GPU

    style_path = 'style.jpg'  # 风格图片存放路径
    lr = 1e-3  # 学习率

    env = 'neural-style'  # visdom env
    plot_every = 10  # 每10个batch可视化一次

    epoches = 2  # 训练epoch

    content_weight = 1e5  # content_loss 的权重
    style_weight = 1e10  # style_loss的权重

    model_path = None  # 预训练模型的路径
    debug_file = '/tmp/debugnn'  # touch $debug_fie 进入调试模式

    content_path = 'input.png'  # 需要进行分割迁移的图片
    result_path = 'output.png'  # 风格迁移结果的保存路径


def train(**kwargs):
    opt = Config()
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis = Visualizer(opt.env)

    # 数据加载
    transfroms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 255)
    ])
    dataset = tv.datasets.ImageFolder(opt.data_root, transfroms)
    dataloader = data.DataLoader(dataset, opt.batch_size)

    # 转换网络
    transformer = TransformerNet()
    if opt.model_path:
        # 载入ckpt
        transformer.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    transformer.to(device)

    # 损失网络 Vgg16
    # 注意这里非常重要！Vgg的forward是改写了的！只会输出固定的4个层！！
    vgg = Vgg16().eval()# 设置train=False
    vgg.to(device)# 放入gpu
    for param in vgg.parameters():
        param.requires_grad = False# 不需要训练

    # 优化器
    optimizer = t.optim.Adam(transformer.parameters(), opt.lr)

    # 获取风格图片的数据
    style = get_style_data(opt.style_path)
    vis.img('style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
    style = style.to(device)

    # 风格图片的gram矩阵
    # 对于每个feature map求矩阵
    with t.no_grad():
        features_style = vgg(style)
        gram_style = [gram_matrix(y) for y in features_style]

    # 损失统计
    style_meter = tnt.meter.AverageValueMeter()
    content_meter = tnt.meter.AverageValueMeter()

    for epoch in range(opt.epoches):
        content_meter.reset()
        style_meter.reset()

        for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):


            # 训练
            optimizer.zero_grad()
            x = x.to(device)
            y = transformer(x)
            y = normalize_batch(y)
            x = normalize_batch(x)
            # x进入了转换网络，然后得到y，之后x和y都会在vgg中提取特征！！！
            features_y = vgg(y)
            features_x = vgg(x)

            # 对于content来讲，提取的特征都使用relu2_2层，是之前已经搭好的计算图输出（vgg的输出）

            content_loss = opt.content_weight * F.mse_loss(features_y.relu2_2, features_x.relu2_2)

            # 对于style的loss，会求gram矩阵
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gram_y = gram_matrix(ft_y)
                style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= opt.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            # 损失平滑
            content_meter.add(content_loss.item())
            style_meter.add(style_loss.item())

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # 可视化
                vis.plot('content_loss', content_meter.value()[0])
                vis.plot('style_loss', style_meter.value()[0])
                # 因为x和y经过标准化处理(utils.normalize_batch)，所以需要将它们还原
                vis.img('output', (y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                vis.img('input', (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

        # 保存visdom和模型
        vis.save([opt.env])
        t.save(transformer.state_dict(), 'checkpoints/%s_style.pth' % epoch)


@t.no_grad()
def stylize(**kwargs):
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device = t.device('cuda') if opt.use_gpu else t.device('cpu')

    # 图片处理
    content_image = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()

    # 模型
    style_model = TransformerNet().eval()
    style_model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    style_model.to(device)

    # 风格迁移与保存
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    tv.utils.save_image(((output_data / 255)).clamp(min=0, max=1), opt.result_path)


if __name__ == '__main__':
    import fire

    fire.Fire()