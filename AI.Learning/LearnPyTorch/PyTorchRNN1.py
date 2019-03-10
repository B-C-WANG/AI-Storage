import torch as t
import torch.nn as nn

# 关于RNN用法之后讲解，这里讲API格式

# 连续LSTM层
# batch size为3，序列长度为2，序列每个元素4维
input = t.randn(2,3,4)
# 输入4维，隐藏层3到1
lstm = nn.LSTM(4,3,1)
# batch size为3,3个隐藏元
h0 = t.randn(1,3,3)
c0 = t.randn(1,3,3)
out,hn = lstm(input,(h0,c0))

# 单个LSTM层
input = t.randn(2, 3, 4)
# 一个LSTMCell对应的层数只能是一层
lstm = nn.LSTMCell(4, 3)
hx = t.randn(3, 3)
cx = t.randn(3, 3)
out = []
for i_ in input:
    hx, cx=lstm(i_, (hx, cx))
    out.append(hx)
t.stack(out)



# 有4个词，每个词用5维的向量表示
embedding = nn.Embedding(4, 5)
print(embedding.weight.shape)
# 可以用预训练好的词向量初始化embedding
embedding.weight.data = t.arange(0,20).view(4,5)
# 输入一个3，2,1,0向量
# 需要注意，embedding输入是4个词，每个int代表一个词
# 所以只允许输入是0 1 2 3，之后embedding成相应大小
input = t.Tensor([3,1,3,3,2,2]).long()
#input = t.arange(3, 0, -1).long()
print(input)
output = embedding(input)
print(output)

# batch_size=3，计算对应每个类别的分数（只有两个类别）
score = t.randn(3, 2)
# 三个样本分别属于1，0，1类，label必须是LongTensor
label = t.Tensor([1, 0, 1]).long()

# loss与普通的layer无差异
criterion = nn.CrossEntropyLoss()
loss = criterion(score, label)
print(loss)


from torch import  optim
optimizer = optim.SGD(params=net.parameters(), lr=1)
optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()

# 不同层可以设置不同的学习率，这很重要，因为每一层实际上都是独立根据grad对weight进行修正的

# 为不同子网络设置不同的学习率，在finetune中经常用到
# 如果对某个参数不指定学习率，就使用最外层的默认学习率

optimizer =optim.SGD([
                {'params': net.features.parameters()}, # 学习率为1e-5
                {'params': net.classifier.parameters(), 'lr': 1e-2}
            ], lr=1e-5)


# 只为两个全连接层设置较大的学习率，其余层的学习率较小
special_layers = nn.ModuleList([net.classifier[0], net.classifier[3]])
special_layers_params = list(map(id, special_layers.parameters()))# 获得id存储起来
base_params = filter(lambda p: id(p) not in special_layers_params,# 根据parameters的id来得到，实际上如果以指针存储对象就不需要使用id
                     net.parameters())
# 设置的是parameters对象，所以可以通过遍历得到
optimizer = t.optim.SGD([
            {'params': base_params},
            {'params': special_layers.parameters(), 'lr': 0.01}
        ], lr=0.001 )


# 方法1: 调整学习率，新建一个optimizer
old_lr = 0.1
optimizer1 =optim.SGD([
                {'params': net.features.parameters()},
                {'params': net.classifier.parameters(), 'lr': old_lr*0.1}
            ], lr=1e-5)
optimizer1

# 方法2: 调整学习率, 手动decay, 保存动量
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
optimizer

# 利用nn.init初始化
from torch.nn import init
linear = nn.Linear(3, 4)

t.manual_seed(1)
# 等价于 linear.weight.data.normal_(0, std)
init.xavier_normal_(linear.weight)

# 直接初始化
import math
t.manual_seed(1)

# xavier初始化的计算公式
std = math.sqrt(2)/math.sqrt(7.)
linear.weight.data.normal_(0,std)


# 保存模型
t.save(net.state_dict(), 'net.pth')

# 加载已保存的模型
net2 = Net()
net2.load_state_dict(t.load('net.pth'))