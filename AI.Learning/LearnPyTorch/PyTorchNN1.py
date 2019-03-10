import torch.nn as nn
import torch.nn.functional as F
import torch as t

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()


        # 输入通道，输出通道，kernelsize等参数，具体见文档
        self.conv1 = nn.Conv2d(1,6,5)
        self.conv2 = nn.Conv2d(6,16,5)

        # 全连接，注意这里为什么看上去参数不匹配：
        '''
        输入W为32，kernel为5，Padding为0，步长为1，于是
        输出N = (32 - 5 + 2*0)/1 + 1 = 28
        之后还有maxpooling！maxpooling默认不是滑动窗口的
        而是步长和kernel一致，kernel为2的话直接在2x2的每个
        区域内求max，最终size折半
        '''
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,10)

    def forward(self, x):
        # 正向传播层，最后一个参数是maxpool的kernel size
        print("print size when forward:", x.size())
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        print("print size when forward:", x.size())
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        print("print size when forward:", x.size())
        # view类似于reshape
        x = x.view(x.size()[0], -1)
        # 任何时候forward都可以自由地输出东西，或者使用for循环
        print("print size when forward:",x.size())
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()
print(net)
# 从这里可以得到所有层的信息！这很重要，因为有时候需要截断
# 全连接层来获得CNN的特征向量
for name,params in net.named_parameters():
    print(name,": ",params.size())
# 注意通道数在第二个维度，第一个是n sample
input = t.randn(1,1,32,32)
# 使用forward来得到结果，注意forward里面的print函数也调用了，于是可以很好地查看
out = net(input)
print(out.size())
print(out)
# 清零梯度
net.zero_grad()
# 因为反向传播需要一个loss，这里传一个固定的回去
# 这里的loss不是一个浮点数而是张量，之后会讲到
out.backward(t.ones(1,10))
# 这样的话所有的层都会有一个grad了，
print(net.conv2.weight.grad)
# 注意这里input是没有梯度的！
print("The grad of input: ",input.grad)

# 额外：
# 样本数目和tf一样是一个batch一个batch进去的，所以
# 在固定的shape上还需要多一个第一个维度作为样本数量
# 可以使用input.unsqueeze(0)将batch_size设为1，但是不建议


# 损失函数
output = net(input)
target = t.Tensor(list(range(10))).view(1,10)
print(target)
loss_fn = nn.MSELoss()
loss = loss_fn(output,target)
# 得到loss标量，注意这个是tensor并且有grad fn
# 这里就解释了为什么之前需要用向量来反向传播，因为loss之后
# 会自动得到向量的梯度
print(loss)
# 可以看到bias梯度都是1，是relu的求导
print(net.fc3.bias.grad)
net.zero_grad()
# 使用loss反向传播！
loss.backward()
print(net.conv1.bias.grad)

# 在得到梯度之后，需要进行更新
# 也就是weight = weight - lr * grad
def do_optim_manually():
    # 手动计算梯度并传播，也就是将data减去梯度乘以lr
    # 注意这里是用的一个sample的梯度，也就是SGD随机梯度下降
    # 如果需要一个batch，可以采用for循环累积梯度然后取均值
    learning_rate = 0.01
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim
# 建立一个optimizer，相当于帮助自动根据grad进行weight更新
optimizer = optim.SGD(net.parameters(),lr=0.01)
optimizer.zero_grad()
# 标准反向传播过程
output = net(input)
loss = loss_fn(output,target)
loss.backward()
# 优化
optimizer.step()



