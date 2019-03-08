'''
PyTorch中的矩阵运算也支持GPU加速
因此可以胜任更多的任务

函数名以_结尾的函数会修改自身，否则返回结果
'''
import torch as t
import copy
a = t.arange(0, 6)
b = a.view(-1,3)
c = copy.deepcopy(b)
print(b,b.shape)
# 增加一个维度，注意要有下划线，才能替代b的值，否则
# 需要b = b.unsqueeze(-2)
b.unsqueeze_(-2)
print(b,b.shape)

# squeeze和unsqueeze操作不会改变里面的数据总量，只会改变shape
print(c,c.shape)
c = c.unsqueeze_(0)
print(c,c.shape)
c = c.squeeze_()
print(c)

a = t.randn(3, 4)
print(a,a.shape)
b = a[None]# 相当于增加一维
print(b,b.shape)

c = a[:,None,:,None,None]# 相当于增加3个维度
print(c,c.shape)

c = c[c>1]# 不管之前维度有多大，条件选择得到都是一维向量
print(c)

print(t.LongTensor([0,1]))
# 返回01作为True False的矩阵
print(a>1)

a = t.arange(0, 16).view(4, 4)
print(a)

index = t.LongTensor([[0,1,2,3]])
print(index)
# index是一个二维向量，相当于在第一个维度上，分别取第0 1 2 3个元素
# 选取对角线的元素
a.gather(0, index)
# 选取反对角线上的元素，注意与上面的不同
index = t.LongTensor([[3,2,1,0]])
a.gather(0, index)
# 选取两个对角线上的元素
index = t.LongTensor([[0,1,2,3],[3,2,1,0]]).t()
b = a.gather(1, index)

# 把两个对角线元素放回去到指定位置
c = t.zeros(4,4)
#c.scatter_(1, index, b)

# 关于数据类型

# 设置默认tensor，注意参数是字符串
t.set_default_tensor_type('torch.DoubleTensor')

a = t.Tensor(2,3)
print(a.dtype) # 现在a是DoubleTensor,dtype是float64

b = a.float()
print(b.dtype)


t.zeros_like(a, dtype=t.int16) #可以修改某些属性

# 降维操作，如果保持dim就会设为1，否则删除这个dim
b = t.ones(2, 3)
b.sum(dim = 0, keepdim=True)


# 转置操作导致矩阵在存储空间不连续，需要调用.contiguous方法变得连续
b = a.t()
b.is_contiguous()
b.contiguous()


# tensor分为头信息区(Tensor)和存储区(Storage)，信息区主要保存着tensor的形状（size）、
# 步长（stride）、数据类型（type）等信息，而真正的数据则保存成连续数组。
a = t.arange(0, 6)
c = a.storage()
print(c)

#Tensor的保存和加载十分的简单，使用t.save和t.load即可
# 完成相应的功能。在save/load时可指定使用的pickle模块，
# 在load时还可将GPU tensor映射到CPU或其它GPU上。


if t.cuda.is_available():
    a = a.cuda(1) # 把a转为GPU1上的tensor,
    t.save(a,'a.pth')

    # 加载为b, 存储于GPU1上(因为保存时tensor就在GPU1上)
    b = t.load('a.pth')
    # 加载为c, 存储于CPU
    c = t.load('a.pth', map_location=lambda storage, loc: storage)
    # 加载为d, 存储于GPU0上
    d = t.load('a.pth', map_location={'cuda:1':'cuda:0'})

# t.set_num_threads可以设置PyTorch进行CPU多线程并行计算时候所占用的线程数，
# 这个可以用来限制PyTorch所占用的CPU数目。

# 例子：线性回归

w = t.rand(1, 1).to(device)
b = t.zeros(1, 1).to(device)

lr = 0.02  # 学习率

for ii in range(500):
    x, y = get_fake_data(batch_size=4)

    # forward：计算loss
    y_pred = x.mm(w) + b.expand_as(y)  # x@W等价于x.mm(w);for python3 only
    loss = 0.5 * (y_pred - y) ** 2  # 均方误差
    loss = loss.mean()

    # backward：手动计算梯度
    dloss = 1
    dy_pred = dloss * (y_pred - y)

    dw = x.t().mm(dy_pred)
    db = dy_pred.sum()

    # 更新参数
    w.sub_(lr * dw)
    b.sub_(lr * db)

    if ii % 50 == 0:
        # 画图
        display.clear_output(wait=True)
        x = t.arange(0, 6).view(-1, 1)
        y = x.mm(w) + b.expand_as(x)
        plt.plot(x.cpu().numpy(), y.cpu().numpy())  # predicted

        x2, y2 = get_fake_data(batch_size=32)
        plt.scatter(x2.numpy(), y2.numpy())  # true data

        plt.xlim(0, 5)
        plt.ylim(0, 13)
        plt.show()
        plt.pause(0.5)

print('w: ', w.item(), 'b: ', b.item())
