import torch as t
import numpy as np


# 创建初始化为0的矩阵
x = t.Tensor(5,3)
print(x)
# 从numpy到tensor
array = np.array([[1,2,3],[4,5,6]])
# 注意需要大写
x = t.Tensor(array)
print(x)
print(x.shape,x.size())
# 从tensor到array
x = np.array(x)
print(x)

x = t.Tensor(x)
y = x
# tensor加法
z = x + y
print(z)

x = t.Tensor([1,2,3])
y = t.Tensor([3,4,5])
# 这个加法值是return过去的
y.add(x)
print(y)
# 这个加法值是会改变y的
# NOTICE：所有以下划线结尾的函数都修改自身
y.add_(x)
print(y)

x = t.ones(5,2)
print(x)
print(x.numpy())

z = x[0,0]
# 当使用切片获得一个tensor对象后，需要使用以下方法转换为具体的值，否则就是tensor对象
print(z)
print(z.item())
print(float(z))
print(np.array(z))


# t.tensor操作默认数据拷贝，使用detach来共享
a = t.Tensor([1,2,3])
b = a.detach()
a += 1
print(b)

# 设置下一步运行是gpu还是cpu
# 这里的to相当于是把数据复制到了目标设备上，然后调用目标设备的计算
device = t.device("cuda:0"if t.cuda.is_available() else "cpu")
x = x.to(device)
print(x)

# 自动微分

x = t.ones(2,2,requires_grad=True)
print(x)
y = x.sum()
print(y)
print(y.grad_fn)
# 查看x的梯度
print(x.grad)
# 在y反向传播后查看梯度
y.backward()
print(x.grad)
# 每次反向传播都会积累梯度
y.backward()
print(x.grad)
# 将梯度清零，注意下划线是inplace操作，否则没有效果
x.grad.zero_()
print(x.grad)

y.backward()
print(x.grad)



