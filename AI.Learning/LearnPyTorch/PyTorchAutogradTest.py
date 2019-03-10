''''''

'''
在pytorch中每个Tensor也都可以看成是计算图
因此也是构建计算图来实现求导等功能
'''

'''
反向传播是在计算图中选择一个根节点，然后给父节点的所有图的Variable添加grad

注意，同样是Tensor，为什么input不会从反向传播获得梯度，而只有Conv2D才会
需要记住区分一下，根据后面的例子学习
'''

import torch as t
import numpy as np
#在创建tensor的时候指定requires_grad

a = t.randn(3,4, requires_grad=True)
# 或者
a = t.randn(3,4).requires_grad_()
# 或者
a = t.randn(3,4)
a.requires_grad=True


import torch.nn as nn
import torch.nn.functional as F
# 重要：区分一下下面的计算图会不会需要求导！！
# 卷积，默认需要求导数，实际上被看成Variable
x = nn.Conv2d(1,6,5)
print(x.weight.requires_grad)
print(x.bias.requires_grad)
x.weight.requires_grad = False
print("after changed:",x.weight.requires_grad)# 把某些网络的权重更改为不可变！
# 随机数，实际上被看成是Tensor，不求导
x = t.randn(2,1)
print(x.requires_grad)
x = t.Tensor([1,2,3])
print(x.requires_grad)
# Dense，不求导
x = nn.Linear(84,10)
print(x.weight.requires_grad)

'''
另外重要的一点，如果某个父节点需要求导，则其全部依赖的子节点的requires_grad
都会求导！！！
但是求导并不代表就会更改里面的权重！
比如上面如果我实现一个固定CNN权重，只更改input权重的网络，比如style transfer
那么为了求取input的权重，CNN的require_grad也会变成True
但是因为CNN不进行优化，所以需要手动设置不更新weight等
'''

'''
在PyTorch实现中，autograd会随着用户的操作，记录生成当前variable的所有操作，
并由此建立一个有向无环图。用户每进行一个操作，相应的计算图就会发生改变。
更底层的实现中，图中记录了操作Function，每一个变量在图中的位置可通过其grad_fn
属性在图中的位置推测得到。在反向传播过程中，autograd沿着这个图从当前变量（
根节点z）溯源，可以利用链式求导法则计算所有叶子节点的梯度。
每一个前向传播操作的函数都有与之对应的反向传播函数用来计算输入的各个variable的
梯度，这些函数的函数名通常以Backward结尾。下面结合代码学习autograd的实现细节。
'''

x = t.ones(1)
b = t.rand(1, requires_grad = True)
w = t.rand(1, requires_grad = True)
y = w * x # 等价于y=w.mul(x)
z = y + b # 等价于z=y.add(b)
print(x.requires_grad,b.requires_grad,w.requires_grad)
# 因为w和b求导依赖于y和z，因此y，z自动设置求导
print(y.requires_grad,z.requires_grad)

print(x.is_leaf, w.is_leaf, b.is_leaf)
print(y.is_leaf,z.is_leaf)

print(z.grad_fn)
print(z.grad_fn.next_functions)

'''
PyTorch使用的是动态图，它的计算图在每次前向传播时都是从头开始构建，
所以它能够使用Python控制语句（如for、if等）根据需求创建计算图。
比如根据输入使用for 循环创建不同层数的NN
'''

'''
不需要反向传播的场景，设置require_grad为False能够避免划分显存，节省一半显存！
'''
# 在关闭grad的空间下创建！
# with语句主要是实现一个enter方法
with t.no_grad():
    x = t.ones(1)
    w = t.rand(1, requires_grad = True)
    y = x * w
# y依赖于w和x，虽然w.requires_grad = True，但是y的requires_grad依旧为False
print(x.requires_grad, w.requires_grad, y.requires_grad)

t.set_grad_enabled(True)
print(x.requires_grad, w.requires_grad, y.requires_grad)


x = t.ones(3, requires_grad=True)
w = t.rand(3, requires_grad=True)
y = x * w
# y依赖于w，而w.requires_grad = True
z = y.sum()
print(x.requires_grad, w.requires_grad, y.requires_grad)

# 非叶子节点grad计算完之后自动清空，y.grad是None
z.backward()
print(x.grad, w.grad, y.grad)

# 也就是说，torch会自动把非叶子节点，也就是最后output的梯度清空（因为只是临时依赖，不需要存储）
# 方法：可以使用torch的自动求导方法：
dy_dz = t.autograd.grad(z, y)
print(dy_dz)

# 另外的方法是使用hook注入，每次会运行到那里会调用
def hook_variable(grad):
    np.save("hook.npy",grad.numpy())

x = t.ones(3, requires_grad=True)
w = t.rand(3, requires_grad=True)
y = x * w
# 注册hook
hook_handle = y.register_hook(hook_variable)
z = y.sum()
z.backward()
hook_handle.remove()

data = np.load("hook.npy")
print(data)

# 如果自定义的函数不支持自动反向求导，可以自己写

from torch.autograd import Function


class MultiplyAdd(Function):

    @staticmethod
    def forward(ctx, w, x, b):
        ctx.save_for_backward(w, x)# 这里会保存变量！！！
        output = w * x + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        w, x = ctx.saved_tensors# 读取在forward阶段保存的变量
        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output * 1
        return grad_w, grad_x, grad_b

x = t.ones(1)
w = t.rand(1, requires_grad = True)
b = t.rand(1, requires_grad = True)
# 开始前向传播
z=MultiplyAdd.apply(w, x, b)
print("Z is",z)
# 开始反向传播
z.backward()

# x不需要求导，中间过程还是会计算它的导数，但随后被清空
x.grad, w.grad, b.grad


class Sigmoid(Function):

    @staticmethod
    def forward(ctx, x):
        output = 1 / (1 + t.exp(-x))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_x = output * (1 - output) * grad_output
        return grad_x