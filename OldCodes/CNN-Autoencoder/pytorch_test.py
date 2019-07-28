if __name__ == "__main__":
    import torch
    x = torch.Tensor([1.0])
    xx = x.cuda()
    print(xx)

    from torch.backends import cudnn
    print(cudnn.is_acceptable(xx))


    from torch.autograd import Variable

    x = Variable(torch.randn(1,10))
    prev_h = Variable(torch.randn(1,20))
    W_h = Variable(torch.randn(20,20))
    W_x = Variable(torch.randn(20,10))

    i2h = torch.mm(W_x,x.t())
    h2h = torch.mm(W_h,prev_h.t())
    next_h = i2h + h2h
    next_h = next_h.tanh()

    next_h.backward(torch.ones(1,20))

    print(next_h)
    print(x)
    print(W_h)
    print(W_x)
    print(prev_h)
    print(i2h)
    print(h2h)