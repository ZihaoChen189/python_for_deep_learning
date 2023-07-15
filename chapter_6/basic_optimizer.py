import numpy as np

class SGD:
    def __init__(self, lr):
        self.lr = lr

    def udpate(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            # initialize on the matrix shape
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def udpate(self, params, grads):
        if self.h == None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Dropout:
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask  # for elements of self.mask were True(1)
        else:
            return x * (1 - self.dropout_ratio)
        
    def backward(self, dout):
        return dout * self.mask  # # for elements of self.mask were True(1)


'''
Note:
1. How to initialize the weight or bias matrix:
simple solution:1/sqrt(n), it is Xavier
for the Relu activation function: He initialize sqrt(2/n)

2. batch normalization:
normalize the data with mean as o, variance as 1

3. normalization:
L2, dropout(similar to Relu structure)

4. parameter:
weight_decay = 10 ** np.random.uniform(-8, -4)
lr = 10 ** np.random.uniform(-6, -2)

'''    
    