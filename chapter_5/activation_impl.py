import numpy as np

def softmax(x):
    help = np.max(x)
    exp_a = np.exp(x - help)  # slove nan question
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):
    if y.ndim == 1:  # simple change for one dimension data
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7)) / batch_size


class Relu:
    def __init__(self):
        self.mask = None  # save the location of variables who <= 0

    def forward(self, x):
        self.mask = (x<=0)  # True: <=0; False: >0
        out = x.copy()
        out[self.mask] = 0  # works!
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0  # same location of those variables
        dx = dout
        return dx
    

class Sigmoid:
    def __init__(self):
        # save the output value of forward(), which would be used in backward()
        self.out = None  

    def forward(self, x):
        out = 1 / (1+np.exp(-x))
        return out
    
    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)  # remember the graph! not dx!
        return dx
    

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None  # save the loss
        self.y = None  # the softmax output
        self.t = None  # one-hot vector

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # x was still the original x
        it.iternext()   
        
    return grad
