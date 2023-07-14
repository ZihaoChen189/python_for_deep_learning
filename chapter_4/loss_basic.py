import numpy as np

def mean_squared_error(y, t):
    return 0.5*np.sum((y-t)**2)


def sigmoid(x):
    return 1/(1+np.exp(-x)) 


def cross_entropy_error_bad_one(y, t):
    delta = 1e-7  # program protection
    return -np.sum(t*np.log(y+delta))  


def cross_entropy_error(y, t):
    if y.ndim == 1:  # simple change for one dimension data
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t*np.log(y+1e-7)) / batch_size


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h)) / (2*h)


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


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x


def softmax(x):
    help = np.max(x)
    exp_a = np.exp(x - help)  # slove nan question
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
