import numpy as np
import NN_basic as functions  # import defined function!

def init_network():
    net = {}
    net['W1'] = np.array([[1, 3, 5], [2, 4, 6]])  # (2, 3)
    net['b1'] = np.array([1, 2, 3])  # (3, )
    net['W2'] = np.array([[1, 5], [2, 6], [3, 8]])  # (3, 2)
    net['b2'] = np.array([1, 2])  # (2, )
    net['W3'] = np.array([[1, 2], [3, 4]])  # (2, 2)
    net['b3'] = np.array([1, 5])  # (2, )
    return net


def forward(net, x):
    W1, W2, W3 = net['W1'], net['W2'], net['W3']
    b1, b2, b3 = net['b1'], net['b2'], net['b3']
    a1 = np.dot(x, W1) + b1
    z1 = functions.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = functions.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    output = functions.relu(a3)  # just random one
    return output


#
network = init_network()
x = np.array([1, 5])
y = forward(network, x)
print(y)
