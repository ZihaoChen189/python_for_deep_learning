# step function for the "numpy array" not just a single variable:
import numpy as np

def step_function(x):
    # y = x > 0  # BOOL list for which one was bigger than x ?
    # return y.astype(np.int)  # converted as int list not bool list
    return np.array(x > 0, dtype = int)  # excellent one!

# figure
# import matplotlib.pylab as plt
# x = np.arange(-5.0, 5.0, 0.1)
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)  # ylimit, limit the size of y-axis
# plt.show()


def sigmoid(x):
    return 1/(1+np.exp(-x))  # computation happened on every variable

# Both the step function and sigmoid function would output between 0 and 1.
# They were both non-linear functions -> neutral network cannot use the linear function as
# activation functions.
# They would BOTH increase with the input increased just in different shapes.


def relu(x):
    return np.maximum(0, x)  # not just max!

# warning: np.dot(A, B) was TOTALLY different from A*B


def softmax(x):
    help = np.max(x)
    exp_a = np.exp(x - help)  # nan question
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
