from book_given_function import load_mnist
import pickle
import numpy as np
import NN_basic as functions  # import defined function!

# print(x_train.shape)  # (60000, 784)
# print(t_train.shape)  # (60000,)
# print(x_test.shape)  # (10000, 784)
# print(t_test.shape)  # (10000,)

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, 
                                                  normalize=True,
                                                  one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(net, x):
    W1, W2, W3 = net['W1'], net['W2'], net['W3']
    b1, b2, b3 = net['b1'], net['b2'], net['b3']
    a1 = np.dot(x, W1) + b1
    z1 = functions.sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = functions.sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    output = functions.softmax(a3)
    return output


# batch data
x, t = get_data()
network = init_network()
batch_size = 100
accuracy = 0
# for loop: [0, 100], [100, 200]...
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  # find maximum values of each row
    # how many labels for "p" were correct (equal to origin some BATCH of "t")
    # np.sum() was used to do count
    accuracy += np.sum(p==t[i:i+batch_size])  # p and t were both BATCH!
print("Accuracy" + str(float(accuracy)/len(x)) )
