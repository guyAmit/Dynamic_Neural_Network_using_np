import numpy as np
from NeuralNetModel import NeuralNet
from net_utills import tanh, softmax

if __name__ == '__main__':
    net = NeuralNet(input_dim=5)
    net.add_layer(units=5, activation=tanh)
    net.add_layer(units=3, activation=tanh)
    net.add_layer(units=3, activation=softmax)

    x = np.random.randn(5, 1)
    c = np.array([0, 1, 0]).reshape((3, 1))
    d = (1 / 5 ** 2) * np.random.randn(5, 1)
    # net.gradient_test_X(X=x, C=c, d=d,epsilon=0.8, max_iter=10)

    d = {}
    d['W1'] = (1 / 5 ** 2) * np.random.randn(5, 5)
    d['b1'] = (1 / 5 ** 2) * np.random.randn(5, 1)
    d['W2'] = (1 / 5 ** 2) * np.random.randn(3, 5)
    d['b2'] = (1 / 3 ** 2) * np.random.randn(3, 1)
    d['W3'] = (1 / 3 ** 2) * np.random.randn(3, 3)
    d['b3'] = (1 / 3 ** 2) * np.random.randn(3, 1)

    net.gradient_test_theta(X=x, C=c, d=d, epsilon=0.5, max_iter=10)
