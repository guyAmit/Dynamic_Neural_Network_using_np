import numpy as np


def cross_entropy(net_out, labels):
    return -np.mean(np.sum(np.log(net_out) * labels, axis=0, keepdims=True), axis=1)


def JacT_mV_W(V_l, X_l_1):
    dW = V_l.dot(X_l_1.T)
    return dW


def JacT_mV_b(V_l):
    db = np.mean(V_l, axis=1, keepdims=True)
    return db


def grad_cross_entropy(net_out, labels):
    dX = net_out - labels
    return dX


def grad_relu(X):
    grad = X > 0
    return np.int64(grad)


def grad_tanh(linear_part):
    return 1 - np.tanh(linear_part) ** 2


def relu(v):
    return np.maximum(0, v)


def tanh(v):
    return np.tanh(v)


def softmax(v):
    return np.exp(v) / np.sum(np.exp(v), axis=0)


def create_mini_batches(X_train, C_train, batch_size):
    perm = np.random.permutation(range(X_train.shape[1]))
    X_train = X_train[:, perm]
    C_train = C_train[:, perm]
    mini_batches = []
    for idx in range(1, X_train.shape[1] // batch_size):
        if idx * batch_size >= X_train.shape[1]:
            batch_x = X_train[:, (idx - 1) * batch_size:]
            batch_c = C_train[:, (idx - 1) * batch_size:]
            mini_batches.append((batch_x, batch_c))
        else:
            batch_x = X_train[:, (idx - 1) * batch_size: idx * batch_size]
            batch_c = C_train[:, (idx - 1) * batch_size: idx * batch_size]
            mini_batches.append((batch_x, batch_c))
    return mini_batches
