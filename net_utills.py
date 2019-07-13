import numpy as np


def cross_entropy(net_out, labels):
    return -np.mean(np.sum(np.log(net_out) * labels, axis=0), axis=0)


def grad_W_cross_entropy(X_l_1, net_out, W_l, labels):
    dW = np.zeros(W_l.shape)
    for k in range(W_l.shape[0]):
        dW[k, :] = np.mean(X_l_1 * (net_out[k, :] - labels[k, :]), axis=1).T
    return dW


def grad_b_cross_entropy(net_out, labels):
    db = np.mean((net_out - labels), axis=1)
    return db


def grad_X_cross_entropy(X_l_1, W_l, labels):
    m = X_l_1.shape[1]
    dX = (-1 / m) * W_l.T.dot(np.exp(W_l.dot(X_l_1)) / np.sum(np.exp(W_l.dot(X_l_1)), axis=0) - labels)
    return dX


def grad_relu(liner_part):
    Z = np.array(liner_part, copy=True)
    Z[liner_part <= 0] = 0
    Z[liner_part > 0] = 1
    return Z


def grad_tanh(linear_part):
    return 1 - np.tanh(linear_part)**2


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
