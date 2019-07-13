import numpy as np
from optimizers import optimizer
from net_utills import *


class layer:
    W = None
    b = None
    dropout = 1
    activation = None
    l2_regulaizer = 0

    def __init__(self, size, dropout=1, activation=relu, l2_regulaizer=0.1):
        self.activation = activation
        self.W = (1 / size[1]) * np.random.randn(*size)
        self.b = (1 / size[0]) * np.random.randn(size[0], 1)
        self.l2_regulaizer = l2_regulaizer
        self.dropout = dropout

    def get_reg_term(self):
        if self.l2_regulaizer != 0:
            return self.l2_regulaizer * np.sum(self.W * self.W, axis=-1)
        else:
            return ValueError('No regularization on this layer')


class NeuralNet:

    def __init__(self, input_dim):
        self.layers_dims = []
        self.layers_dims.append(input_dim)
        self.layers_num = 0
        self.layers = {}
        self.optimizer = None

    def add_layer(self, units, activation, dropout, l2_regulaizer):
        self.layers_dims.append(units)
        size = (units, self.layers_dims[self.layers_num])
        self.layers_num += 1
        self.layers['L' + str(self.layers_num)] = \
            layer(size=size, dropout=dropout, activation=activation,
                  l2_regulaizer=l2_regulaizer)

    def feed_forward(self, X, predict=False):
        cache = {}
        cache['X0'] = X
        X_l = X
        for l in range(1, self.layers_num + 1):
            layer_l = self.layers['L' + str(l)]
            Z = layer_l.W
            if not predict:
                if layer_l.dropout < 1:
                    D_l = np.random.rand(*layer_l.W.shape) > layer.dropout
                    Z = Z * D_l
                    cache['D' + str(l)] = D_l
            Z = Z.dot(X_l) + layer_l.b
            X_l = layer_l.activation(Z)
            if not predict:
                cache['X' + str(l)] = X_l
                cache['Z' + str(l)] = Z
        return X_l, cache

    def backprop(self, batch, cache):
        X_train, C_train = batch
        net_out = cache['X' + str(self.layers_num)]
        X_L_1 = cache['X' + str(self.layers_num - 1)]
        last_layer = self.layers['L' + str(self.layers_num)]

        grads = {}
        grads['dW' + str(self.layers_num)] = grad_W_cross_entropy(X_l_1=X_L_1, net_out=net_out, W_l=last_layer.W,
                                                                  labels=C_train)
        grads['db' + str(self.layers_num)] = grad_b_cross_entropy(net_out=net_out, labels=C_train)
        grads['dX' + str(self.layers_num)] = grad_X_cross_entropy(X_l_1=X_L_1, W_l=last_layer.W, labels=C_train)

        for l in range(self.layers_num - 1, 0, -1):
            layer_l = self.layers['L' + str(l)]
            X_L_1 = cache['X' + str(l - 1)]
            Z = cache['Z' + str(l)]
            if layer_l.activation == relu:
                V = grad_relu(Z) * grads['dX' + str(l + 1)]
            else:
                V = grad_tanh(Z) * grads['dX' + str(l + 1)]

            m = X_L_1.shape[1]
            grads['dW' + str(l)] = (1 / m) * V.dot(X_L_1.T)
            grads['db' + str(l)] = (1 / m) * np.sum(V, axis=1, keepdims=True)
            grads['dX' + str(l)] = layer_l.W.T.dot(V)  # TODO:  check W.T page 68

        return grads

    def update_params(self, grads, cache):
        for l in range(1, self.layers_num):
            layer_l = self.layers['L' + str(l)]
            # parameters update rule
            dW = grads['dW' + str(l)]
            if layer_l.dropout < 1:
                D_l = cache['D' + str(l)]
                layer_l.W = layer_l.W - self.optimizer.step(layer_num=l, grad=dW * D_l, param='dW')
            else:
                layer_l.W = layer_l.W - self.optimizer.step(layer_num=l, grad=dW, param='dW')

            layer_l.b = layer_l.b - self.optimizer.step(layer_num=l, grad=grads['db' + str(l)], param='db')
        del grads

    def predict(self, X):
        predictions, _ = self.feed_forward(X=X, predict=True)
        predictions = predictions.argmax(axis=0).reshape((-1, 1))
        return predictions

    def compile(self, optimizer):
        b = self.layers['L' + str(self.layers_num)].b
        self.layers['L' + str(self.layers_num)].b = np.zeros(b.shape)
        self.optimizer = optimizer

    def fit(self, X_train, C_train, X_val, C_val, epoch, batch_size):
        costs_train = []
        errors_train = []
        costs_val = []
        errors_val = []
        labels_train = np.argmax(C_train, axis=0).reshape((-1, 1))
        labels_val = np.argmax(C_val, axis=0).reshape((-1, 1))
        mini_batches = create_mini_batches(X_train=X_train, C_train=C_train, batch_size=batch_size)
        for itr in range(epoch):
            for (batch_x, batch_c) in mini_batches:
                A_l, cache = self.feed_forward(batch_x, predict=False)
                grads = self.backprop((batch_x, batch_c), cache)
                self.update_params(grads, cache)

            if itr % 10 == 0 or itr == epoch - 1:
                A_l, _ = self.feed_forward(X=X_train, predict=False)
                cost_train = cross_entropy(A_l, C_train)
                error_train = np.mean(self.predict(X=X_train) != labels_train)
                net_out, _ = self.feed_forward(X_val, predict=False)
                cost_val = cross_entropy(net_out, C_val)
                error_val = np.mean(self.predict(X_val) != labels_val)
                print('Train: error after {} iterations is {} and cost is {}'.format(itr, error_train, cost_train))
                print('Val: error after {} iterations is {} and cost is {}'.format(itr, error_val, cost_val))
                costs_val.append(cost_val)
                errors_val.append(error_val)
                costs_train.append(cost_train)
                errors_train.append(error_train)

        return costs_train, errors_train, costs_val, errors_val
