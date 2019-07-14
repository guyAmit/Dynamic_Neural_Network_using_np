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
        X, Y = batch
        m = X.shape[1]
        grads = {}
        X_l = cache['X' + str(self.layers_num)]
        X_l_1 = cache['X' + str(self.layers_num - 1)]
        V_l = grad_cross_entropy(X_l, Y)
        grads['dW' + str(self.layers_num)] = JacT_mV_W(V_l=V_l, X_l_1=X_l_1)
        grads['db' + str(self.layers_num)] = JacT_mV_b(V_l=V_l)

        for l in reversed(range(1, self.layers_num)):
            layer_p_1 = self.layers['L' + str(l + 1)]
            layer_l = self.layers['L' + str(l)]
            X_l = cache['X' + str(l)]
            X_l_1 = cache['X' + str(l - 1)]
            # ToDo : add dropout to W
            V_l = layer_p_1.W.T.dot(V_l)
            if layer_l.activation == relu:
                V_l = np.multiply(V_l, np.int64(X_l > 0))
            else:
                V_l = np.multiply(V_l, grad_tanh(X_l))
            # ToDo : add L2 to W
            dW = JacT_mV_W(V_l=V_l, X_l_1=X_l_1)
            db = JacT_mV_b(V_l=V_l)

            assert dW.shape == layer_l.W.shape
            assert db.shape == layer_l.b.shape
            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db
        return grads


    def update_params(self, grads, cache):
        for l in range(1, self.layers_num + 1):
            layer_l = self.layers['L' + str(l)]
            # parameters update rule
            dW = grads['dW' + str(l)]
            if layer_l.dropout < 1:
                D_l = cache['D' + str(l)]
                self.layers['L' + str(l)].W = layer_l.W - self.optimizer.step(layer_num=l, grad=dW * D_l, param='dW')
            else:
                self.layers['L' + str(l)].W = layer_l.W - self.optimizer.step(layer_num=l, grad=dW, param='dW')

            self.layers['L' + str(l)].b = layer_l.b - self.optimizer.step(layer_num=l,
                                                                          grad=grads['db' + str(l)], param='db')
        del grads

    def predict(self, X):
        predictions, _ = self.feed_forward(X=X, predict=True)
        predictions = predictions.argmax(axis=0).reshape((-1, 1))
        return predictions

    def compile(self, optimizer):
        self.optimizer = optimizer

    def fit(self, X_train, C_train, X_val, C_val, epoch, batch_size, p=False):
        costs_train = []
        succs_train = []
        costs_val = []
        succs_val = []
        labels_train = np.argmax(C_train, axis=0).reshape((-1, 1))
        labels_val = np.argmax(C_val, axis=0).reshape((-1, 1))
        mini_batches = create_mini_batches(X_train=X_train, C_train=C_train, batch_size=batch_size)
        for itr in range(epoch):
            for (batch_x, batch_c) in mini_batches:
                A_l, cache = self.feed_forward(batch_x, predict=False)
                grads = self.backprop((batch_x, batch_c), cache)
                self.update_params(grads, cache)

            if p and itr % 10 == 0 or itr == epoch - 1:
                A_l, _ = self.feed_forward(X=X_train, predict=False)
                cost_train = cross_entropy(A_l, C_train)
                succ_train = np.mean(self.predict(X=X_train) == labels_train)
                net_out, _ = self.feed_forward(X_val, predict=False)
                cost_val = cross_entropy(net_out, C_val)
                succ_val = np.mean(self.predict(X_val) == labels_val)
                print('Train: succ after {} iterations is {} and cost is {}'.format(itr, succ_train, cost_train))
                print('Val: succ after {} iterations is {} and cost is {}'.format(itr, succ_val, cost_val))
                costs_val.append(cost_val)
                succs_val.append(succ_val)
                costs_train.append(cost_train)
                succs_train.append(succ_train)

        return costs_train, succs_train, costs_val, succs_val
