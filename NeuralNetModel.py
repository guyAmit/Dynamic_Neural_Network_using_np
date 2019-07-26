import numpy as np
from optimizers import optimizer
from net_utills import *
import matplotlib.pyplot as plt
import copy
import seaborn as sns

sns.set_style('darkgrid')


class layer:
    W = None
    b = None
    dropout = 1
    activation = None
    l2_regulaizer = 0

    def __init__(self, size, activation, dropout=1, l2_regulaizer=0.1):
        self.activation = activation
        self.W = (1 / size[1]) * np.random.randn(*size)
        self.b = (1 / size[0]) * np.random.randn(size[0], 1)
        self.l2_regulaizer = l2_regulaizer
        self.dropout = dropout

    def get_reg_term(self):
        if self.l2_regulaizer != 0:
            return self.l2_regulaizer * np.sum(self.W * self.W)
        return 0


class NeuralNet:

    def __init__(self, input_dim):
        self.layers_dims = []
        self.layers_dims.append(input_dim)
        self.layers_num = 0
        self.layers = {}
        self.optimizer = None

    def add_layer(self, units, activation, dropout=1, l2_regulaizer=0):
        self.layers_dims.append(units)
        size = (units, self.layers_dims[self.layers_num])
        self.layers_num += 1
        self.layers['L' + str(self.layers_num)] = \
            layer(size=size, activation=activation, dropout=dropout,
                  l2_regulaizer=l2_regulaizer)

    def feed_forward(self, X, predict=False):
        cache = {}
        cache['X0'] = X
        X_l = X
        for l in range(1, self.layers_num + 1):
            layer_l = self.layers['L' + str(l)]
            Z = layer_l.W.dot(X_l) + layer_l.b
            X_l = layer_l.activation(Z)
            if not predict:
                if layer_l.dropout < 1:
                    D_l = np.random.rand(*X_l.shape) < layer.dropout
                    X_l = np.multiply(X_l, D_l)
                    X_l /= layer_l.dropout
                    cache['D' + str(l)] = D_l
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

        # apply dropout derivative
        if self.layers['L' + str(self.layers_num)].dropout < 1:
            V_l = np.multiply(V_l, cache['D' + str(self.layers_num)])
            V_l /= self.layers['L' + str(self.layers_num)].dropout

        grads['dW' + str(self.layers_num)] = JacT_mV_W(V_l=V_l, X_l_1=X_l_1)
        grads['db' + str(self.layers_num)] = JacT_mV_b(V_l=V_l)

        for l in reversed(range(1, self.layers_num)):
            layer_p_1 = self.layers['L' + str(l + 1)]
            layer_l = self.layers['L' + str(l)]
            Z_l = cache['Z' + str(l)]
            X_l_1 = cache['X' + str(l - 1)]
            V_l = layer_p_1.W.T.dot(V_l)

            # apply dropout derivative
            if self.layers['L' + str(l)].dropout < 1:
                V_l = np.multiply(V_l, cache['D' + str(l)])
                V_l /= self.layers['L' + str(l)].dropout

            if layer_l.activation == relu:
                V_l = np.multiply(V_l, np.int64(Z_l > 0))
            else:
                V_l = np.multiply(V_l, grad_tanh(Z_l))

            dW = JacT_mV_W(V_l=V_l, X_l_1=X_l_1)
            # apply L2 reg
            if layer_l.l2_regulaizer != 0:
                dW += (2 / m) * layer_l.l2_regulaizer * layer_l.W
            db = JacT_mV_b(V_l=V_l)

            assert dW.shape == layer_l.W.shape
            assert db.shape == layer_l.b.shape
            grads['dW' + str(l)] = dW
            grads['db' + str(l)] = db

        grads['dX0'] = self.layers['L1'].W.T.dot(V_l)

        return grads

    def update_params(self, grads, cache):
        for l in range(1, self.layers_num + 1):
            layer_l = self.layers['L' + str(l)]
            # parameters update rule
            self.layers['L' + str(l)].W = layer_l.W - self.optimizer.step(layer_num=l, grad=grads['dW' + str(l)],
                                                                          param='dW')

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

            if p and (itr % 1 == 0 or itr == epoch - 1):
                A_l, _ = self.feed_forward(X=X_train, predict=False)
                cost_train = cross_entropy(self, A_l, C_train)
                succ_train = np.mean(self.predict(X=X_train) == labels_train)
                net_out, _ = self.feed_forward(X_val, predict=False)
                cost_val = cross_entropy(self, net_out, C_val)
                succ_val = np.mean(self.predict(X_val) == labels_val)
                print('Train: succ after {} iterations is {} and cost is {}'.format(itr, succ_train, cost_train))
                print('Val: succ after {} iterations is {} and cost is {}'.format(itr, succ_val, cost_val))
                costs_val.append(cost_val)
                succs_val.append(succ_val)
                costs_train.append(cost_train)
                succs_train.append(succ_train)

        return costs_train, succs_train, costs_val, succs_val

    def gradient_test_theta(self, X, C, d, epsilon, max_iter):
        net_out, cache = self.feed_forward(X, predict=False)
        f_x = cross_entropy(self, net_out, labels=C)
        grads = self.backprop((X, C), cache)

        epsilons = [epsilon ** i for i in range(1, max_iter + 1)]
        tests = np.zeros((max_iter, 2))
        layers = {}
        for l in range(1, self.layers_num + 1):
            L = layer(size=(self.layers_dims[l], self.layers_dims[l - 1]), activation=tanh)
            L.W = self.layers['L' + str(l)].W.copy()
            L.b = self.layers['L' + str(l)].b.copy()
            layers['L' + str(l)] = L

        for i in range(0, max_iter):

            for l in range(1, self.layers_num+1):
                self.layers['L' + str(l)].W += d['W' + str(l)] * epsilons[i]
                self.layers['L' + str(l)].b += d['b' + str(l)] * epsilons[i]

            net_out, cache = self.feed_forward(X, predict=False)
            f_x_p_d = cross_entropy(self, net_out, labels=C)

            for l in range(1, self.layers_num+1):
                self.layers['L' + str(l)].W = layers['L' + str(l)].W.copy()
                self.layers['L' + str(l)].b = layers['L' + str(l)].b.copy()


            f_x_p_g = f_x.copy()
            for l in range(1, self.layers_num+1):
                f_x_p_g += np.dot((epsilons[i] * d['W' + str(l)]).reshape((-1, 1)).T,
                                  grads['dW' + str(l)].reshape((-1, 1))).reshape((1,))
                f_x_p_g += (epsilons[i] * d['b' + str(l)]).T.dot(grads['db' + str(l)]).reshape((1,))

            tests[i, 0] = np.abs(f_x_p_d - f_x)
            tests[i, 1] = np.abs(f_x_p_d - f_x_p_g)

        plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
        plt.title('Gradient Test - Theta')
        plt.legend(['quadratic result'])
        plt.xlabel('iterations')
        plt.ylabel('results')
        plt.savefig('task6/Theta quadratic result.png')
        plt.show()

        plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
        plt.title('Gradient Test - Theta')
        plt.legend(['linear result'])
        plt.xlabel('iterations')
        plt.ylabel('results')
        plt.savefig('task6/Theta linear result.png')
        plt.show()

        plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
        plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
        plt.title('Gradient Test - Theta')
        plt.legend(['quadratic result', 'linear result'])
        plt.xlabel('iterations')
        plt.ylabel('results')
        plt.savefig('task6/Theta quadratic result linear result.png')
        plt.show()

    def gradient_test_X(self, X, C, d, epsilon, max_iter):
        net_out, cache = self.feed_forward(X, predict=False)
        f_x = cross_entropy(self, net_out, labels=C)
        grads = self.backprop((X, C), cache)

        epsilons = [epsilon ** i for i in range(1, max_iter + 1)]
        tests = np.zeros((max_iter, 2))
        for i in range(0, max_iter):
            f_x_d = cross_entropy(self, self.feed_forward(X + epsilons[i] * d, predict=False)[0], labels=C)
            f_x_d_g = f_x + np.dot((epsilons[i] * d).T, grads['dX0'])
            tests[i, 0] = np.abs(f_x_d - f_x)
            tests[i, 1] = np.abs(f_x_d - f_x_d_g)

        plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
        plt.title('Gradient Test - X')
        plt.legend(['quadratic result'])
        plt.xlabel('iterations')
        plt.ylabel('results')
        plt.savefig('task6/X quadratic result.png')
        plt.show()

        plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
        plt.title('Gradient Test - X')
        plt.legend(['linear result'])
        plt.xlabel('iterations')
        plt.ylabel('results')
        plt.savefig('task6/X linear result.png')
        plt.show()

        plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
        plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
        plt.title('Gradient Test- X')
        plt.legend(['quadratic result', 'linear result'])
        plt.xlabel('iterations')
        plt.ylabel('results')
        plt.savefig('task6/X quadratic result linear result.png')
        plt.show()
