import numpy as np


class optimizer:

    def __init__(self, Type, layers_dims, learning_rate):
        self.learning_rate = learning_rate
        self.type = Type

        if Type == 'Momentum':
            self.beta = 0.9
            self.V = {}
            for l in range(1, len(layers_dims)):
                self.V['dW' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
                self.V['db' + str(l)] = np.zeros((layers_dims[l], 1))

        if Type == 'Adam':
            self.beta1 = 0.9
            self.beta2 = 0.99
            self.epsilon = 1e-6
            self.V = {}
            self.S = {}
            self.t = 0
            for l in range(1, len(layers_dims)):
                self.V['dW' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
                self.V['db' + str(l)] = np.zeros((layers_dims[l], 1))
                self.S['dW' + str(l)] = np.zeros((layers_dims[l], layers_dims[l - 1]))
                self.S['db' + str(l)] = np.zeros((layers_dims[l], 1))

    def step(self, layer_num, grad, param=None):
        if self.type == 'SGD':
            return self.learning_rate * grad
        if self.type == 'Momentum':
            self.V[param + str(layer_num)] = self.beta * self.V[param + str(layer_num)] + (1 - self.beta) * grad
            return self.learning_rate * self.V[param + str(layer_num)]
        if self.type == 'Adam':
            self.t += 1
            self.V[param + str(layer_num)] = self.beta1 * self.V[param + str(layer_num)] + (1 - self.beta1) * grad
            V_corrected = self.V[param + str(layer_num)] / (1 - np.power(self.beta1, self.t))
            self.S[param + str(layer_num)] = self.beta2 * self.S[param + str(layer_num)] + \
                                             (1 - self.beta2) * np.power(grad, 2)
            S_corrected = self.S[param + str(layer_num)] / (1 - np.power(self.beta2, self.t))
            return self.learning_rate * V_corrected / (self.epsilon + np.sqrt(S_corrected))
