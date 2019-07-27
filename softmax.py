import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from net_utills import create_mini_batches

sns.set_style("darkgrid")


class softmax_reg:
    def __init__(self, class_num, input_size):
        self.W = (1 / input_size) * np.random.randn(input_size, class_num)

    def softmax(self, v):
        return np.exp(v) / np.sum(np.exp(v), keepdims=True, axis=1)

    def net_forward(self, X):
        return self.softmax(X.T.dot(self.W))

    def predict(self, X):
        return np.argmax(self.net_forward(X), axis=1).reshape(-1, 1)

    def cross_entropy(self, net_out, labels):
        m = net_out.shape[0]
        return (-1 / m) * np.sum(np.sum(np.log(net_out) * labels.T, axis=1, keepdims=True), axis=0)

    def grad_W_cross_entropy(self, X, net_out, labels):
        dW = np.zeros(self.W.shape)
        for k in range(self.W.shape[1]):
            dW[:, k] = np.mean(X * (net_out[:, k] - labels.T[:, k]).T, axis=1)
        return dW

    def grad_X_cross_entropy(self, X, labels):
        m = X.shape[1]
        _sum = np.sum(np.exp(self.W.T.dot(X)))
        dX = (1 / m) * self.W.dot(np.exp(self.W.T.dot(X)) / _sum - labels)
        return dX

    # Task 1
    def gradient_W_test(self, X, C, d, epsilon, max_iter):
        f_x = self.cross_entropy(self.net_forward(X), C)
        epsilons = [epsilon ** i for i in range(1, max_iter + 1)]
        tests = np.zeros((max_iter, 2))
        W = self.W.copy()
        for i in range(0, max_iter):
            self.W = self.W + epsilons[i] * d
            f_x_d = self.cross_entropy(self.net_forward(X), C)
            self.W = W
            f_x_d_g = f_x + np.dot((epsilons[i] * d).reshape((-1, 1)).T,
                                   self.grad_W_cross_entropy(X, self.net_forward(X), C).reshape((-1, 1)))
            tests[i, 0] = np.abs(f_x_d - f_x)
            tests[i, 1] = np.abs(f_x_d - f_x_d_g)

        plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
        plt.title('Gradient Test - W')
        plt.legend(['quadratic result'])
        plt.xlabel('iterations')
        plt.ylabel('results')
        # plt.savefig('imgs/softmax/W quadratic result.png')
        plt.show()

        plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
        plt.title('Gradient Test - W')
        plt.legend(['linear result'])
        plt.xlabel('iterations')
        plt.ylabel('results')
        # plt.savefig('imgs/softmax/W linear result.png')
        plt.show()

        plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
        plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
        plt.title('Gradient Test - W')
        plt.legend(['quadratic result', 'linear result'])
        plt.xlabel('iterations')
        plt.ylabel('results')
        # plt.savefig('imgs/softmax/W quadratic result linear result.png')
        plt.show()

    # Task 1
    def gradient_X_test(self, d, X, C, epsilon, max_iter):
        net_out = self.net_forward(X)
        f_x = self.cross_entropy(net_out, C)
        epsilons = [epsilon ** i for i in range(1, max_iter + 1)]
        tests = np.zeros((max_iter, 2))
        for i in range(0, max_iter):
            f_x_d = self.cross_entropy(self.net_forward(X + epsilons[i] * d), C)
            f_x_d_g = f_x + np.dot((epsilons[i] * d).T, self.grad_X_cross_entropy(X, C))
            tests[i, 0] = np.abs(f_x_d - f_x)
            tests[i, 1] = np.abs(f_x_d - f_x_d_g)

        plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
        plt.title('Gradient Test - X')
        plt.legend(['quadratic result'])
        plt.xlabel('iterations')
        plt.ylabel('results')
        # plt.savefig('imgs/softmax/X quadratic result.png')
        plt.show()

        plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
        plt.title('Gradient Test - X')
        plt.legend(['linear result'])
        plt.xlabel('iterations')
        plt.ylabel('results')
        # plt.savefig('imgs/softmax/X linear result.png')
        plt.show()

        plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
        plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
        plt.title('Gradient Test- X')
        plt.legend(['quadratic result', 'linear result'])
        plt.xlabel('iterations')
        plt.ylabel('results')
        # plt.savefig('imgs/softmax/X quadratic result linear result.png')
        plt.show()

    # Task 2_3
    def fit(self, X_train, C_train, X_val, C_val, learning_rate, epoch, batch_size, p=False):
        costs_train = []
        succs_train = []
        costs_val = []
        succs_val = []
        labels_train = np.argmax(C_train, axis=0).reshape((-1, 1))
        labels_val = np.argmax(C_val, axis=0).reshape((-1, 1))
        mini_batches = create_mini_batches(X_train, C_train, batch_size=batch_size)
        for itr in range(epoch):
            for (batch_x, batch_c) in mini_batches:
                self.W = self.W - learning_rate * self.grad_W_cross_entropy(batch_x, self.net_forward(batch_x), batch_c)
            if itr % 1 == 0 or itr == epoch - 1:
                cost_train = self.cross_entropy(self.net_forward(X_train), C_train)
                succ_train = np.mean(self.predict(X_train) == labels_train)
                cost_val = self.cross_entropy(self.net_forward(X_val), C_val)
                succ_val = np.mean(self.predict(X_val) == labels_val)
                if p == True:
                    print('Train: succ after {} iterations is {} and cost is {}'.format(itr, succ_train, cost_train))
                    print('Val: succ after {} iterations is {} and cost is {}'.format(itr, succ_val, cost_val))
                costs_val.append(cost_val)
                succs_val.append(succ_val)
                costs_train.append(cost_train)
                succs_train.append(succ_train)

        return costs_train, succs_train, costs_val, succs_val
