import numpy as np
import NeuralNetModel as nn
import matplotlib.pyplot as plt
from net_utills import *
from scipy.io import loadmat
import optimizers
import seaborn as sns

sns.set_style("darkgrid")


# Task 4
def jacovian_W_test(net, x, d, epsilon, max_iter):
    layer_1 = net.layers['L1']
    f_x = tanh(layer_1.W.dot(x) + layer_1.b)
    epsilons = [epsilon ** i for i in range(1, max_iter + 1)]
    tests = np.zeros((max_iter, 2))
    for i in range(0, max_iter):
        f_x_d = tanh((layer_1.W + (epsilons[i] * d).reshape(layer_1.W.shape)).dot(x) + layer_1.b)
        f_x_d_j = f_x + jacovian_w(layer_1.W, x, layer_1.b).dot((epsilons[i] * d))
        tests[i, 0] = np.linalg.norm(f_x_d - f_x)
        tests[i, 1] = np.linalg.norm(f_x_d - f_x_d_j)
    plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
    plt.title('Jacovian W Test')
    plt.legend(['quadratic result'])
    plt.xlabel('iterations')
    plt.ylabel('results')
    plt.savefig('imgs/neural net single/W quadratic result.png')
    plt.show()

    plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
    plt.title('Jacovian W Test')
    plt.legend(['linear result'])
    plt.xlabel('iterations')
    plt.ylabel('results')
    plt.savefig('imgs/neural net single/W linear result.png')
    plt.show()

    plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
    plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
    plt.title('Jacovian W Test')
    plt.legend(['quadratic result', 'linear result'])
    plt.xlabel('iterations')
    plt.ylabel('results')
    plt.savefig('imgs/neural net single/W quadratic result linear result.png')
    plt.show()


def jacobian_transpose_W_test(net, x, epsilon, max_iter):
    layer_1 = net.layers['L1']
    f_x = tanh(layer_1.W.dot(x) + layer_1.b)
    tests = np.zeros((max_iter, 1))
    v = (1 / 8**0.5) * np.random.randn(8, 1)
    u = (1 / 4**0.5) * np.random.randn(4, 1)
    Jac = jacovian_w(layer_1.W, x, layer_1.b)
    for i in range(0, max_iter):
        u *= epsilon
        v *= epsilon
        tests[i, 0] = np.abs(u.T.dot(Jac).dot(v) - v.T.dot(Jac.T).dot(u)) + 1e-10

    plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'b')
    plt.semilogy(range(1, max_iter + 1), [epsilon ** i for i in range(1, max_iter + 1)], 'r')
    plt.title('Jacovian Transpose W Test')
    plt.legend(['Jacovian', 'epsilon'])
    plt.xlabel('iterations')
    plt.ylabel('results')
    plt.savefig('imgs/neural net single/W Jacovian Transpose Test.png')
    plt.show()


def jacovian_b_test(net, x, d, epsilon, max_iter):
    layer_1 = net.layers['L1']
    f_x = tanh(layer_1.W.dot(x) + layer_1.b)
    epsilons = [epsilon ** i for i in range(1, max_iter + 1)]
    tests = np.zeros((max_iter, 2))
    for i in range(0, max_iter):
        f_x_d = tanh(layer_1.W.dot(x) + layer_1.b + epsilons[i] * d)
        f_x_d_j = f_x + jacovian_b(layer_1.W, x, layer_1.b).dot(epsilons[i] * d)
        tests[i, 0] = np.linalg.norm(f_x_d - f_x)
        tests[i, 1] = np.linalg.norm(f_x_d - f_x_d_j)

    plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
    plt.title('jacovian b Test')
    plt.legend(['quadratic result'])
    plt.xlabel('iterations')
    plt.ylabel('results')
    plt.savefig('imgs/neural net single/b single layer quadratic result.png')
    plt.show()

    plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
    plt.title('Jacovian b Test')
    plt.legend(['linear result'])
    plt.xlabel('iterations')
    plt.ylabel('results')
    plt.savefig('imgs/neural net single/b single layer linear result.png')
    plt.show()

    plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
    plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
    plt.title('b Jacovian Test')
    plt.legend(['quadratic result', 'linear result'])
    plt.xlabel('iterations')
    plt.ylabel('results')
    plt.savefig('imgs/neural net single/b single layer quadratic result linear result.png')
    plt.show()


def jacobian_transpose_b_test(net, x, epsilon, max_iter):
    layer_1 = net.layers['L1']
    f_x = tanh(layer_1.W.dot(x) + layer_1.b)
    tests = np.zeros((max_iter, 1))
    v = (1 / 2**0.5) * np.random.randn(4, 1)
    u = (1 / 4**0.5) * np.random.randn(4, 1)
    Jac = jacovian_b(layer_1.W, x, layer_1.b)
    for i in range(0, max_iter):
        u *= epsilon
        v *= epsilon
        tests[i, 0] = np.abs(u.T.dot(Jac).dot(v) - v.T.dot(Jac.T).dot(u)) + 1e-10

    plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'b')
    plt.semilogy(range(1, max_iter + 1), [epsilon ** i for i in range(1, max_iter + 1)], 'r')
    plt.title('Jacovian Transpose b Test')
    plt.legend(['Jacovian', 'epsilon'])
    plt.xlabel('iterations')
    plt.ylabel('results')
    plt.savefig('imgs/neural net single/b Jacovian Transpose Test.png')
    plt.show()


def jacobian_x_test(net, x, d, epsilon, max_iter):
    layer_1 = net.layers['L1']
    f_x = tanh(layer_1.W.dot(x) + layer_1.b)
    epsilons = [epsilon ** i for i in range(1, max_iter + 1)]
    tests = np.zeros((max_iter, 2))
    for i in range(0, max_iter):
        f_x_d = tanh(layer_1.W.dot(x + epsilons[i] * d) + layer_1.b)
        f_x_d_j = f_x + jacovian_x(layer_1.W, x, layer_1.b).dot(epsilons[i] * d)
        tests[i, 0] = np.linalg.norm(f_x_d - f_x)
        tests[i, 1] = np.linalg.norm(f_x_d - f_x_d_j)
    plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
    plt.title('Jacovian x Test')
    plt.legend(['quadratic result'])
    plt.xlabel('iterations')
    plt.ylabel('results')
    plt.savefig('imgs/neural net single/X quadratic result.png')
    plt.show()

    plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
    plt.title('Jacovian x Test')
    plt.legend(['linear result'])
    plt.xlabel('iterations')
    plt.ylabel('results')
    plt.savefig('imgs/neural net single/X linear result.png')
    plt.show()

    plt.semilogy(range(1, max_iter + 1), tests[:, 1], 'b')
    plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'r')
    plt.title('Jacovian x Test')
    plt.legend(['quadratic result', 'linear result'])
    plt.xlabel('iterations')
    plt.ylabel('results')
    plt.savefig('imgs/neural net single/X quadratic result linear result.png')
    plt.show()


def jacobian_transpose_x_test(net, x, epsilon, max_iter):
    layer_1 = net.layers['L1']
    f_x = tanh(layer_1.W.dot(x) + layer_1.b)
    tests = np.zeros((max_iter, 1))
    v = (1 / 2**0.5) * np.random.randn(2, 1)
    u = (1 / 4**0.5) * np.random.randn(4, 1)
    Jac = jacovian_x(layer_1.W, x, layer_1.b)
    for i in range(0, max_iter):
        u *= epsilon
        v *= epsilon
        tests[i, 0] = np.abs(u.T.dot(Jac).dot(v) - v.T.dot(Jac.T).dot(u)) + 1e-10

    plt.semilogy(range(1, max_iter + 1), tests[:, 0], 'b')
    plt.semilogy(range(1, max_iter + 1), [epsilon ** i for i in range(1, max_iter + 1)], 'r')
    plt.title('Jacovian Transpose x Test')
    plt.legend(['Jacovian', 'epsilon'])
    plt.xlabel('iterations')
    plt.ylabel('results')
    plt.savefig('imgs/neural net single/x Jacovian Transpose Test.png')
    plt.show()


def jacovian_b(W, x, b):
    return np.diagflat(grad_tanh(W.dot(x) + b))


def jacovian_w(W, x, b):
    return np.diagflat(grad_tanh(W.dot(x) + b)).dot(np.kron(x.T, np.identity(W.shape[0])))


def jacovian_T_w_mv(W, x, b, v):
    return np.diagflat(grad_tanh(W.dot(x) + b)).dot(v).dot(x.T)


def jacovian_x(W, x, b):
    return np.diagflat(grad_tanh(W.dot(x) + b)).dot(W)


if __name__ == '__main__':
    # X_train, C_train, X_val, C_val = load_data(filename='SwissRollData.mat')
    net = nn.NeuralNet(input_dim=2)
    net.add_layer(units=4, activation=tanh, dropout=1, l2_regulaizer=0)
    x = (1 / 2**0.5) * np.random.randn(2, 1)
    epsilon = 0.5
    d_x = (1 / 2**0.5) * np.random.randn(2, 1)
    d_b = (1 / 4**0.5) * np.random.randn(4, 1)
    d_w = (1 / (4 * 2)**0.5) * np.random.randn(4 * 2, 1)

    jacobian_x_test(net, x=x, d=d_x, epsilon=epsilon, max_iter=10)
    jacobian_transpose_x_test(net, x, epsilon, 10)

    jacovian_b_test(net, x=x, d=d_b, epsilon=epsilon, max_iter=10)
    jacobian_transpose_b_test(net, x=x, epsilon=epsilon, max_iter=10)

    jacovian_W_test(net, x=x, d=d_w, epsilon=epsilon, max_iter=10)
    jacobian_transpose_W_test(net, x=x, epsilon=epsilon, max_iter=10)