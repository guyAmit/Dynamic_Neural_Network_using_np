import numpy as np
from scipy.io import loadmat
import softmax
import matplotlib.pyplot as plt


def load_data(filename=''):
    Data = {}
    loadmat(file_name=filename, mdict=Data)
    X_train = Data['Yt']
    C_train = Data['Ct']
    X_val = Data['Yv']
    C_val = Data['Cv']
    return X_train, C_train, X_val, C_val


if __name__ == '__main__':
    X_train, C_train, X_val, C_val = load_data(filename='SwissRollData.mat')
    soft_reg = softmax.softmax_reg(class_num=2, input_size=2)
    # print(soft_reg.cross_entropy(soft_reg.softmax(X_train.T.dot(soft_reg.W)), C_train))
    # print(soft_reg.grad_cross_entropy(X_val, soft_reg.net_forward(X_val), C_val))
    # print(X_val.shape)  # (2, None) (None, 2)
    # print(C_val.shape)  # (2, None)  (None, 2)
    d = np.random.randn(X_val.shape[0], 1)
    soft_reg.gradient_X_test(d=d, X=X_val, C=C_val, epsilon=0.8, max_iter=10)

    # d_w = np.random.randn(*soft_reg.W.shape)
    # soft_reg.gradient_W_test(d=d_w, X=X_val, C=C_val, epsilon=0.8, max_iter=10)
    #
    # costs_train, errors_train, costs_val, errors_val = soft_reg.fit(X_train=X_train, C_train=C_train, X_val=X_val,
    #                                                                 C_val=C_val,
    #                                                                 learning_rate=0.001, epoch=50,
    #                                                                 batch_size=128)
    # plt.plot(errors_val)
    # plt.plot(errors_train)
    # plt.legend(['errors_val', 'errors_train'])
    # plt.title('softmax_errors_{}'.format('SwissRollData'))
    # plt.savefig('imgs/softmax_errors_{}.png'.format('SwissRollData'))
    # plt.show()
    #
    # plt.plot(costs_val)
    # plt.plot(costs_train)
    # plt.legend(['costs_val', 'costs_train'])
    # plt.title('softmax_costs_{}'.format('SwissRollData'))
    # plt.savefig('imgs/softmax_costs_{}.png'.format('SwissRollData'))
    # plt.show()
