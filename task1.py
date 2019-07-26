import numpy as np
from scipy.io import loadmat
import softmax



# task 1
if __name__ == '__main__':
    soft_reg = softmax.softmax_reg(class_num=2, input_size=2)
    x = np.random.randn(2, 1)
    c = np.array([0, 1]).reshape((2, 1))

    d_x = np.random.randn(x.shape[0], 1)
    soft_reg.gradient_X_test(d=d_x, X=x, C=c, epsilon=0.8, max_iter=10)

    d_w = np.random.randn(*soft_reg.W.shape)
    soft_reg.gradient_W_test(d=d_w, X=x, C=c, epsilon=0.8, max_iter=10)
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
