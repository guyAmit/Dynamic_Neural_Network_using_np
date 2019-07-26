import numpy as np
import softmax
import seaborn as sns
from scipy.io import loadmat
import matplotlib.pyplot as plt
sns.set_style("darkgrid")


def load_data(filename=''):
    Data = {}
    loadmat(file_name=filename, mdict=Data)
    X_train = Data['Yt']
    C_train = Data['Ct']
    X_val = Data['Yv']
    C_val = Data['Cv']
    return X_train, C_train, X_val, C_val


def get_grad_test_results(data_name, class_num, input_size):
    X_train, C_train, X_val, C_val = load_data(data_name + '.mat')
    batch_sizes = [64, 128, 256]
    learning_rates = [0.1, 0.01, 0.001]
    epoch_nums = [50, 100, 150]
    results = np.zeros((27, 7))
    counter = 0
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for epoch_num in epoch_nums:
                regressor = softmax.softmax_reg(class_num=class_num, input_size=input_size)
                costs_train, succ_train, costs_val, succ_val = \
                    regressor.fit(X_train, C_train, X_val, C_val, learning_rate, epoch_num, batch_size, p=False)
                last_index = len(costs_train) - 1
                results[counter, :] = [learning_rate, batch_size, epoch_num,
                                          costs_train[last_index],
                                          succ_train[last_index],
                                          costs_val[last_index],
                                          succ_val[last_index]]
                counter += 1
    name = 'task23/ results {}.csv'.format(data_name)
    np.savetxt(name, results, delimiter=',', fmt='% 4f')

if __name__ == '__main__':

    get_grad_test_results('SwissRollData', class_num=2, input_size=2)
    get_grad_test_results('PeaksData', class_num=5, input_size=2)
    get_grad_test_results('GMMData', class_num=5, input_size=5)
    # X_train, C_train, X_val, C_val = load_data('GMMData.mat')
    # soft_reg = softmax.softmax_reg(class_num=5, input_size=5)
    # costs_train, succ_train, cost_val, succ_val = soft_reg.fit(X_train=X_train, C_train=C_train, X_val=X_val,
    #                                                                 C_val=C_val,
    #                                                                 learning_rate=0.001, epoch=100,
    #                                                                 batch_size=128)
    # plt.plot(range(len(succ_train)), succ_train, c='r')
    # plt.plot(range(len(succ_val)), succ_val, c='b')
    # plt.legend(['succ train', 'succ val'])
    # plt.title('task23/ success percentage on {}'.format('GMMData'))
    # plt.savefig('task23/ success percentage on {}'.format('GMMData'))
    # plt.show()
    # plt.plot(range(len(costs_train)), costs_train, c='r')
    # plt.plot(range(len(cost_val)), cost_val, c='b')
    # plt.legend(['cost train', 'cost val'])
    # plt.title('task23/ costs percentage on {}'.format('GMMData'))
    # plt.savefig('task23/ costs percentage on {}'.format('GMMData'))
    # plt.show()
