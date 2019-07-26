from scipy.io import loadmat
from NeuralNetModel import NeuralNet
import numpy as np
import net_utills
from optimizers import optimizer


def load_data(filename=''):
    Data = {}
    loadmat(file_name=filename, mdict=Data)
    X_train = Data['Yt']
    C_train = Data['Ct']
    X_val = Data['Yv']
    C_val = Data['Cv']
    return X_train, C_train, X_val, C_val


def build_net(layers_dims, learning_rate):
    net = NeuralNet(layers_dims[0])
    for l in range(1, len(layers_dims)):
        net.add_layer(layers_dims[l], activation=net_utills.relu, dropout=1, l2_regulaizer=0.7)
    net.add_layer(layers_dims[-1], activation=net_utills.softmax, dropout=1, l2_regulaizer=0.7)
    adam_optimizer = optimizer(Type='Adam', layers_dims=net.layers_dims, learning_rate=learning_rate)
    net.compile(adam_optimizer)
    return net


def get_grad_test_results(data_name, class_num, input_size):
    X_train, C_train, X_val, C_val = load_data(data_name + '.mat')
    batch_sizes = [64, 128, 256]
    learning_rates = [0.01, 0.001, 0.001]
    epoch_nums = [20, 35, 50]
    layer_dims = [[input_size, 15, 15, class_num], [input_size, 5, 5, class_num]]
    results = np.zeros((54, 11))
    counter = 0
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for epoch_num in epoch_nums:
                for layer_dim in layer_dims:
                    net = build_net(layer_dim, learning_rate)
                    costs_train, succ_train, costs_val, succ_val = \
                        net.fit(
                            X_train=X_train, C_train=C_train,
                            X_val=X_val, C_val=C_val,
                            epoch=epoch_num, batch_size=batch_size, p=True)
                    last_index = len(costs_train) - 1
                    results[counter, :3] = [learning_rate, batch_size, epoch_num]
                    if len(layer_dim) == 5:
                        results[counter, 3:7] = layer_dim
                    else:
                        results[counter, 3:7] = layer_dim
                    results[counter, 7:] = [costs_train[last_index],
                                            succ_train[last_index],
                                            costs_val[last_index],
                                            succ_val[last_index]]
                    counter += 1

    name = 'task7/results {}.csv'.format(data_name)
    np.savetxt(name, results, delimiter=',', fmt='% 4f')

if __name__ == '__main__':
    get_grad_test_results('SwissRollData', class_num=2, input_size=2)
    # get_grad_test_results('PeaksData', class_num=5, input_size=2)
    # get_grad_test_results('GMMData', class_num=5, input_size=5)
