import numpy as np
import net_utills
from scipy.io import loadmat
from NeuralNetModel import layer, NeuralNet
from optimizers import optimizer


def load_data(filename=''):
    Data = {}
    loadmat(file_name=filename, mdict=Data)
    X_train = Data['Yt']
    Y_train = Data['Ct']
    X_val = Data['Yv']
    Y_val = Data['Cv']
    return X_train, Y_train, X_val, Y_val


if __name__ == '__main__':
    X_train, Y_train, X_val, Y_val = load_data(filename='SwissRollData.mat')
    net = NeuralNet(input_dim=2)
    net.add_layer(units=15, activation=net_utills.relu, l2_regulaizer=1, dropout=1)
    net.add_layer(units=15, activation=net_utills.relu, l2_regulaizer=1, dropout=1)
    net.add_layer(units=2, activation=net_utills.softmax, l2_regulaizer=1, dropout=1)
    sgd_optimizer = optimizer(Type='Momentum', layers_dims=net.layers_dims, learning_rate=0.01)
    net.compile(sgd_optimizer)
    costs_train, errors_train, costs_val, errors_val = \
        net.fit(X_train=X_train, C_train=Y_train, X_val=X_val, C_val=Y_val, epoch=100, batch_size=64)
