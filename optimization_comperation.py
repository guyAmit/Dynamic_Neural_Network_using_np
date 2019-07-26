import net_utills
from scipy.io import loadmat
from NeuralNetModel import NeuralNet
from optimizers import optimizer
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")


def load_data(filename=''):
    Data = {}
    loadmat(file_name=filename, mdict=Data)
    X_train = Data['Yt']
    Y_train = Data['Ct']
    X_val = Data['Yv']
    Y_val = Data['Cv']
    return X_train, Y_train, X_val, Y_val


def create_net(optimizer_type, layers_dims, input_size, output_size, batch_size, learning_rate, epochs, data):
    X_train, Y_train, X_val, Y_val = data
    net1 = NeuralNet(input_dim=input_size)
    for l in layers_dims:
        net1.add_layer(units=l, activation=net_utills.relu)
    net1.add_layer(units=output_size, activation=net_utills.softmax)
    optimization_method = optimizer(Type=optimizer_type, layers_dims=net1.layers_dims, learning_rate=learning_rate)
    net1.compile(optimization_method)
    costs_train, succ_train, costs_val, succ_val = \
        net1.fit(X_train=X_train, C_train=Y_train, X_val=X_val, C_val=Y_val, epoch=epochs, batch_size=batch_size,
                 p=True)
    return costs_train, succ_train, costs_val, succ_val


if __name__ == '__main__':
    data_path = 'SwissRollData.mat'
    data = load_data(filename=data_path)

    optimization_methods = ['SGD', 'Momentum', 'Adam']
    for method in optimization_methods:
        costs_train, _, _, _ = create_net(optimizer_type=method, data=data, layers_dims=[5, 5], input_size=2,
                                          output_size=2, batch_size=64, learning_rate=0.001, epochs=20)
        plt.plot(range(len(costs_train)), costs_train)
        if method == 'Adam':
            plt.legend(['SGD', 'Momentum', 'Adam'])
            plt.title('cost {}'.format(data_path))
            plt.savefig('task9/{}_net_cost.png'.format(data_path))
            plt.show()
