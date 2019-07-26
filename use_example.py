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


if __name__ == '__main__':
    data_path = 'GMMData.mat'
    X_train, Y_train, X_val, Y_val = load_data(filename=data_path)


    net = NeuralNet(input_dim=5)
    net.add_layer(units=40, activation=net_utills.tanh,
                  l2_regulaizer=0.7, dropout=0.8)
    net.add_layer(units=40, activation=net_utills.tanh)
    net.add_layer(units=20, activation=net_utills.tanh, dropout=0.8)
    net.add_layer(units=5, activation=net_utills.softmax)
    Adam_optimizer = optimizer(Type='Adam', layers_dims=net.layers_dims,
                               learning_rate=0.001)
    net.compile(Adam_optimizer)
    costs_train, succ_train, costs_val, succ_val = \
        net.fit(X_train=X_train, C_train=Y_train,
                X_val=X_val, C_val=Y_val, epoch=35,
                batch_size=64, p=True)
    plt.plot(range(len(succ_train)), succ_train, c='r')
    plt.plot(range(len(succ_val)), succ_val, c='b')
    plt.legend(['succ train', 'succ val'])
    plt.title('success percentage on {}'.format(data_path))
    plt.savefig('{}_net_succ.png'.format(data_path))
    plt.show()
    plt.plot(range(len(costs_train)), costs_train, c='r')
    plt.plot(range(len(costs_val)), costs_val, c='b')
    plt.legend(['cost train', 'cost val'])
    plt.title('cost {}'.format(data_path))
    plt.savefig('{}_net_cost.png'.format(data_path))
    plt.show()
