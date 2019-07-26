# Final Project: Deep Learning from Scratch 

## Table of Contents

- [About](#about)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Creators](#creators)

## About <a name = "about"></a>

Deep sqeuential - easy to use - nueral network implamented using numpy.

### Network Features
- L2 regulaizer
- Dropout
- Optimizer (Adam, Momentum, SGD)
 
## Prerequisites <a name = "prerequisites"></a>

libraries that should be installed in order to use the network: 
```
numpy
seaborn
matplotlib
```

## Usage <a name = "usage"></a>
### Network Inputs
- Input dimenstions
- Units - how many neurons will be in the layer
- Activation function (relu, tanh, sofmax)
- L2 regulaizer
- Dropout
- Optimizer (Adam, Momentum, SGD)
- Leaning rate
- Batch size
- Number of epoches
</br>
</br>
</br>
</br>
</br>
</br>
</br>

### Usage Example:
```
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
   
```

## Creators <a name = "creators"></a>
- [Guy Amit](https://github.com/guyAmit)
- [Yuval Lahav]()