import os
os.sys.path.extend([os.pardir, os.curdir])

import numpy as np

from dataset.mnist import load_mnist
from networks.two_layer import TwoLayer
# from networks.two_layer_naive import TwoLayer
from networks.simple_cnn import SimpleCNN
from common.optimizer import SGD, Momentum, AdaGrad, RMSProp, Adam, Nesterov


def train(network, x_train, y_train, x_test, y_test, iter_times=10000, hidden_size=10, batch_size=100, lr=0.1):
    nn = network
    optimizers = {
        'SGD': SGD(lr),
        'Momentum': Momentum(lr),
        'Nesterov': Nesterov(lr),
        'AdaGrad': AdaGrad(lr),
        'RMSProp': RMSProp(0.02),  # lr == 0.1 may make loss += ln(eps), eps == 1e-15
        'Adam': Adam(0.005)
    }
    opt = optimizers['Adam']

    for i in range(iter_times):
        if i % max(x_train.shape[0] // batch_size, 1) == 0:
            print('{:.1%}'.format(i / iter_times))
        batch_mask = np.random.choice(x_train.shape[0], batch_size)
        x_batch, y_batch = x_train[batch_mask], y_train[batch_mask]
        grads = nn.grad(x_batch, y_batch)
        opt.update(nn.params, grads)

    print('Train acc: {:.4}  Test acc: {:.4}'.format(nn.accuracy(x_train, y_train), nn.accuracy(x_test, y_test)))


def gradient_check(network, x_batch, y_batch):
    nn = network

    grad_n = nn.numerical_gradient(x_batch, y_batch)
    grad_bp = nn.grad(x_batch, y_batch)
    for k in nn.params:
        diff = np.mean(np.abs(grad_bp[k] - grad_n[k]))
        print('{}: {}'.format(k, diff))


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = load_mnist(normalize=True, one_hot_label=True)
    xavier_init = np.sqrt(1 / x_train.shape[1])
    he_init = np.sqrt(2 / x_train.shape[1])

    network = TwoLayer(x_train.shape[1], 10, y_train.shape[1], he_init)
    gradient_check(network, x_train[:3], y_train[:3])
    train(network, x_train, y_train, x_test, y_test)

    gradient_check(
        SimpleCNN(input_dim=(1, 10, 10)),
        np.random.rand(100).reshape((1, 1, 10, 10)),
        np.array([1]).reshape((1, 1))
    )
    (x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
    network = SimpleCNN(input_dim=(1, 28, 28))
    train(network, x_train, y_train, x_test, y_test, iter_times=10000)
