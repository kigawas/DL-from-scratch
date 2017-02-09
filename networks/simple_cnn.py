import os
os.sys.path.extend([os.pardir, os.curdir])

from collections import OrderedDict

import numpy as np

from common.gradient import numerical_grad
from common.layer import Affine, SoftmaxWithLoss, Relu, Convolution, Pooling


class SimpleCNN(object):
    '''
    Structure: conv->relu->pooling(max)->affine->relu->affine->softmax_with_loss
    '''

    def __init__(self,
                 input_dim=(1, 10, 10),
                 conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size=10,
                 output_size=10,
                 weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        pad = conv_param['pad']
        stride = conv_param['stride']
        input_size = input_dim[1]

        conv_output_size = 1 + (input_size - filter_size + 2 * pad) // stride
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))

        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['w2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['w3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['w1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(2, 2, stride=2)

        self.layers['Affine1'] = Affine(self.params['w2'], self.params['b2'])
        self.layers['Relu2'] = Relu()

        self.layers['Affine2'] = Affine(self.params['w3'], self.params['b3'])
        self.output_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.output_layer.forward(y, t)

    def accuracy(self, x, t):
        predicted_label = self.predict(x).argmax(axis=1)
        if t.ndim != 1:
            test_label = t.argmax(axis=1)
        else:
            test_label = t
        return float(np.sum(predicted_label == test_label)) / x.shape[0]

    def numerical_gradient(self, x, t):
        lost_func = lambda w: self.loss(x, t)
        grads = {}
        for k in self.params:
            grads[k] = numerical_grad(lost_func, self.params[k])
        return grads

    def grad(self, x, t):
        self.loss(x, t)
        dout = self.output_layer.backward()
        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        grads = {}
        grads['w1'] = self.layers['Conv1'].dw
        grads['b1'] = self.layers['Conv1'].db
        grads['w2'] = self.layers['Affine1'].dw
        grads['b2'] = self.layers['Affine1'].db
        grads['w3'] = self.layers['Affine2'].dw
        grads['b3'] = self.layers['Affine2'].db
        return grads
