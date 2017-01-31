import os
os.sys.path.extend([os.pardir, os.curdir])

from collections import OrderedDict

import numpy as np

from common.gradient import numerical_grad
from common.layers import Affine, SoftmaxWithLoss, Relu


class TwoLayer(object):
    '''
    >>> from common.functions import softmax
    >>> n = TwoLayer(2, 10, 3)
    >>> output = softmax(n.predict(np.array([[1, 2]])))
    >>> abs(np.sum(output) - 1.0) < 0.0001
    True
    >>> output = softmax(n.predict(np.array([[1, 2], [3, 4]])))
    >>> np.all(abs(np.sum(output, axis=1) - 1.0) < 0.0001)
    True
    '''

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['w1'], self.params['b1'])
        self.layers['Relu'] = Relu()
        self.layers['Affine2'] = Affine(self.params['w2'], self.params['b2'])
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
        grads['w1'] = self.layers['Affine1'].dw
        grads['b1'] = self.layers['Affine1'].db
        grads['w2'] = self.layers['Affine2'].dw
        grads['b2'] = self.layers['Affine2'].db
        return grads


if __name__ == '__main__':
    import doctest
    doctest.testmod()
