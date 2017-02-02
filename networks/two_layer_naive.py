import os
os.sys.path.extend([os.pardir, os.curdir])

import numpy as np

from common.function import cross_entropy, sigmoid, softmax
from common.gradient import numerical_grad


class TwoLayer(object):
    '''
    >>> n = TwoLayer(2, 10, 3)
    >>> output = n.predict(np.array([[1, 2]]))
    >>> abs(np.sum(output) - 1.0) < 0.0001
    True
    >>> output = n.predict(np.array([[1, 2], [3, 4]]))
    >>> np.all(abs(np.sum(output, axis=1) - 1.0) < 0.0001)
    True
    '''

    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['w1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['w2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)
        return y

    def accuracy(self, x, t):
        predicted_label = self.predict(x).argmax(axis=1)
        test_label = t.argmax(axis=1)
        return float(np.sum(predicted_label == test_label)) / x.shape[0]

    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy(y, t)

    def numerical_gradient(self, x, t):
        lost_func = lambda w: self.loss(x, t)
        grads = {}
        for k in self.params:
            grads[k] = numerical_grad(lost_func, self.params[k])
        return grads

    def grad(self, x, t):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        # forward
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / x.shape[0]  # softmax with entropy loss's gradient, dL/dy
        grads['w2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, w2.T)
        dz1 = (1.0 - sigmoid(a1)) * sigmoid(a1) * da1  # sigmoid's gradient
        grads['w1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


if __name__ == '__main__':
    import doctest
    doctest.testmod()
