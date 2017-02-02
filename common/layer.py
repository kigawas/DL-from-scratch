import os
os.sys.path.extend([os.pardir, os.curdir])

import numpy as np

from common.function import softmax, cross_entropy


class Layer(object):

    def __init__(self):
        pass

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


class Relu(Layer):
    '''
    >>> r = Relu()
    >>> x = np.array( [[1.0, -0.5], [-2.0, 3.0]] )
    >>> out = r.forward(x)
    >>> np.all(out == r.backward(out))
    True
    '''

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Sigmoid(Layer):
    '''
    >>> s = Sigmoid()
    >>> out = s.forward(0)
    >>> s.backward(1)
    0.25
    '''

    def forward(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out

    def backward(self, dout):
        return dout * (1.0 - self.out) * self.out


class Affine(Layer):
    '''
    Batch affine layer
    >>> w = np.array([[1, 2, 3], [-1, -2, -3]])
    >>> x = np.array([[1, 2], [2, 1]])
    >>> b = np.array([1, 2, 3])
    >>> y = x.dot(w) + b
    >>> y
    array([[0, 0, 0],
           [2, 4, 6]])
    >>> a = Affine(w, b)
    >>> np.all(a.forward(x) == y)
    True
    >>> a.backward(np.array([[-1, 0, 1], [-2, 1, 3]]))
    array([[ 2, -2],
           [ 9, -9]])
    >>> y = a.forward(np.array([[1, 2]]))
    >>> a.backward(np.array([[-1, 0, 1]]))
    array([[ 2, -2]])
    '''

    def __init__(self, w, b):
        self.w, self.b = w, b

    def forward(self, x):
        self.x_shape = x.shape
        self.x = x.reshape((x.shape[0], -1))
        return np.dot(self.x, self.w) + self.b

    def backward(self, dout):
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return np.dot(dout, self.w.T).reshape(self.x_shape)


class SoftmaxWithLoss(Layer):

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy(self.y, self.t)
        return self.loss

    def backward(self):
        return (self.y - self.t) / self.t.shape[0]


if __name__ == '__main__':
    import doctest
    doctest.testmod()
