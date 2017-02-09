import os
os.sys.path.extend([os.pardir, os.curdir])

import numpy as np

from common.function import softmax, cross_entropy
from common.util import im2col, col2im


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
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # one-hot
            return (self.y - self.t) / self.t.shape[0]
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
            return dx


class Dropout(Layer):
    '''
    >>> np.random.seed(5)
    >>> d = Dropout()
    >>> x = np.array([[0.7, 0.2, 0.4], [-0.1, -0.2, -0.3]])
    >>> out = d.forward(x)
    >>> out
    array([[ 0. ,  0.2,  0. ],
           [-0.1, -0.2, -0.3]])
    >>> d.backward(out)
    array([[ 0. ,  0.2,  0. ],
           [-0.1, -0.2, -0.3]])
    '''

    def __init__(self, ratio=0.3):
        self.ratio = ratio

    def forward(self, x, training=True):
        if training:
            self.mask = np.random.rand(*x.shape) > self.ratio
            return x * self.mask
        else:
            return x * (1.0 - self.ratio)

    def backward(self, dout):
        return dout * self.mask


class Convolution(Layer):

    def __init__(self, w, b, stride=1, pad=0):
        self.w, self.b, self.stride, self.pad = w, b, stride, pad

    def forward(self, x):
        self.x = x

        fn, c, fh, fw = self.w.shape  # filter
        n, c, h, w = self.x.shape

        out_h = 1 + (h + 2 * self.pad - fh) // self.stride
        out_w = 1 + (h + 2 * self.pad - fw) // self.stride

        self.col = im2col(self.x, fh, fw, self.stride, self.pad)
        self.col_w = self.w.reshape((fn, -1)).T

        out = np.dot(self.col, self.col_w) + self.b
        out = out.reshape(n, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        fn, c, fh, fw = self.w.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, fn)

        self.db = np.sum(dout, axis=0)
        self.dw = np.dot(self.col.T, dout).transpose(1, 0).reshape(fn, c, fh, fw)

        dcol = np.dot(dout, self.col_w.T)
        dx = col2im(dcol, self.x.shape, fh, fw, self.stride, self.pad)

        return dx


class Pooling(Layer):

    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h, self.pool_w, self.stride, self.pad = pool_h, pool_w, stride, pad

    def forward(self, x):
        n, c, h, w = x.shape

        out_h = 1 + (h - self.pool_h) // self.stride
        out_w = 1 + (h - self.pool_w) // self.stride

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape((-1, self.pool_h * self.pool_w))

        self.x = x
        self.arg_max = np.argmax(col, axis=1)

        out = np.max(col, axis=1)
        out = out.reshape(n, out_h, out_w, c).transpose(0, 3, 1, 2)
        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        # just like Relu, firstly dx = dout and secondly set all non-max elements to 0
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx


if __name__ == '__main__':
    import doctest
    doctest.testmod()
