import numpy as np


def identity(x):
    '''
    >>> identity(1)
    1
    >>> identity(np.array([[1, 2], [3, 4]]))
    array([[1, 2],
           [3, 4]])
    '''
    return x


def step(x):
    '''
    >>> step(np.array([[1, -2], [3, -4]]))
    array([[1, 0],
           [1, 0]])
    '''
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    '''
    >>> sigmoid(0)
    0.5
    >>> abs(sigmoid(-100) - 0.0) < 0.00001
    True
    >>> abs(sigmoid(100) - 1.0) < 0.00001
    True
    '''
    return 1 / (1 + np.exp(-x))


def relu(x):
    '''
    >>> relu(np.array([[1, -2], [3, -4]]))
    array([[1, 0],
           [3, 0]])
    '''
    return np.maximum(0, x)


def softmax(x):
    '''
    >>> res = softmax(np.array([0, 200, 10]))
    >>> np.sum(res)
    1.0
    >>> np.all(np.abs(res - np.array([0, 1, 0])) < 0.0001)
    True
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200], [200, 0, 10]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.,  1.])
    >>> res = softmax(np.array([[0, 200, 10], [0, 10, 200]]))
    >>> np.sum(res, axis=1)
    array([ 1.,  1.])
    '''
    assert x.ndim <= 2
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


def cross_entropy(y, t):
    '''
    Batch cross entropy
    >>> t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    >>> y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    >>> cn1 = cross_entropy(np.array(y), np.array(t))
    >>> t = [[0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    >>> y = [[0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0], [0.1, 0.05, 0.0, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.6]]
    >>> cn2 = cross_entropy(np.array(y), np.array(t))
    >>> abs(cn1 - cn2) < 0.0001
    True
    '''
    if y.ndim == 1:
        t, y = t.reshape((1, -1)), y.reshape((1, -1))

    if t.size == y.size:
        t = np.argmax(t, axis=1)

    batch = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch), t])) / batch


if __name__ == '__main__':
    import doctest
    doctest.testmod()
