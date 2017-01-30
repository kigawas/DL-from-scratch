# coding: utf-8
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
    >>> (sigmoid(-100) - 0.0) < 0.00001
    True
    >>> (sigmoid(100) - 1.0) < 0.00001
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


if __name__ == '__main__':
    import doctest
    doctest.testmod()
