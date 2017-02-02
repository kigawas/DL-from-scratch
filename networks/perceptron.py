import os
os.sys.path.append(os.path.abspath(os.curdir))

import numpy as np


class Perceptron(object):
    '''
    Perceptron implemented with the classic PLA
    >>> from common.function import step
    >>> p = Perceptron(f=step)
    >>> X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    >>> and_y = np.array([0, 0, 0, 1])
    >>> or_y = np.array([0, 1, 1, 1])
    >>> p.fit(X, and_y)
    >>> np.all(p.predict(X) == and_y)
    True
    >>> p.errors
    [1, 3, 1, 0]
    >>> p.fit(X, or_y)
    >>> np.all(p.predict(X) == or_y)
    True
    >>> p.errors
    [3, 1, 1, 1, 0]
    '''

    def __init__(self, f=np.sign, lr=1, iter_times=100):
        self.f, self.lr, self.iter_times = f, lr, iter_times

    def add_one(self, X):
        '''
        Add a column with all ones
        >>> p = Perceptron()
        >>> p.add_one(np.array([[0, 0], [0, 1]]))
        array([[ 1.,  0.,  0.],
               [ 1.,  0.,  1.]])
        '''
        return np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

    def predict(self, X):
        '''
        >>> p = Perceptron()
        >>> p.predict(np.array([[1, 2], [3, 4]]))
        Traceback (most recent call last):
            ...
        AttributeError: 'Perceptron' object has no attribute 'w'
        '''
        X = self.add_one(X)
        return self.f(X.dot(self.w))

    def fit(self, X, y):
        self.w = np.zeros(1 + X.shape[1])
        self.errors = []

        for _ in range(self.iter_times):
            update = self.lr * (y - self.predict(X))
            error_number = np.sum(update != 0)
            self.errors.append(error_number)
            if error_number == 0:
                break
            # use reshape to convert to an n*1 column vector
            self.w += np.sum(self.add_one(X) * update.reshape((-1, 1)), axis=0)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
