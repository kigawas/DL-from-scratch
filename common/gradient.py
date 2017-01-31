import numpy as np


def numerical_diff(f, x):
    '''
    >>> abs(numerical_diff(lambda x: x * x, 1) - 2) < 0.0001
    True
    '''
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_grad(f, x):
    '''
    >>> numerical_grad(lambda x: np.sum(x**2), np.array([3.0, 4.0]))
    array([ 6.,  8.])
    >>> numerical_grad(lambda x: np.sum(x**2), np.array([[3.0, 4.0], [-3.0, -4.0]]))
    array([[ 6.,  8.],
           [-6., -8.]])
    '''
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        old_x = x[i]
        x[i] = old_x + h
        fx_front = f(x)
        x[i] = old_x - h
        fx_back = f(x)
        grad[i] = (fx_front - fx_back) / (2 * h)
        x[i] = old_x
        it.iternext()
    return grad


def gradient_descent(f, x, lr=0.1, iter_times=100):
    '''
    >>> minimum = gradient_descent(lambda x: np.sum(x**2), np.array([3.0, 4.0]))
    >>> np.all((minimum - 0) < 0.0001)
    True
    >>> minimum = gradient_descent(lambda x: np.sum(x**2), np.array([3.0, 4.0]), lr=10)
    >>> np.all((minimum - 0) < 0.0001)
    False
    '''
    for i in range(iter_times):
        grad = numerical_grad(f, x)
        x -= lr * grad
    return x


if __name__ == '__main__':
    import doctest
    doctest.testmod()
