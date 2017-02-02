import numpy as np


class Optimizer(object):

    def __init__(self, lr=0.1):
        self.lr = lr

    def update(self, params, grads):
        raise NotImplementedError


class SGD(Optimizer):
    '''
    Traditional mini-Batch stochastic gradient descent
    '''

    def update(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]


class Momentum(Optimizer):
    '''
    Inspired by kinematics & dynamics, actually the "momentum" should be "friction coefficient"
    '''

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr, self.momentum = lr, momentum
        self.v = {}

    def update(self, params, grads):
        if not self.v:
            for k, v in params.items():
                self.v[k] = np.zeros_like(v)

        for k in params:
            self.v[k] = self.momentum * self.v[k] - self.lr * grads[k]
            params[k] += self.v[k]


class Nesterov(Momentum):
    '''
    A variant of Momentum
    '''

    def update(self, params, grads):
        if not self.v:
            for k, v in params.items():
                self.v[k] = np.zeros_like(v)

        for k in params:
            last_v = self.v[k].copy()
            self.v[k] = self.momentum * self.v[k] - self.lr * grads[k]
            params[k] += -self.momentum * last_v + (1 + self.momentum) * self.v[k]


class AdaGrad(Optimizer):
    '''
    Divide learning rate by sqrt(square sum of gradient) when training
    '''

    def __init__(self, lr=0.01, eps=1e-8):
        self.lr, self.eps = lr, eps
        self.h = {}

    def update(self, params, grads):
        if not self.h:
            for k, v in params.items():
                self.h[k] = np.zeros_like(v)

        for k in params:
            self.h[k] += grads[k] * grads[k]
            params[k] -= self.lr * grads[k] / (np.sqrt(self.h[k]) + self.eps)


class RMSProp(Optimizer):
    '''
    A variant of AdaGrad using EMA
    Reference: http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
    '''

    def __init__(self, lr=0.01, decay=0.9, eps=1e-8):
        self.lr, self.decay, self.eps = lr, decay, eps
        self.h = {}

    def update(self, params, grads):
        if not self.h:
            for k, v in params.items():
                self.h[k] = np.zeros_like(v)

        for k in params:
            self.h[k] = self.decay * self.h[k] + (1 - self.decay) * grads[k] * grads[k]
            params[k] -= self.lr * grads[k] / (np.sqrt(self.h[k]) + self.eps)


class Adam(Optimizer):
    '''
    A naive intuition: AdaGrad + Momentum
    Reference: https://arxiv.org/pdf/1412.6980v9.pdf
    '''

    def __init__(self, lr=0.0001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v = {}, {}
        self.t = 0

    def update(self, params, grads):
        if not self.m:
            for k, v in params.items():
                self.m[k] = np.zeros_like(v)
                self.v[k] = np.zeros_like(v)

        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        for k in params:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k] * grads[k]
            params[k] -= lr_t * self.m[k] / (np.sqrt(self.v[k]) + self.eps)


if __name__ == '__main__':
    pass
