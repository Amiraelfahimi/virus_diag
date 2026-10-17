"""Optimiseurs utilisés pour entraîner les modèles."""
from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def update(self, params, grads):
        """Met à jour en place le dict `params` à partir du dict `grads`."""


class GradientDescent(Optimizer):
    def __init__(self, lr=0.1):
        self.lr = lr

    def update(self, params, grads):
        for k in params:
            params[k] -= self.lr * grads[k]


class Adam(Optimizer):
    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.beta1, self.beta2, self.eps = lr, beta1, beta2, eps
        self.m, self.v, self.t = {}, {}, 0

    def update(self, params, grads):
        self.t += 1
        for k in params:
            m = self.m.get(k, np.zeros_like(params[k]))
            v = self.v.get(k, np.zeros_like(params[k]))
            m = self.beta1 * m + (1 - self.beta1) * grads[k]
            v = self.beta2 * v + (1 - self.beta2) * grads[k] ** 2
            self.m[k], self.v[k] = m, v
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
