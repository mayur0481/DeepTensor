import numpy as np

class Identity:
    def __init__(self):
        self.name = 'Identity'

    def __call__(self, x):
        return x

    def grad(self, x):
        return np.ones(shape = x.shape)



class Relu:
    def __init__(self):
        self.name = 'Relu'


    def __call__(self, x):
        t = x.copy()
        t[t<0] = 0
        return t

    def grad(self, x):
        t = x.copy()
        t[t>0] = 1
        t[t<0] = 0
        return t



class Sigmoid:
    def __init__(self):
        self.name = 'Sigmoid'

    def __call__(self, x):
        return 1/(1+np.exp(-x))

    def grad(self, x):
        t = self(x)
        return t * (1 - t)



class Softmax:
    def __init__(self):
        self.name = 'Softmax'

    def __call__(self, x):
        t = x - np.max(x, axis = 1, keepdims = True)
        t = np.exp(t)
        return t/(t.sum(axis = 1, keepdims = True))

    def grad(self, x):
        t = self(x)
        return t*(1-t)



class LeakyRelu:
    def __init__(self, alpha = 0.1):
        self.name = 'LeakyRelu'
        self.alpha = alpha

    def __call__(self, x):
        t = x.copy()
        t[t<0] *= self.alpha
        return t

    def grad(self, x):
        t = x.copy()
        t[t>0] = 1
        t[t<0] = self.alpha
        return t



class ELU:
    def __init__(self, alpha = 0.1):
        self.name = 'ELU'
        self.alpha = alpha

    
    def __call__(self, x):
        t = x.copy()
        t[t<0] = self.alpha * (np.exp(t[t<0]) - 1)
        return t

    def grad(self, x):
        t = x.copy()
        t[t<0] = self.alpha * np.exp(t[t<0])
        return t



class SELU:
    def __init__(self):
        self.name = 'SELU'
        self.alpha = 1.67326324
        self.scale = 1.05070098

    def __call__(self, x):
        t = x.copy()
        t[t>0] *= self.scale
        t[t<0] = self.scale * self.alpha * (np.exp(t[t<0]) - 1)
        return t

    def grad(self, x):
        t = x.copy()
        t[t>0] = self.scale
        t[t<0] = self.scale * self.alpha * (np.exp(t[t<0]))
        return t


class SiLU:
    def __init__(self):
        self.name = 'SiLU'

    def __call__(self, x):
        return x * (1/(1+np.exp(-x)))

    def grad(self, x):
        sig = 1/(1+np.exp(-x))
        return sig * (1 + x*(1-sig))













