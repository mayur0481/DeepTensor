import numpy as np
class Crossentropy:
    def __init__(self):
        self.name = 'Crossentropy'
        self.prev = None
        self.loss = None
        self.grad = None

    def compute_loss(self, y):
        w = self.prev.a
        self.loss = -np.log(w[np.arange(w.shape[0]), y]).sum()/(w.shape[0])
        return

    def compute_grad(self,y):
        w = self.prev.a
        grad = np.zeros(shape = w.shape)
        grad[np.arange(w.shape[0]), y] = (-1/w.shape[0]) * (1/w[np.arange(w.shape[0]), y])
        self.grad = grad
        self.prev.a_grad = grad
        return



class MeanSquaredError:
    def __init__(self):
        self.name = 'MeanSquaredError'
        self.prev = None
        self.loss = None
        self.grad = None

    def compute_loss(self, y):
        w = self.prev.a
        self.loss = ((w-y)**2).sum()/w.shape[0]
        return

    def compute_grad(self, y):
        self.grad = 2/(y.shape[0]) * (self.prev.a - y)
        self.prev.a_grad = self.grad
        return



class MeanAbsoluteError:
    def __init__(self):
        self.name = 'MeanAbsoluteError'
        self.prev = None
        self.loss = None
        self.grad = None

    def compute_loss(self, y):
        self.loss = np.abs(self.prev.a - y).sum()/y.shape[0]
        return

    def compute_grad(self, y):
        t = y - self.prev.a
        pos = t[t>0]
        neg = t[t<0]
        t[pos] = -1
        t[neg] = 1
        return t


class BinaryCrossEntropy:
    def __init__(self):
        self.name = 'BinaryCrossEntropy'
        self.prev = None
        self.loss = None
        self.grad = None
    
    def compute_loss(self, y):
        w = self.prev.a
        mask1 = (y==1)
        mask0 = (y==0)
        self.loss = -(np.log(w[mask1]).sum() + np.log(1 - w[mask0]).sum())

    def compute_grad(self, y):
        w = self.prev.a.copy()
        mask1 = (y==1)
        mask0 = (y==0)
        w[mask1] = -1/(w[mask1])
        w[mask0] = 1/(1 - w[mask0])
        self.grad = w
        self.prev.a_grad = w
        return


 





