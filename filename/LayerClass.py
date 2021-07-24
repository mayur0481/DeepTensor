import numpy as np


# Parent Layer Class

class Layer:
    def __init__(self):
        self.next = None
        self.prev = None

    def add(self, layer):
        return

    def __call__(self, layer):
        self.add(layer)
        return 



# Fully Connected Layer

class Dense(Layer):
    def __init__(self, units, activation):
        super().__init__()
        self.units = units
        self.activation = activation
        self.W = None
        self.b = None
        self.W_grad = None
        self.b_grad = None
        self.built = False
        self.z = None
        self.a = None
        self.a_grad = None


    
    def build(self, prev_units):
        self.W = np.random.normal(0, 1, size = (prev_units, self.units))
        self.b = np.random.normal(0, 1, size = (1, self.units))
        self.built = True
        return

    
    def add(self, layer):
        if not self.built:
            self.build(layer.units)
        self.prev = layer
        layer.next = self
        return

    
    def forward(self):
        self.z = self.prev.a @ self.W + self.b
        self.a = self.activation(self.z)
        return

    
    def compute_grad(self):
        z_grad = self.a_grad * self.activation.grad(self.z)
        self.W_grad = self.prev.a.T @ z_grad
        self.b_grad = z_grad.sum(axis = 0)
        self.prev.a_grad = z_grad @ (self.W).T
        return

    
    def compute(self, x):
        if not self.built:
            print('Layer is not built. Forward pass Failed')
            exit(1)

        z = x @ self.W + self.b
        return self.activation(z)



# Input Layer

class Input:
    def __init__(self, input_size):
        self.name = 'Input'
        self.a = None
        self.units = input_size
        self.a_grad = None

    
    def __call__(self, layer):
        print('Input Layer cannot be attached to another layer')
        exit(1)


    def set_a(self, x):
        self.a = x
        return




















