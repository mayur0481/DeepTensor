#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np


# In[7]:


class _DenseLayerGradContainer:
    def __init__(self, layer):
        self.layer = layer
        self.a_grad = None
        self.z_grad = None
        self.w_grad = None
        self.b_grad = None
    
class _DenseLayerComputeContainer:
    def __init__(self, layer):
        self.layer = layer
        self.z = None
        self.output = None
        self.input = None


class _DenseLayerGraph:
    def __init__(self, layer):
        self.next_layer = None
        self.back_layer = None
        self.layer = layer

class _ActivationContainer:
    def __init__(self, layer, activation):
        self.layer = layer
        self.activation = activation
        self.name = self.activation.name
    
    def calc(self, x):
        return self.activation.calc(x)
    
    def grad_z(self, x, y):
        return self.activation.grad_z(x, y)

class _DenseLayerWeights:
    def __init__(self, layer, units):
        self.layer = layer
        self.units = units
        self.W = None
        self.b = None

class _LayerStatus:
    def __init__(self, layer, first_layer = False, final_layer = False):
        self.layer = layer
        self.first_layer = first_layer
        self.final_layer = final_layer


# In[8]:

# Activation Function Class

class Activation:
    def __init__(self):
        self.name = None

class Sigmoid(Activation):
    def __init__(self):
        self.name = 'sigmoid'
    def calc(self, x):
        return 1/(1+np.exp(-x))
    def deriv(self, x):
        j = self.calc(x)
        return j * (1-j)
    def grad_z(self, a_grad, z):
        return a_grad * self.deriv(z)

class Relu(Activation):
    def __init__(self):
        self.name = 'relu'
    
    def calc(self, x):
        y = x.copy()
        y[y<0] = 0
        return y
    
    def deriv(self, x):
        y = x.copy()
        y[y<0] = 0
        y[y>0] = 1
        return y
    
    def grad_z(self, a_grad, z):
        return a_grad * self.deriv(z)

class Identity(Activation):
    def __init__(self):
        self.name = 'identity'
    def calc(self, x):
        return x
    
    def deriv(self, x):
        return np.ones(x.size)
    
    def grad_z(self, a_grad, z):
        return a_grad * self.deriv(z)

class Softmax(Activation):
    def __init__(self):
        self.name = 'softmax'
    
    def calc(self, x):
        temp = x - np.max(x)
        temp = np.exp(temp)
        temp = temp/temp.sum()
        return temp
    
    def deriv(self, x):
        return x*np.eye(x.size) - np.outer(x, x)

    
    def grad_z(self, a_grad, z):
        return a_grad@self.deriv(z)


# In[35]:

# Fully Connected Layer Class

class DenseLayer:
    def __init__(self, units, activation = Identity()):
        self.weight_container = _DenseLayerWeights(self, units)
        self.activation = _ActivationContainer(self, activation)
        self.computes = _DenseLayerComputeContainer(self)
        self.layer_status = _LayerStatus(self)
        self.grad_container = _DenseLayerGradContainer(self)
        self.graph = _DenseLayerGraph(self)
        
    def add(self, layer):
        layer.build(self.weight_container.units)
        layer.graph.back_layer = self
        self.graph.next_layer = layer

    def forward(self):
        self.computes.z = self.computes.input@self.weight_container.W + self.weight_container.b
        self.computes.output = self.activation.calc(self.computes.z)
        self.graph.next_layer.computes.input = self.computes.output
        
    def build(self, input_size):
        self.weight_container.W = np.random.normal(0, 0.001, size = (input_size, self.weight_container.units)) 
        self.weight_container.b = np.zeros(self.weight_container.units)
    
    def backpass(self):
        self.grad_container.z_grad = self.activation.grad_z(self.grad_container.a_grad, self.computes.z)
        self.grad_container.b_grad = self.grad_container.z_grad
        self.grad_container.w_grad = np.outer(self.computes.input, self.grad_container.z_grad)
        self.graph.back_layer.grad_container.a_grad = np.dot(self.weight_container.W, self.grad_container.z_grad)
    
    def w(self):
        return self.weight_container.W.copy()
    
    def b(self):
        return self.weight_container.b.copy()
    
    def w_grad(self):
        if self.grad_container.w_grad is None:
            print('Backpropogation not initiated. (Gradient has not been computed yet)')
            return None
        return self.grad_container.w_grad.copy()
    
    def b_grad(self):
        if self.grad_container.b_grad is None:
            print('Backpropogation not initiated. (Gradient has not been computed yet)')
            return None
        return self.grad_container.b_grad.copy()
    
    def z_grad(self):
        if self.grad_container.z_grad is None:
            print('Backpropogation not initiated. (Gradient has not been computed yet)')
            return None
        return self.grad_container.z_grad.copy()
    
    def a_grad(self):
        return self.grad_container.a_grad.copy()
    
    
    def get_connections(self):
        return {'prev':self.graph.back_layer,
                'next':self.graph.next_layer}
    
    def get_computes(self):
        return {'z':self.computes.z.copy(),
                'a':self.computes.output.copy(),
                'x':self.computes.input.copy()}
    
    def get_output(self):
        return self.computes.output.copy()
    
    def get_input(self):
        return self.computes.input.copy()
    
    def get_z(self):
        return self.computes.z.copy()
    
    def __str__(self):
        return f'DenseLayer: {self.weight_container.units} units, {self.activation.name} activation'
    
    
class InputLayer:
    def __init__(self, input_size):
        self.weight_container = _DenseLayerWeights(self,input_size)
        self.computes = _DenseLayerComputeContainer(self)
        self.layer_status = _LayerStatus(self, first_layer = True)
        self.graph = _DenseLayerGraph(self)
        self.grad_container = _DenseLayerGradContainer(self)
        
    def forward(self):
        self.computes.output = self.computes.input
        self.graph.next_layer.computes.input = self.computes.output
    
    def add(self, layer):
        layer.build(self.weight_container.units)
        layer.graph.back_layer = self
        self.graph.next_layer = layer
    
    def set_input(self, x):
        assert x.size == self.weight_container.units
        self.computes.input = x
        
    def get_graph(self):
        return {'prev':self.graph.back_layer,
                'next':self.graph.next_layer}
    
    def get_output(self):
        return self.computes.output.copy()
    
    def get_input(self):
        return self.computes.input.copy()
    


# In[71]:


class _OutputContainer:
    def __init__(self):
        self.computes = _DenseLayerComputeContainer(self)
        
class _DataContainer:
    def __init__(self):
        self.x_train = None
        self.y_train = None
    
    def set_data(self, x_train, y_train):
        self.y_train = y_train.copy()[0].values
        self.x_train = x_train.copy().values
        
        
class _ModelIO:
    def __init__(self, model):
        self.model = model
        self.input_layer = None
        self.final_layer = None


class _LayerContainer:
    def __init__(self, model, layers):
        self.model = model
        self.model_length = len(layers) + 1
        self.layers = layers


class LearningRateScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer


class InverseDecay(LearningRateScheduler):
    def __init__(self, optimizer, factor = 0.1, step = 10):
        super().__init__(optimizer)
        self.step = step
        self.epoch = 0
        self.factor = factor
        self.name = 'InverseDecay'

    def update_lr(self):
        self.epoch += 1
        if self.epoch % self.step == 0:
            self.optimizer.learning_rate /= self.factor


class ExponentialDecay(LearningRateScheduler):
    def __init__(self, optimizer, rate = 0.5, step = 10):
        super().__init__(optimizer)
        self.step = step
        self.epoch = 0
        self.rate = rate
        self.name = 'ExponentialDecay'

    def update_lr(self):
        self.epoch += 1
        if self.epoch % self.step == 0:
            self.optimizer.learning_rate *= np.exp(-self.rate)

class NoDecay(LearningRateScheduler):
    def __init__(self, optimizer):
        super().__init__(optimizer)

    def update_lr(self):
        return



# Sequential Neural Network Class

class Model:
    def __init__(self, layers, input_size = None):
        self.layer_container = _LayerContainer(self, layers)
        self.model_io = _ModelIO(self)
        self.model_output = _OutputContainer()
        self.loss_fn = None
        self.optimizer = None
        self.data_container = _DataContainer()
        self.built = False
        if input_size is not None:
            self.build(input_size)
        self.optimizer_state = False
    
    def build(self, input_size):
        self.layer_container.layers.insert(0, InputLayer(input_size))
        self.model_io.input_layer = self.layer_container.layers[0]
        for i in range(1, self.layer_container.model_length):
            self.layer_container.layers[i-1].add(self.layer_container.layers[i])
        
        self.layer_container.layers[i].layer_status.final_layer = True
        self.model_io.final_layer = self.layer_container.layers[i]
        self.model_io.final_layer.graph.next_layer = self.model_output
        self.built = True
        
    def forward_pass(self, x):
        self.model_io.input_layer.set_input(x)
        for i in range(self.layer_container.model_length):
            self.layer_container.layers[i].forward()
        
        self.loss_fn.input = self.model_output.computes.input
        return self.model_output.computes.input.copy()
    
    def compile_model(self, loss, optimizer):
        self.loss_fn = loss
        self.loss_fn.model = self
        self.optimizer = optimizer
        self.optimizer.model = self
        if self.built:
            self.optimizer.initialize()

    
    def backprop(self, y):
        self.loss_fn.compute_loss(y)
        for i in range(self.layer_container.model_length-1, 0, -1):
            self.layer_container.layers[i].backpass()
        
    def optimize(self):
        if not self.optimizer_state:
            self.optimizer.initialize()
            self.optimizer_state = True
        
        self.optimizer.optimize()
        
    
    def fit(self, x_train, y_train, epochs = 10, verbose = 0):
        self.data_container.set_data(x_train, y_train)
        if not self.built:
            self.build(x_train.shape[1])
        for e in range(epochs):
            if verbose:
                print(e)
            self.optimizer.update_lr()
            for i in range(y_train.size):
                self.forward_pass(self.data_container.x_train[i])
                self.backprop(self.data_container.y_train[i])
                self.optimize()
    
    def predict(self, x):
        return self.forward_pass(x)
    

    def layer(self, index):
        return self.layer_container.layers[index]
        
        
            
    def summary(self):
        for i in range(1,self.layer_container.model_length):
            l = self.layer_container.layers[i]
            print(f'Layer{i}: units({l.weight_container.units}), activation = {self.layer_container.layers[i].activation.name}')

        print(f'Optimizer:{self.optimizer.name}, Loss:{self.loss_fn.name}')

            
 


# In[72]:

# Loss Functions

class LossFunction:
    def __init__(self):
        self.model = None
        self.input = None
        self.grad = None
        self.loss = None

class CategoricalCrossentropy(LossFunction):
    name = 'CategoricalCrossentropy'
    def compute_loss(self, cls):
        assert 0 <= cls < self.input.size, 'Class Label exceeds the number of units in the final layer'
        y = np.zeros(self.input.size)
        y[cls] = 1
        self.loss = -np.log(self.input[cls])
        self.grad = self.input - y
        self.model.model_io.final_layer.grad_container.a_grad = self.grad
        
class MeanSquaredError(LossFunction):
    name = 'MeanSquaredError'
    def compute_loss(self, y):
        try:
            print(self.input)
            self.loss = (y - self.input)**2
        except:
            print((self.input))
        self.grad = 2*(self.input - y)
        self.model.model_io.final_layer.grad_container.a_grad = self.grad


class BinaryCrossentropy(LossFunction):
    name = 'BinaryCrossetropy'
    def compute_loss(self, y):
        assert y in [0, 1], 'Found class label that does not belong to [0, 1]'
        phi = self.input
        self.grad = (phi - y)/(phi*(1-phi))
        self.loss = -(y*np.log(phi) + (1-y)*log(1-phi))
        self.model.model_io.final_layer.grad_container.a_grad = self.grad


class MultiClassSVM(LossFunction):
    name  = 'MultiClassSVM'
    def __init__(self, delta = 1):
        super().__init__()
        self.delta = delta
    def compute_loss(self, cls):
        assert 0 <= cls < self.input.size, 'Class Label exceed the number of units in the final layer'
        y = self.input - self.input[cls] + self.delta
        y[cls] = 0
        y[y<0] = 0
        self.loss = y.sum()
        self.grad = np.zeros(y.size)
        self.grad[y>0] = 1
        self.grad[cls] = -self.grad.sum()
        self.model.model_io.final_layer.grad_container.a_grad = self.grad


class MeanAbsoluteError(LossFunction):
    name = 'MeanAbsoluteError'
    def compute_loss(self, y):
        self.loss = np.abs(y - self.input)
        self.grad = -1
        if y - self.input < 0:
            self.grad = 1
        elif y == self.input:
            self.grad = 0
        self.model.model_io.final_layer.grad_container.a_grad = self.grad



# In[109]:


class  _VelocityContainer:
    def __init__(self, model, opt):
        self.model = model
        self.optimizer = opt
        self.W_velocity = []
        self.b_velocity = []
        self.set_velocity()
    
    def set_velocity(self):
        ls = self.model.layer_container.layers
        for i in range(1, self.model.layer_container.model_length):
            self.W_velocity.append(np.zeros(shape = (ls[i].weight_container.W.shape)))
            self.b_velocity.append(np.zeros(shape = (ls[i].weight_container.b.shape)))
    
    


# In[117]:

# Optimizer Class

class Optimizer:
    pass


class SGD(Optimizer):
    def __init__(self, learning_rate = 0.001, lr_scheduler = NoDecay(self)):
        self.model = None
        self.learning_rate = learning_rate
        self.name = 'SGD'
        self.lr_scheduler = lr_scheduler
    
    def optimize(self):
        ls = self.model.layer_container.layers
        for i in range(self.model.layer_container.model_length-1, 0, -1):
            ls[i].weight_container.W = ls[i].weight_container.W - self.learning_rate * ls[i].grad_container.w_grad
            ls[i].weight_container.b = ls[i].weight_container.b - self.learning_rate * ls[i].grad_container.b_grad 
    
    def initialize(self):
        return

    def update_lr(self):
        self.lr_scheduler.update_lr()
    

class SGDMomentum(Optimizer):
    def __init__(self, learning_rate = 0.001, mu = 0.9):
        self.model = None
        self.learning_rate = learning_rate
        self.mu = mu
        self.eta = eta
        self.velocity_container = None
        self.name = 'SGDMomentum'
    
    def initialize(self):
        self.velocity_container =_VelocityContainer(self.model, self)
    
    def optimize(self):
        ls = self.model.layer_container.layers
        for i in range(self.model.layer_container.model_length-1, 0, -1):
            self.velocity_container.W_velocity[i-1] = self.mu * self.velocity_container.W_velocity[i-1] + self.learning_rate * ls[i].grad_container.w_grad
            self.velocity_container.b_velocity[i-1] = self.mu * self.velocity_container.b_velocity[i-1] + self.learning_rate * ls[i].grad_container.b_grad
            ls[i].weight_container.W = ls[i].weight_container.W - self.velocity_container.W_velocity[i-1]
            ls[i].weight_container.b = ls[i].weight_container.b - self.velocity_container.b_velocity[i-1]


class NesterovMomentum:
    def __init__(self, learning_rate = 0.001, mu = 0.9):
        self.model = None
        self.learning_rate = learning_rate
        self.mu = mu
        self.velocity_container = None
        self.name = 'NesterovMomentum'


    def initialize(self):
        self.velocity_container = _VelocityContainer(self.model, self)


    def optimize(self):
        ls = self.model.layer_container.layers
        for i in range(self.model.layer_container.model_length-1, 0, -1):
            prev_vW = self.velocity_container.W[i-1].copy()
            prev_vb = self.velocity_container.b[i-1].copy()
            self.velocity_container.W_velocity[i-1] = self.mu * self.velocity_container.W_velocity[i-1] - self.learning_rate * ls[i].grad_container.w_grad
            self.velocity_container.b_velocity[i-1] = self.mu * self.velocity_container.b_velocity[i-1] - self.learning_rate * ls[i].grad_container.b_grad
            ls[i].weight_container.W = ls[i].weight_container.W - self.mu * prev_vW + (1 + self.mu) * (self.velocity_container.W_velocity[i-1])
            ls[i].weight_container.b = ls[i].weight_container.b - self.mu * prev_b + (1 + self.b) * (self.velocity_container.b_velocity[i-1])







