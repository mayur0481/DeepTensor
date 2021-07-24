import LayerClass

class SequentialANN:
    def __init__(self, layers, input_size):
        self.layers = layers
        self.input_size = input_size
        self.input_layer = LayerClass.Input(input_size)
        self.output_layer = self.layers[-1]
        self.build_()
        self.optimizer = None
        self.loss = None
        self.compiled = False

    def build_(self):
        for i in range(len(layers)):
            if i == 0:
                self.layers[i](self.input_layer)
                continue

            self.layers[i](self.layers[i-1])

        return

    def compile_model(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss = loss
        self.compiled = True
        return



    def forward_pass(self, x):
        self.input_layer.set_a(x)
        for l in self.layers:
            l.forward()
        return

    
    def backprop(self):
        self.loss









