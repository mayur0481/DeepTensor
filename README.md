# DeepTensor
Implementing Neural Networks with Keras like API
This is a self project with the goal to be able to replicate the Keras Sequential API for building Artificial Neural Networks.
Keras allows us to simply stack layers on top of each other, provide activations, optimizers, loss functions and construct neural networks.
And then all the magic happens at the backend.
By breaking computation into a graph like structure, computing gradients becomes relatively easy.
Here I attempt at builiding a similar API , with all the tensor operations handeled using Numpy.
