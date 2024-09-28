import torch
from linear import Linear

class SGD:
    """
    Stochastic Gradient Descent
    """

    def __init__(self, layers, learning_rate):
        """
        Constructor
        :param layers: Tuple of shape (n_feature, n_hidden1, n_hidden2, ..., n_output)
        :param learning_rate: Learning rate
        :return: None
        """
        self.layers = layers
        self.learning_rate = learning_rate

    def step(self):
        """
        Update parameters using gradient descent
        :return: None
        """
        for layer in self.layers:
            if isinstance(layer, Linear):
                ### START CODE HERE ### (≈ 2 lines of code)
                layer.W -= self.learning_rate * layer.dW  # Update weights
                layer.b -= self.learning_rate * layer.db  # Update biases
                ### END CODE HERE ###


def debug():
    from model import ShallowNeuralNet

    N = 5   # number of samples
    D_i, D_h, D_o = 4, 3, 2
    nn = ShallowNeuralNet((D_i, D_h, D_o))
    sgd = SGD(nn.layers, learning_rate=2)

    # Input matrix
    X = torch.randn(D_i, N)

    #######################
    # Forward propagation #
    #######################
    nn(X)

    # Weight during propagation
    print('Forward propagation')
    for i, layer in enumerate(nn.layers):
        print(f'#{i+1} {layer.__class__.__name__}')
        if isinstance(layer, Linear):
            print(f'\tW = {layer.W}')
            print(f'\tdW = {layer.dW}')
            print(f'\tb = {layer.b}')
            print(f'\tdb = {layer.db}')
    print('---')

    ####################
    # Back propagation #
    ####################
    dF = torch.randn((D_o, N))
    nn.backward(dF)

    # Weight after back propagation
    print('After Backpropagation')
    for i, layer in enumerate(nn.layers):
        print(f'#{i+1} {layer.__class__.__name__}')
        if isinstance(layer, Linear):
            print(f'\tW = {layer.W}')
            print(f'\tdW = {layer.dW}')
            print(f'\tb = {layer.b}')
            print(f'\tdb = {layer.db}')
    print('---')

    ###################
    # Gradient update #
    ###################
    sgd.step()

    # Weight after gradient update
    print('After gradient update')
    for i, layer in enumerate(nn.layers):
        print(f'#{i+1} {layer.__class__.__name__}')
        if isinstance(layer, Linear):
            print(f'\tW = {layer.W}')
            print(f'\tb = {layer.b}')


if __name__ == '__main__':
    debug()
