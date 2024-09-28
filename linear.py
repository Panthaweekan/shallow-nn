import torch
import numpy as np

class Linear:
    """
    Linear function
    """

    def __init__(self, D_i, D_o):
        """
        :param D_i: Dimension of input
        :param D_o: Dimension of output (Number of neurons)
        """

        # Parameters
        # W is weight matrix: array of shape (D_o, D_i)
        # b is bias vector: array of shape (D_o, 1)
        
        ### START CODE HERE ### (≈ 1 line of code)
        # He initialization (gain=2, mode=fan_in)
        self.W = torch.randn((D_o, D_i)) * np.sqrt(2. / D_i)
        ### END CODE HERE ###

        self.b = torch.zeros(D_o, 1)
        
        # Forward propagation cache
        self.H = None

        # Computed gradients
        self.dW = None
        self.db = None

    def __call__(self, H, training=True):
        """
        Forward propagation for linear function
        :param H: Input data of shape (D_i, N) where N is batch size
        :return: Output of linear unit
        """
        F = self.W @ H + self.b  # Matrix multiplication and bias addition

        assert F.shape == (self.W.shape[0], H.shape[1]), "Output shape mismatch"

        if training:
            self.H = H.clone()  # Store input for backpropagation

        return F

    def backward(self, dF):
        """
        Backpropagation for linear function
        :param dF: Gradient of the loss w.r.t. the output (F), shape (D_o, N)
        :return: Gradient of the loss w.r.t. the input (dH), shape (D_i, N)
        """

        ### START CODE HERE ###
        # Gradient with respect to weights: dW = dF @ H^T
        dW = dF @ self.H.T  # Shape: (D_o, N) @ (N, D_i) -> (D_o, D_i)
        # Gradient with respect to bias: sum over batch dimension
        db = torch.sum(dF, dim=1, keepdim=True)  # Shape: (D_o, 1)
        # Gradient with respect to input: dH = W^T @ dF
        dH = self.W.T @ dF  # Shape: (D_i, D_o) @ (D_o, N) -> (D_i, N)
        ### END CODE HERE ###

        # Store computed gradients
        self.dW = dW.clone()
        self.db = db.clone()

        return dH



def debug():
    D_i, D_o, N = 3, 2, 5  # Input dimension, output dimension, batch size
    linear = Linear(D_i, D_o)

    H = torch.randn((D_i, N))  # Random input
    F = linear(H)  # Forward pass

    print('F =', F)

    dF = torch.randn_like(F)  # Random gradient from next layer (dF)
    dH = linear.backward(dF)  # Backward pass
    print('dH =', dH)
    print('dW =', linear.dW)
    print('db =', linear.db)


if __name__ == '__main__':
    debug()
