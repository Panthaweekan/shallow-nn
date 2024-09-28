import torch

class ReLU:
    """
    Rectified Linear Unit (ReLU)
    """

    def __init__(self):
        """
        Constructor
        """

        # Forward propagation cache
        self.F = None

    def __call__(self, F, training=True):
        """
        Forward propagation for ReLU activation
        :param F: pre-activation
        :return: post-activation
        """
        ### START CODE HERE ### (≈ 1 line of code)
        H = torch.relu(F)  # Apply ReLU activation
        ###  END CODE HERE  ###

        if training:
            self.F = torch.clone(F)  # Store input for backpropagation

        return H

    def backward(self, dH):
        """
        Backward propagation for ReLU activation.
        :param dH: post-activation gradient (of loss w.r.t. H)
        :return:
            dF:  pre-activation gradient (of loss w.r.t. F)
        """

        ### START CODE HERE ### (≈ 1 line of code)
        dF = dH * (self.F > 0).float()  # Derivative of ReLU: 1 if F > 0, else 0
        ### END CODE HERE ###

        assert dF.shape == self.F.shape
        return dF


def debug():
    D_i, N = 2, 3

    relu = ReLU()

    F = torch.randn((D_i, N))
    print('F =', F)
    H = relu(F)
    print('H =', H)

    dH = torch.randn((D_i, N))
    print('dH =', dH)
    dF = relu.backward(dH)
    print('dF =', dF)


if __name__ == '__main__':
    debug()
