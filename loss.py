import torch

class CELoss:
    """
    Multi-class Cross Entropy Loss
    """

    def __init__(self):
        self.Y = None  # Ground truth
        self.F = None  # Predictions

    def __call__(self, Y, F, training=True):
        """
        Forward propagation of CE Loss
        :param Y: Ground truth, shape (n_output, batch size)
        :param F: Model prediction, shape (n_output, batch size)
        :return: Cross-entropy loss (scalar)
        """

        ### START CODE HERE ### (≈ 1-2 lines of code)
        # Apply softmax to F to get probabilities
        F_softmax = torch.softmax(F, dim=0)  # Softmax over classes
        # Compute cross-entropy loss
        loss = -torch.sum(Y * torch.log(F_softmax + 1e-9)) / Y.shape[1]
        ### END CODE HERE ###

        assert Y.shape == F.shape, "Shapes of Y and F must match"
        if training:
            self.Y = Y.clone()
            self.F = F.clone()

        return loss

    def backward(self):
        """
        Backward propagation for CE Loss
        :return: Gradient w.r.t. F
        """
        
        ### START CODE HERE ### (≈ 1 lines of code)
        F_softmax = torch.softmax(self.F, dim=0)
        dF = (F_softmax - self.Y) / self.Y.shape[1]  # Gradient of cross-entropy loss
        ### END CODE HERE ###

        return dF




def debug():
    (D_i, N) = 3, 4

    Y = torch.tensor([
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
    ], dtype=torch.float32)
    print('Y =', Y)
    
    F = torch.tensor([
        [-2, 3, 8, -7],
        [10, 2, 8, 4],
        [5, 7, 1, 3],
    ], dtype=torch.float32)
    print('F =', F)

    ce_loss = CELoss()
    loss = ce_loss(Y, F)
    print('Loss =', loss)

    dF = ce_loss.backward()
    print('dF =', dF)


if __name__ == '__main__':
    debug()
