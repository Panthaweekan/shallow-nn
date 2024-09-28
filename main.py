import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data import get_mnist1d_loader
from model import ShallowNeuralNet
from loss import CELoss
from optimizer import SGD

# Set global seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Acquire data loader
dl_train, dl_validate = get_mnist1d_loader(batch_size=100)

# Model
nn = ShallowNeuralNet((40, 100, 100, 10))
print(nn)

# Loss function
ce_loss = CELoss()

# Optimizer
sgd = SGD(nn.layers, learning_rate=0.005)

# Training loop
num_epochs = 50
train_losses, val_losses = [], []
for epoch in range(num_epochs):

    # TRAINING loop
    train_epoch_loss, train_acc, n_train = 0.0, 0.0, 0
    for i, (x_batch, y_batch) in enumerate(dl_train):

        n_train += len(y_batch)
        x_batch = torch.permute(x_batch, (1,0))
        y_batch_one_hot = F.one_hot( y_batch, num_classes=nn.dim_out) # one-hot ground truth
        y_batch_one_hot = torch.permute(y_batch_one_hot, (1,0))

        # Forward propagation
        logit = nn(x_batch)

        # Compute forward loss
        train_epoch_loss += ce_loss(y_batch_one_hot, logit)

        # Compute gradient of loss function
        dloss = ce_loss.backward()

        # Backpropagation
        grads = nn.backward(dloss)

        # Gradient update
        ### START CODE HERE ### (≈ 1 line of code)
        sgd.step()
        ### END CODE HERE ###

        # Compute accuracy (for debugging)
        y_pred = torch.argmax(logit, dim=0) # convert one-hot to index
        train_acc += torch.sum(y_pred == y_batch).item()


    # VALIDATION loop (don't train anything in this loop)
    val_epoch_loss, val_acc, n_val = 0.0, 0.0, 0
    for x_batch, y_batch in dl_validate:

        n_val += len(y_batch)
        x_batch = torch.permute(x_batch, (1,0))
        y_batch_one_hot = F.one_hot( y_batch, num_classes=nn.dim_out) # one-hot ground truth
        y_batch_one_hot = torch.permute(y_batch_one_hot, (1,0))

        # Forward propagation
        logit = nn(x_batch, training=False)

        # Compute forward loss
        val_epoch_loss += ce_loss(y_batch_one_hot, logit, training=False)

        # Compute accuracy (for debugging)
        y_pred = torch.argmax(logit, dim=0) # convert one-hot to index
        val_acc += torch.sum(y_pred == y_batch).item()

    print(f'Epoch {epoch}: train_loss {train_epoch_loss/n_train:.6f}, train_acc {100*train_acc/n_train:5.2f}%, val_loss {val_epoch_loss/n_val:.4f}, val_acc {100*val_acc/n_val:5.2f}%')
    train_losses.append(train_epoch_loss/n_train)
    val_losses.append(val_epoch_loss/n_val)

# plot loss history
plt.plot(range(len(train_losses)), train_losses, label='training loss')
plt.plot(range(len(val_losses)), val_losses, label='validation loss')
plt.legend()
plt.savefig('/content/drive/MyDrive/Deep-Learning-Workspace/shallow_nn/history.png')