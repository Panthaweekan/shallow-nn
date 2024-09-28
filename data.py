import pickle

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def get_mnist1d_loader(batch_size):
    """
    Load MNIST1D dataset
    :param batch_size: Batch size
    :return: Tuple of DataLoader (Training set, Test set)
    """

    with open('dataset/mnist1d_data.pkl', 'rb') as handle:
        data = pickle.load(handle)

    x_train = torch.Tensor(data['x'])
    x_test  = torch.Tensor(data['x_test'])
    y_train = torch.LongTensor(data['y'])
    y_test  = torch.LongTensor(data['y_test'])

    ds_train = TensorDataset(x_train, y_train)
    ds_test = TensorDataset(x_test, y_test)

    print(f'Number of training examples = {len(ds_train)}')
    print(f'Number of test examples = {len(ds_test)}')

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    return dl_train, dl_test


def debug():
    dl_trn, dl_val = get_mnist1d_loader(batch_size=100)

    print('Training set:')
    for i, (x_batch, y_batch) in enumerate(dl_trn):
        print(f'\tBatch #{i+1}: X is {x_batch.shape}, Y is {y_batch.shape}')

    print('Validation set:')
    for i, (x_batch, y_batch) in enumerate(dl_val):
        print(f'\tBatch #{i+1}: X is {x_batch.shape}, Y is {y_batch.shape}')



if __name__ == '__main__':

    debug()




