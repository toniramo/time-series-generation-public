import pickle

import numpy as np
import torch


def get_range(x, e=1e10):
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()[0]
    ymins = x.min(axis=0)
    ymaxs = x.max(axis=0)
    
    ymins = np.nan_to_num(ymins, nan=-e, neginf=-e)
    ymaxs = np.nan_to_num(ymaxs, nan=e, posinf=e)
    return ymins, ymaxs

def sample_indices(dataset_size, batch_size):
    size = min(dataset_size, batch_size)
    indices = torch.from_numpy(np.random.choice(dataset_size, size=size, replace=False)).cpu()#.cuda()
    # functions torch.-multinomial and torch.-choice are extremely slow -> back to numpy
    return indices


def pickle_it(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def to_numpy(x):
    """
    Casts torch.Tensor to a numpy ndarray.

    The function detaches the tensor from its gradients, then puts it onto the cpu and at last casts it to numpy.
    """
    return x.detach().cpu().numpy()
