import numpy as np

import torch
from torch import tensor

from utils import set_random_seed


def analyze_dist_of_single_att(label_set: tensor):
    x = label_set.reshape(1, -1)[0]
    result = torch.bincount(x)
    return result


def random_seed(seed=None, modelkey=None):
    """Return various models."""
    if modelkey is None:
        if seed is None:
            # model_init_seed = np.random.randint(0, 2**32 - 10)
            model_init_seed = np.random.randint(0, 2**31)
        else:
            model_init_seed = seed
    else:
        model_init_seed = modelkey
    set_random_seed(model_init_seed)
    return model_init_seed
