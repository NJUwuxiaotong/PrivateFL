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


def bound(value, range):
    """
    :param value: float
    :param range: float >= 0
    :return:
    """
    if -1.0 * range <= value <= range:
        return value
    elif value < -1.0 * range:
        return -1.0 * range
    else:
        return range


def random_value_with_probs(probs, num=1, is_duplicate=False):
    """
    :param probs: list
    :return: index
    """

    chosen_indexes = list()
    chosen_indexes_no = 0
    p = np.array(probs)

    while chosen_indexes_no <= num:
        chosen_index = np.random.choice(range(len(probs)), p=p.ravel())
        if is_duplicate or \
                (not is_duplicate and chosen_index not in chosen_indexes ):
            chosen_indexes.append(chosen_index)
            chosen_indexes_no += 1
    return chosen_indexes


def model_l2_norm(model_params):
    l2_norm = 0.0
    for name, params in model_params.items():
        l2_norm += torch.norm(model_params[name], p=2)
    return l2_norm


def gradient_l2_norm(model_gradient):
    l2_norm = 0.0
    for gradient in model_gradient:
        l2_norm += torch.norm(gradient, p=2)
    return l2_norm
