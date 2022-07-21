import torch
from torch import tensor


def analyze_dist_of_single_att(label_set: tensor):
    x = label_set.reshape(1, -1)[0]
    result = torch.bincount(x)
    return result
