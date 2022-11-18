import numpy as np


def gaussian_mechanism(mu, sigma, num):
    return np.random.normal(mu, sigma, num)


def laplace_mechanism(loc, scale, num):
    return np.random.laplace(loc, scale, num)
