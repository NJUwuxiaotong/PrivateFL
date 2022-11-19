import numpy as np


def get_value_gaussian_dist(mu, sigma, num):
    return np.random.normal(mu, sigma, num)


def get_value_lap_dist(loc, scale, num):
    return np.random.laplace(loc, scale, num)
