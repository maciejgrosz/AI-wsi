import numpy as np


def rbf_kernel(xi, xj, sigma):
    return np.exp(-np.linalg.norm(xi - xj) / (2 * sigma ** 2))


def poly_kernel(xi, xj, p):
    return (np.dot(xi, xj) + 1) ** p


