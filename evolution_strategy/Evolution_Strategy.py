from math import sqrt

import numpy as np
from matplotlib import pyplot as plt


def multinormal(d):
    return np.random.multivariate_normal(np.zeros(d), np.eye(d))


def evolution_strategy(input_vector, cost_function, sigma, _lambda, mi, SA, max_iter):
    # When SA == True algorithm is using self-adaptation method
    # When SA == False algorithm is using LMR method
    t = 0
    tau = 1 / sqrt(len(input_vector))
    wage = 1 / mi
    values = []
    vectors = []
    sigmas = []
    results = []
    while t < max_iter:
        for k in range(0, _lambda):
            ksi = tau * np.random.normal(0, 1)
            z = multinormal(10)
            if SA:
                sigma = sigma * np.exp(ksi)
            gen_vector = np.asarray(input_vector) + sigma * z
            sigmas.append(sigma)
            values.append(cost_function(gen_vector))
            vectors.append(gen_vector)

        index_results = sorted(range(len(values)), key=lambda l: values[l])
        mi_vectors = [vectors[i] for i in index_results[:mi]]
        if SA:
            mi_sigmas = [sigmas[i] for i in index_results[:mi]]
            sigma = sum(np.multiply(wage, mi_sigmas))
        else:
            sigma = sigma * np.exp(tau * np.random.normal(0, 1))
        input_vector = sum(np.multiply(wage, mi_vectors))
        results.append(cost_function(input_vector))
        t += 1
    return results
