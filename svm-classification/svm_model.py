import numpy as np
from cvxopt import matrix, solvers


class SVM:
    def __init__(self, train_x, train_y, k_param, kernel, C):
        self.train_x = train_x
        self.train_y = train_y
        self.k_param = k_param
        self.kernel = kernel
        self.C = C

    def fit(self):
        n = len(self.train_x)
        H = np.zeros(n * n).reshape(n, n)
        for i in range(n):
            for j in range(n):
                H[i][j] = self.train_y[i] * self.train_y[j] * self.kernel(self.train_x[i], self.train_x[j], self.k_param)
        q = np.negative(np.ones(n))  # * -1
        A = self.train_y
        b = 0.0
        G = np.vstack((np.eye(n) * (-1), np.eye(n)))
        h = np.hstack((np.zeros(n), np.ones(n) * self.C))
        P = matrix(H)
        q = matrix(q)
        G = matrix(G)
        h = matrix(h)
        A = matrix(A, (1, n), 'd')
        b = matrix(b)
        solution = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(solution['x'])
        return alphas

    def b_vector(self, vector, alphas):
        calc_vector = 0
        for i in range(len(self.train_x)):
            calc_vector += alphas[i]*self.train_y[i]*self.kernel(self.train_x[i], vector, self.k_param)
        return calc_vector

    def b(self, alphas):
        b = 0
        for i in range(len(self.train_x)):
            b += self.train_y[i] - self.b_vector(self.train_x[i], alphas)
        return b/len(self.train_x)

    def decision(self, alphas, test_x, b):
        a = 0
        for i in range(len(self.train_y)):
            a += self.train_y[i] * alphas[i] * self.kernel(self.train_x[i], test_x, self.k_param)
        return float(np.sign(a + b))
