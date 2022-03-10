import collections

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from svm_model import SVM
from kernels import rbf_kernel, poly_kernel


def test_svm(X_train, Y_train, X_test, Y_test, sigma, degree, kernel, C):
    kernel_parameter = sigma
    if kernel.__name__ == 'poly_kernel':
        kernel_parameter = degree
    svm_model = SVM(X_train, Y_train, kernel_parameter, kernel, C)
    alphas = svm_model.fit()
    b = svm_model.b(alphas)
    dec = []
    good_pred = 0
    Y_test_arr = np.array(Y_test)
    for i in range(len(X_test)):
        dec.append(svm_model.decision(alphas, X_test[i], b))
        if dec[i] == Y_test_arr[i]:
            good_pred += 1
    print(f'good predictions {good_pred}/{len(X_test)}  -> {(good_pred / len(X_test)) * 100}%')
    good_wines = 0
    for i in dec:
        if i == 1:
            good_wines += 1
    occur = collections.Counter(Y_test_arr)
    print(f'number of good wines {good_wines}/{len(dec)} there is {occur[1]} good wines')
    print(f'number of bad {len(dec) - good_wines}/{len(dec)} there is {occur[-1]} bad wines')

def main():
    df = pd.read_csv('winequality-red.csv', header=0, sep=';')
    condition = df['quality'] > 5
    df['quality'] = np.where(condition, 1, -1)
    X = df.drop('quality', axis=1).copy()
    Y = df['quality'].copy()
    X = normalize(X, axis=0)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2, shuffle=False, random_state=42)
    sigma = 0.1
    degree = 100
    C = 1
    test_svm(X_train, Y_train, X_test, Y_test, sigma, degree, poly_kernel, C)


if __name__ == '__main__':
    SystemExit(main())
