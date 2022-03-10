import argparse

from matplotlib import pyplot as plt
from numpy import linalg as LA
from tqdm import tqdm
import numpy as np
from Evolution_Strategy import evolution_strategy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--Sigma", type=int)
    parser.add_argument("-l", "--Lambda", type=int)
    parser.add_argument("-m", "--Mi", type=int)
    parser.add_argument("-f", "--function", type=str)
    return parser.parse_args()


def generating_vector_x(s, d):
    np.random.seed(s)
    return np.random.uniform(-100, 100, d)


def cost_function(vector):
    return (
        ((LA.norm(vector) ** 2 - len(vector)) ** 2) ** (1 / 8)
        + len(vector) ** (-1) * ((1 / 2) * LA.norm(vector) ** 2 + sum(vector))
        + 1 / 2
    )


def spheric_func(vector):
    return sum(list(map(lambda x: pow(x, 2), vector)))


def main(arg: argparse.Namespace) -> int:
    max_iter = 1000
    sum_SA_results = [0] * max_iter
    sum_LMR_results = [0] * max_iter
    n = 5
    sigma = arg.Sigma
    _lambda = arg.Lambda
    mi = arg.Mi
    func_name = arg.function
    function = 0
    if func_name == "q":
        function = cost_function
    elif func_name == "s":
        function = spheric_func
    else:
        print("Wrong or miss argument for function: pass q to use cost function or s to use spheric function")
        exit()

    for i in tqdm(range(0, n), desc="Progress bar"):
        vector = generating_vector_x(i, 10)
        SA_results = evolution_strategy(
            vector,
            function,
            sigma,
            _lambda,
            mi,
            True,
            max_iter,
        )
        LMR_results = evolution_strategy(
            vector,
            function,
            sigma,
            _lambda,
            mi,
            False,
            max_iter,
        )
        sum_SA_results = np.add(sum_SA_results, SA_results)
        sum_LMR_results = np.add(sum_LMR_results, LMR_results)

    mean_SA_results = np.divide(sum_SA_results, n)
    mean_LMR_results = np.divide(sum_LMR_results, n)

    plt.semilogy(range(0, max_iter), mean_SA_results)
    plt.semilogy(range(0, max_iter), mean_LMR_results)
    plt.legend(["SA", "LMR"])
    if func_name == "q":
        plt.title("Cost Function")
    if func_name == "s":
        plt.title("Spheric Function")
    mean_SA_results.sort()
    mean_LMR_results.sort()
    print(f"SA result {mean_SA_results[0]}")
    print(f"LMR result {mean_LMR_results[0]}")
    plt.show()
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(args))
