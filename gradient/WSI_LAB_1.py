import random
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint

vector_length = 20  # needed when you want to generate vector
alpha = 100
tolerance = 0.01
step = 0.1
max_loops = 1000
decrease_g = 0.9  # decreasing learn rate have to be < 1
decrease_n = 0.9
step_condition = 0.0001


def generating_vector_x():
    vector_x = []
    for i in range(0, vector_length):
        vector_x.append(random.randint(-100, 100))
    return vector_x


def objective(vector):
    return (
        alpha ** ((i - 1) / (len(vector) - 1)) * xi ** 2
        for i, xi in enumerate(vector, start=1)
    )


def auto_derivative(vector):
    h = 1e-5
    derivatives = []
    vector_h = []
    vector_h2 = []
    for vec in vector:
        vector_h.append(vec + h)
        vector_h2.append(vec - h)

    derivatives.append(
        np.divide(
            (np.subtract(list(objective(vector_h)), list(objective(vector_h2)))),
            (2 * h),
        )
    )
    return derivatives


def nested_list(values):
    lst = []
    for value in values:
        if isinstance(value, float):
            lst.append(value)
        else:
            lst.extend(nested_list(value))
    return lst


def gradient_descent(vector, learn_rate):
    loop = 0
    objectives_list = []
    while np.linalg.norm(auto_derivative(vector), 2) >= tolerance:  # L2 norm
        # https://cmci.colorado.edu/classes/INFO-4604/files/slides-4_optimization.pdf
        loop = loop + 1
        prev = sum(nested_list(list(objective(vector))))
        gradient = auto_derivative(vector)
        multiplied_vector = nested_list(learn_rate * np.asarray(gradient))
        vector = np.subtract(vector, multiplied_vector)
        objectives_list.append(sum(nested_list(list(objective(vector)))))
        if prev - sum(nested_list(list(objective(vector)))) < step_condition:
            learn_rate = learn_rate * decrease_g
            if learn_rate < step_condition:
                print(
                    f"[Gradient]Learn rate decreased to: {learn_rate}, function cannot find minimum"
                )
                break
            if decrease_g != 1:
                objectives_list.pop()
                loop = loop - 1
        if loop > max_loops:
            break

    print(f"\nloops in gradient descent: {loop} \n")
    pprint(f"gradient descent output {objectives_list}")
    return objectives_list


def second_derivative(vector):
    h = 0.1
    derivatives = []
    vector_h = []
    vector_h2 = []
    for vec in vector:
        vector_h.append(vec + h)
        vector_h2.append(vec - h)
# Disgusting 
    derivatives.append(
        np.divide(
            (
                [
                    sum(i)
                    for i in zip(
                    np.subtract(
                        list(objective(vector_h)),
                        np.multiply(2, list(objective(vector))),
                    ),
                    list(objective(vector_h2)),
                )
                ]
            ),
            (h * h),
        )
    )
    derivatives = nested_list(derivatives)
    return derivatives


def hessian(vector):
    x = second_derivative(vector)
    hes = np.zeros((len(x), len(x)), float)
    np.fill_diagonal(hes, x)
    return hes


def inv_hessian(vector):
    inv = np.linalg.inv(hessian(vector))
    return inv


def newton(vector, learn_rate):
    loop = 0
    objectives_list = []

    while np.linalg.norm(auto_derivative(vector), 2) >= tolerance:
        loop = loop + 1
        prev = sum(nested_list(list(objective(vector))))
        p = np.matmul(inv_hessian(vector), nested_list(auto_derivative(vector)))
        multiplied_hes_grad = learn_rate * p
        vector = np.subtract(vector, multiplied_hes_grad)
        objectives_list.append(sum(nested_list(list(objective(vector)))))
        if loop > max_loops:
            break
        if prev - sum(nested_list(list(objective(vector)))) < step_condition:
            learn_rate = learn_rate * decrease_n
            if learn_rate < 0.0001:
                print(
                    f"[Newton] Learn rate decreased to: {learn_rate}, function cannot find minimum"
                )
                break
            if decrease_n != 1:
                objectives_list.pop()
                loop = loop - 1

    print(f"\nloops in newton: {loop} \n")
    pprint(f"newton output {objectives_list}")

    return objectives_list


def main():
    learn_rate = step
    # input_vector = generating_vector_x()
    # input_vector = [76, -5, 96, 18, -43, -63, 54, 71, 48, 7]
    input_vector = [
        -3,
        84,
        -11,
        -57,
        84,
        -15,
        56,
        69,
        70,
        -57,
        -28,
        -80,
        10,
        20,
        73,
        -28,
        -28,
        -44,
        -54,
        -19,
    ]
    print(f"\ninput_vector: {input_vector}")
    gradient_objective_list = gradient_descent(input_vector, learn_rate)
    newton_objective_list = newton(input_vector, learn_rate)
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle("1st -> gradient ||  2nd -> newton")
    ax1.plot(gradient_objective_list, "ro")
    ax2.plot(newton_objective_list, "ro")

    plt.show()


if __name__ == "__main__":
    main()
