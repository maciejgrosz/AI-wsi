from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


def objective_function(x):
    return x ** 2 * np.sin(x) + 100 * np.sin(x) * np.cos(x)


class ActivationRelu:
    @staticmethod
    def fun(x):
        return np.where(x >= 0.0, x, 0.0)

    @staticmethod
    def prim(x):
        return np.where(x >= 0.0, 1.0, 0.0)


class ActivationSigmoid:
    @staticmethod
    def fun(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def prim(x):
        value = 1 / (1 + np.exp(-x))
        return value * (1 - value)


class ActivationTanh:
    @staticmethod
    def fun(x):
        return np.tanh(x)

    @staticmethod
    def prim(x):
        return 1 - np.tanh(x) ** 2


class ActivationLinear:
    @staticmethod
    def fun(x):
        return x

    @staticmethod
    def prim(x):
        return np.where(True, 1, 0)


np.random.seed(0)


class NN:
    def __init__(self, layers, functions):
        self.layers = layers
        self.num_layers = len(layers)
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        self.activation_functions = functions
        self.layer_outputs = []
        self.layer_activations = []

    def feed_forward(self, x):
        self.layer_outputs = []
        self.layer_activations = [x]
        for b, w, f in zip(self.biases, self.weights, self.activation_functions):
            self.layer_outputs.append(np.dot(w, self.layer_activations[-1]) + b)
            self.layer_activations.append(f.fun(self.layer_outputs[-1]))
        return self.layer_outputs[-1]

    def loss(self, output, y):
        return 0.5 * (output - y) ** 2

    def d_loss(self, output, y):
        return output - y

    def SGD(self, train_data, epochs, mini_batch_size, lr):
        mini_batches = []
        for _ in tqdm(range(epochs)):
            np.random.shuffle(train_data)
            for j in range(0, len(train_data), mini_batch_size):
                mini_batches.append(train_data[j : j + mini_batch_size])

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, lr)

    def update_mini_batch(self, mini_batch, lr):
        mb_b = [np.zeros(b.shape) for b in self.biases]
        mb_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            mb_b = [b + d for b, d in zip(mb_b, delta_b)]
            mb_w = [nw + dnw for nw, dnw in zip(mb_w, delta_w)]

        self.weights = [
            w - (lr / len(mini_batch)) * nw for w, nw in zip(self.weights, mb_w)
        ]
        self.biases = [
            b - (lr / len(mini_batch)) * nb for b, nb in zip(self.biases, mb_b)
        ]

    def backprop(self, x, y):
        mb_b = [np.zeros(b.shape) for b in self.biases]
        mb_w = [np.zeros(w.shape) for w in self.weights]

        output = np.sum(self.feed_forward(x))
        delta = self.d_loss(output, y) * self.activation_functions[-2].prim(
            self.layer_outputs[-1]
        )
        mb_b[-1] = delta
        mb_w[-1] = np.dot(delta, self.layer_activations[-2].T)

        for l in range(2, self.num_layers):
            z = self.layer_outputs[-l]
            sp = self.activation_functions[l].prim(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            mb_b[-l] = delta
            mb_w[-l] = np.dot(delta, self.layer_activations[-l - 1].transpose())
        return (mb_b, mb_w)

    def predict(self, x):
        results = []
        for x in x_train:
            results.append(self.feed_forward(x))
        return results


def data_prepare(start, stop, n):
    x_train = np.linspace(start, stop, n)
    y_train = objective_function(x_train)
    x_train = x_train / x_train.max()
    y_train = y_train / y_train.max()
    train_data = []
    for x, y in zip(x_train, y_train):
        train_data.append((x, y))
    return x_train, y_train, train_data


if __name__ == "__main__":
    x_train, y_train, train_data = data_prepare(-40, 40, 200)

    act_tanh = ActivationTanh()
    nn = NN([1, 40, 40, 40, 40, 40, 40, 1], [act_tanh] * 8)
    nn.SGD(train_data, 40, 10, 0.003)
    results = nn.predict(x_train)

    plt.scatter(x_train, results, marker="+")
    plt.scatter(x_train, y_train, marker=".")
    plt.show()
