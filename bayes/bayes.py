from os import sep
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import datasets


class Bayes:
    def __init__(self, target_name):
        self.target = target_name
        self.prior_prob = {}
        self.means = {}
        self.var = {}
        self.seperated = {}

    def seperate(self):
        for i in range(len(self.vectors)):
            vector = self.vectors[i]
            class_v = vector[-1]
            if class_v not in self.seperated:
                self.seperated[class_v] = []
            self.seperated[class_v].append(vector[:4])

    def gaussian_likelihood(self, mu, sigma, x):
        return (2 * np.pi * sigma ** 2) ** (-0.5) * np.exp(
            -((x - mu) ** 2) / (2 * sigma ** 2)
        )

    def train(self, training_data):
        self.classes = training_data[self.target].unique()
        self.vectors = training_data.values.tolist()
        class_vals = training_data[self.target].tolist()

        self.seperate()

        for class_ in self.classes:
            self.prior_prob[class_] = 0
        for val in class_vals:
            self.prior_prob[val] += 1
        for class_ in self.classes:
            self.prior_prob[class_] = self.prior_prob[class_] / len(
                class_vals
            )  # get dict with probability of each species

        for class_ in self.seperated:
            self.means[class_] = np.mean(self.seperated[class_], axis=0)
            self.var[class_] = np.var(self.seperated[class_], axis=0)

    def predict(self, predict_vector):
        class_prob = {}
        for class_ in self.classes:
            gauss = []
            for i in range(len(self.vectors[0]) - 1):
                gauss.append(
                    self.gaussian_likelihood(
                        self.means[class_][i], self.var[class_][i], predict_vector[i]
                    )
                )
            class_prob[class_] = self.prior_prob[class_] * np.prod(gauss)
        return max(class_prob, key=class_prob.get)


def test(X, y):
    X = X.values.tolist()
    y = y.values.flatten().tolist()
    correct = []
    for entry, output in zip(X, y):

        if classifier.predict(entry) == output:
            correct.append(entry)
    return correct


iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.DataFrame(iris.target, columns=["Target"])
target_names = iris["target_names"]
target_map = dict((i, target_names[i]) for i in range(len(target_names)))
y["Target"] = pd.Series([(lambda x: target_map[x])(i) for i in iris["target"]])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train_df = pd.concat([X_train, y_train], axis=1)
classifier = Bayes("Target")
classifier.train(train_df)


print(
    f"Accuracy: {len(test(X_test,y_test))/len(X_test) * 100} %  - {len(test(X_test,y_test))}/{len(X_test)}"
)
