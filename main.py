import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


class MyNeuralNetwork:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, X):
        h1 = sigmoid(self.w1 * X[0] + self.w2 * X[1] + self.b1)
        h2 = sigmoid(self.w3 * X[0] + self.w4 * X[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1
