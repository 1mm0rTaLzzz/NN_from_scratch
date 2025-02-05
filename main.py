import numpy as np


# Функция активации сигмоида
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # Преобразует входные значения в диапазон (0,1)


# Производная сигмоиды, используется при обратном распространении ошибки
def sigmoid_derivative(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


# Функция среднеквадратичной ошибки (MSE)
def mse(y_pred, y_true):
    return np.mean(
        (y_pred - y_true) ** 2)  # Среднее арифметическое квадратов разницы между предсказанием и истинным значением


class MyNeuralNetwork:
    def __init__(self):
        np.random.seed(42)  # Фиксируем случайное начальное состояние для воспроизводимости

        self.h1 = None
        self.h2 = None
        self.o1 = None

        # Инициализируем веса случайными значениями из нормального распределения
        self.w1 = np.random.normal()  # Вход -> скрытый нейрон 1
        self.w2 = np.random.normal()  # Вход -> скрытый нейрон 1
        self.w3 = np.random.normal()  # Вход -> скрытый нейрон 2
        self.w4 = np.random.normal()  # Вход -> скрытый нейрон 2
        self.w5 = np.random.normal()  # Скрытые нейроны -> выходной нейрон
        self.w6 = np.random.normal()  # Скрытые нейроны -> выходной нейрон

        # Инициализируем смещения (биасы) случайными значениями
        self.b1 = np.random.normal()  # Смещение скрытого нейрона 1
        self.b2 = np.random.normal()  # Смещение скрытого нейрона 2
        self.b3 = np.random.normal()  # Смещение выходного нейрона

    def feedforward(self, X):
        """Прямое распространение (forward pass)"""

        # Вычисляем выход скрытых нейронов с помощью сигмоиды
        self.h1 = sigmoid(self.w1 * X[0] + self.w2 * X[1] + self.b1)  # Первый скрытый нейрон
        self.h2 = sigmoid(self.w3 * X[0] + self.w4 * X[1] + self.b2)  # Второй скрытый нейрон

        # Вычисляем выходной сигнал сети
        self.o1 = sigmoid(self.w5 * self.h1 + self.w6 * self.h2 + self.b3)

        return self.o1  # Возвращаем предсказанный результат
