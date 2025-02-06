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

    def train(self, X, y, epochs=18000, lr=0.1):
        """Обучение нейросети методом обратного распространения ошибки (backpropagation)"""
        for epoch in range(epochs):  # Количество итераций (эпох)
            total_loss = 0  # Общая ошибка за эпоху

            for i in range(len(X)):  # Проходимся по всем обучающим примерам
                self.feedforward(X[i])  # Выполняем прямое распространение

                # Вычисляем ошибку (MSE)
                loss = mse(self.o1, y[i])
                total_loss += loss  # Суммируем ошибку для текущей эпохи

                # Вычисляем градиент ошибки на выходе (ошибка выходного нейрона)
                d_o1 = 2 * (self.o1 - y[i]) * sigmoid_derivative(self.o1)

                # Градиенты для скрытых нейронов (ошибка скрытых нейронов)
                d_h1 = d_o1 * self.w5 * sigmoid_derivative(self.h1)
                d_h2 = d_o1 * self.w6 * sigmoid_derivative(self.h2)

                # Обновляем веса выходного слоя
                self.w5 -= lr * d_o1 * self.h1
                self.w6 -= lr * d_o1 * self.h2
                self.b3 -= lr * d_o1  # Обновляем смещение выходного нейрона

                # Обновляем веса скрытого слоя
                self.w1 -= lr * d_h1 * X[i][0]
                self.w2 -= lr * d_h1 * X[i][1]
                self.b1 -= lr * d_h1  # Обновляем смещение скрытого нейрона 1

                self.w3 -= lr * d_h2 * X[i][0]
                self.w4 -= lr * d_h2 * X[i][1]
                self.b2 -= lr * d_h2  # Обновляем смещение скрытого нейрона 2

            # Выводим ошибку каждые 1000 эпох
            if epoch % 1000 == 0:
                print(f"Эпоха {epoch}, ошибка {total_loss / len(X)}")

    def predict(self, X):
        """Функция для получения бинарного предсказания (0 или 1)"""
        return 1 if self.feedforward(X) >= 0.5 else 0


# Данные для обучения (таблица истинности XOR)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Входные данные (два бита)
y = np.array([0, 1, 1, 0])  # Ожидаемые результаты (XOR)

nn = MyNeuralNetwork()
nn.train(X, y)

for x in X:
    print(
        f"Вход: {x}, Предсказание (вероятность): {nn.feedforward(x):.4f}, Класс: {nn.predict(x)}")
