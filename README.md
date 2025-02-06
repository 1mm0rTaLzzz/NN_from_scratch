# README: Перцептрон с нуля(без torch и tensorflow)

## Описание проекта
Данный проект реализует простую нейросеть на языке Python с использованием библиотеки NumPy. Нейросеть обучается решать задачу XOR (исключающее ИЛИ) методом обратного распространения ошибки (Backpropagation).

---

## Теоретическое обоснование

### 1. Функция активации: Сигмоида
Формула сигмоиды:
sigmoid(x) = 1 / (1 + exp(-x))


Сигмоида преобразует входные значения в диапазон (0,1), что делает её полезной для задач классификации.

Производная сигмоиды:
sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))

Производная используется при обучении для расчёта градиента ошибки.

### 2. Функция ошибки: Среднеквадратичная ошибка (MSE)
Формула:
MSE = (1 / n) * Σ (y_pred - y_true)^2

Где:
- y_pred — предсказанное значение,
- y_true — истинное значение,
- n — количество примеров в наборе данных.

### 3. Обратное распространение ошибки (Backpropagation)
Обратное распространение ошибки — это метод настройки весов сети на основе ошибки предсказания. Метод использует правило цепного дифференцирования для вычисления градиента функции ошибки по параметрам модели.

#### Основные этапы:
1. **Прямое распространение:** вычисление выхода сети на основе текущих весов.
2. **Вычисление ошибки:** сравнение выхода сети с целевыми значениями с использованием MSE.
3. **Обратное распространение:** вычисление градиентов ошибки по всем весам с использованием производной сигмоиды и правила цепного дифференцирования:
   ∂E / ∂w = (∂E / ∂o) * (∂o / ∂h) * (∂h / ∂w)
   где E — ошибка, o — выход нейрона, h — значение скрытого слоя.
4. **Обновление весов:** корректировка весов с использованием градиентного спуска:
   w = w - η * (∂E / ∂w)
   где η — коэффициент обучения.

Этот метод позволяет сети постепенно минимизировать ошибку и улучшать точность предсказаний.

---

## Архитектура нейросети

Нейросеть состоит из:
- **Входного слоя** (2 входа: два бита)
- **Скрытого слоя** (2 нейрона с функцией активации сигмоиды)
- **Выходного слоя** (1 нейрон с функцией активации сигмоиды)

---
