import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Определение функции
def f(x):
    return 3 * x / (x ** 4 + 7)

# Указанный интервал и шаг
a = -2
b = 0
h = 0.2

# Генерация значений x на интервале с шагом h
x_values = np.arange(a, b + h, h)

# Вычисление значений y для каждого x
y_values = [f(x) for x in x_values]

# Создание таблицы табулирования
table = pd.DataFrame({'x': x_values, 'y': y_values})

# Выбор 11 точек для приближения
selected_points = table.iloc[::1, :]  # каждая точка

# Построение линейного приближения
coefficients_linear = np.polyfit(selected_points['x'], selected_points['y'], 1)
linear_approximation = np.poly1d(coefficients_linear)

# Построение квадратичного приближения
coefficients_quadratic = np.polyfit(selected_points['x'], selected_points['y'], 2)
quadratic_approximation = np.poly1d(coefficients_quadratic)

# Вычисление значений линейного приближения на всем интервале
x_range = np.linspace(a, b, 100)
y_linear = linear_approximation(x_range)

# Вычисление значений квадратичного приближения на всем интервале
y_quadratic = quadratic_approximation(x_range)

# Разница между значениями функции модели и исходными данными для линейного приближения
diff_linear = linear_approximation(selected_points['x']) - selected_points['y']

# Квадрат разницы
squared_diff_linear = diff_linear ** 2

# Среднее значение квадратов разницы
mean_squared_diff_linear = np.mean(squared_diff_linear)

# Среднеквадратическое отклонение для линейного приближения
mse_linear = np.sqrt(mean_squared_diff_linear)

# Вывод промежуточных значений для линейного приближения
print("Линейное приближение:")
print("Разница между значениями функции модели и исходными данными:")
print('\n'.join(f"{i}:{val:.3f}" for i, val in enumerate(diff_linear.round(3))))
print("Квадрат разницы:")
print('\n'.join(f"{i}:{val:.3f}" for i, val in enumerate(squared_diff_linear.round(3))))
print(f"Среднее значение квадратов разницы: {mean_squared_diff_linear:.3f}")
print(f"Среднеквадратическое отклонение: {mse_linear:.3f}")
print()

# Разница между значениями функции модели и исходными данными для квадратичного приближения
diff_quadratic = quadratic_approximation(selected_points['x']) - selected_points['y']

# Квадрат разницы
squared_diff_quadratic = diff_quadratic ** 2

# Среднее значение квадратов разницы
mean_squared_diff_quadratic = np.mean(squared_diff_quadratic)

# Среднеквадратическое отклонение для квадратичного приближения
mse_quadratic = np.sqrt(mean_squared_diff_quadratic)

# Вывод промежуточных значений для квадратичного приближения
print("Квадратичное приближение:")
print("Разница между значениями функции модели и исходными данными:")
print('\n'.join(f"{i}:{val:.3f}" for i, val in enumerate(diff_quadratic.round(3))))
print("Квадрат разницы:")
print('\n'.join(f"{i}:{val:.3f}" for i, val in enumerate(squared_diff_quadratic.round(3))))
print(f"Среднее значение квадратов разницы: {mean_squared_diff_quadratic:.3f}")
print(f"Среднеквадратическое отклонение: {mse_quadratic:.3f}")


best_approximation = None
if mse_linear < mse_quadratic:
    best_approximation = "Линейное приближение"
    best_mse = mse_linear
else:
    best_approximation = "Квадратичное приближение"
    best_mse = mse_quadratic

# Вывод наилучшего приближения
print(f"Наилучшее приближение: {best_approximation}")
print(f"Среднеквадратическое отклонение: {best_mse:.3f}")


selected_points_11 = table.iloc[::1, :]  # каждая точка

# Построение линейного приближения по 11 точкам
coefficients_linear_11 = np.polyfit(selected_points_11['x'], selected_points_11['y'], 1)
linear_approximation_11 = np.poly1d(coefficients_linear_11)

# Построение квадратичного приближения по 11 точкам
coefficients_quadratic_11 = np.polyfit(selected_points_11['x'], selected_points_11['y'], 2)
quadratic_approximation_11 = np.poly1d(coefficients_quadratic_11)

# Вывод коэффициентов приближающих функций
print("Коэффициенты линейного приближения (11 точек):")
print(f"Коэффициент a: {coefficients_linear_11[0]:.3f}")
print(f"Коэффициент b: {coefficients_linear_11[1]:.3f}")
print()
print("Коэффициенты квадратичного приближения (11 точек):")
print(f"Коэффициент a: {coefficients_quadratic_11[0]:.3f}")
print(f"Коэффициент b: {coefficients_quadratic_11[1]:.3f}")
print(f"Коэффициент c: {coefficients_quadratic_11[2]:.3f}")
print()

# Вычисление значений функций для выбранных точек
y_linear_values_11 = linear_approximation_11(selected_points_11['x'])
y_quadratic_values_11 = quadratic_approximation_11(selected_points_11['x'])

# Вывод значений функций для выбранных точек
print("Значения функции для выбранных точек (11 точек):")
print("Линейное приближение:")
for i, val in enumerate(y_linear_values_11):
    print(f"{i}: {val:.3f}")
print()
print("Квадратичное приближение:")
for i, val in enumerate(y_quadratic_values_11):
    print(f"{i}: {val:.3f}")
print()