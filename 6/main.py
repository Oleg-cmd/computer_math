import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from prettytable import PrettyTable

def f1(x, y):
    return x + y

def f2(x, y):
    return y - x

def f3(x, y):
    return x * y

# Метод Эйлера
def euler_method(f, x0, y0, xn, h, E):
    n = int((xn - x0) / h) + 1
    x = np.linspace(x0, xn, n)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])
        if np.abs(y[i + 1] - y[i]) < E:
            x = x[:i+2]
            y = y[:i+2]
            break
    return x, y

# Усовершенствованный метод Эйлера
def improved_euler_method(f, x0, y0, xn, h, E):
    n = int((xn - x0) / h) + 1
    x = np.linspace(x0, xn, n)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h, y[i] + k1)
        y[i + 1] = y[i] + (k1 + k2) / 2
        if np.abs(y[i + 1] - y[i]) < E:
            x = x[:i+2]
            y = y[:i+2]
            break
    return x, y

# Метод Рунге-Кутта 4-го порядка
def rk4_method(f, x0, y0, xn, h, E):
    n = int((xn - x0) / h) + 1
    x = np.linspace(x0, xn, n)
    y = np.zeros(n)
    y[0] = y0
    for i in range(n - 1):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if np.abs(y[i + 1] - y[i]) < E:
            x = x[:i+2]
            y = y[:i+2]
            break
    return x, y

# Метод Адамса
def adams_method(f, x0, y0, xn, h, E):
    x_rk, y_rk = rk4_method(f, x0, y0, x0 + 3 * h, h, E)
    x = np.arange(x0, xn + h, h)
    y = np.zeros(len(x))
    y[:4] = y_rk
    for i in range(3, len(x) - 1):
        y[i + 1] = y[i] + h * (55 * f(x[i], y[i]) - 59 * f(x[i - 1], y[i - 1]) + 37 * f(x[i - 2], y[i - 2]) - 9 * f(x[i - 3], y[i - 3])) / 24
        if np.abs(y[i + 1] - y[i]) < E:
            x = x[:i+2]
            y = y[:i+2]
            break
    x = np.clip(x, x0, xn) 
    return x, y



# Ввод данных с клавиатуры
print("Выберите функцию для решения:")
print("1 - dy/dx = x + y")
print("2 - dy/dx = y - x")
print("3 - dy/dx = xy")
func_id = input("Введите номер функции: ")

if func_id == "1":
    f = f1
elif func_id == "2":
    f = f2
elif func_id == "3":
    f = f3
else:
    print("Некорректный выбор функции")
    exit()

print("Введите начальные условия (x0, y0):")
x0 = float(input("x0: "))
y0 = float(input("y0: "))
xn = float(input("Введите конец интервала (xn): "))
h = float(input("Введите шаг (h): "))
E = float(input("Введите точность (E): "))

print("\nВыберите метод для решения:")
print("1 - Метод Эйлера")
print("2 - Усовершенствованный метод Эйлера")
print("3 - Метод Рунге-Кутта 4-го порядка")
print("4 - Метод Адамса")

method_id = input("Введите номер метода: ")

if method_id == "1":
    method_name = "Метод Эйлера"
    method = euler_method
elif method_id == "2":
    method_name = "Усовершенствованный метод Эйлера"
    method = improved_euler_method
elif method_id == "3":
    method_name = "Метод Рунге-Кутта 4-го порядка"
    method = rk4_method
elif method_id == "4":
    method_name = "Метод Адамса"
    method = adams_method
else:
    print("Некорректный выбор метода")
    exit()

# Вычисляем приближенные значения интеграла дифференциального уравнения выбранным методом
x, y = method(f, x0, y0, xn, h, E)

# Точное решение с использованием solve_ivp
def exact_solution_solve_ivp(f, x0, y0, xn, x_points):
    sol = solve_ivp(f, [x0, xn], [y0], t_eval=x_points)
    return sol.y[0]

y_exact_values = exact_solution_solve_ivp(f, x0, y0, xn, x)

# Оценка точности
E_max = max(np.abs(y_exact_values - y))
print(f"\n{method_name}:")

# Вывод результатов в таблицу
table = PrettyTable()
table.field_names = ["x", "y (численное)", "y (точное)", "delta"]
for x_i, y_i, y_exact_i in zip(x, y, y_exact_values):
    table.add_row([f"{x_i:.4f}", f"{y_i:.4f}", f"{y_exact_i:.4f}", f"{np.abs(y_i - y_exact_i):.4f}"])

print(table)
print(f"Оценка точности: E_max = {E_max:.4f}")

# Строим график
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=method_name)
x_exact = np.linspace(x0, xn, 100)
y_exact = exact_solution_solve_ivp(f, x0, y0, xn, x_exact)
plt.plot(x_exact, y_exact, label="Точное решение", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()
