import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from prettytable import PrettyTable

def f1(x, y):
    return y + (1 + x) * y**2

def f2(x, y):
    return x**2 * y

def f3(x, y):
    return 3 * x**2 * y + x**2 * np.exp(x**3)

# Метод Эйлера
def euler_method(f, x0, y0, xn, h, E):
    n = int((xn - x0) / h) + 1
    x = np.linspace(x0, xn, n)
    y = np.zeros(n)
    y[0] = y0
    table = PrettyTable()
    table.field_names = ["i", "x_i", "y_i", "f(x_i, y_i)", "y_i+1"]
    for i in range(n - 1):
        y[i + 1] = y[i] + h * f(x[i], y[i])
        table.add_row([i, f"{x[i]:.4f}", f"{y[i]:.4f}", f"{f(x[i], y[i]):.4f}", f"{y[i + 1]:.4f}"])
        if np.abs(y[i + 1] - y[i]) < E:
            x = x[:i+2]
            y = y[:i+2]
            break
    print(table)
    return x, y

# Усовершенствованный метод Эйлера
def improved_euler_method(f, x0, y0, xn, h, E):
    n = int((xn - x0) / h) + 1
    x = np.linspace(x0, xn, n)
    y = np.zeros(n)
    y[0] = y0
    table = PrettyTable()
    table.field_names = ["i", "x_i", "y_i", "k1", "k2", "y_i+1"]
    for i in range(n - 1):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h, y[i] + k1)
        y[i + 1] = y[i] + (k1 + k2) / 2
        table.add_row([i, f"{x[i]:.4f}", f"{y[i]:.4f}", f"{k1:.4f}", f"{k2:.4f}", f"{y[i + 1]:.4f}"])
        if np.abs(y[i + 1] - y[i]) < E:
            x = x[:i+2]
            y = y[:i+2]
            break
    print(table)
    return x, y


# Метод Рунге-Кутта 4-го порядка
def rk4_method(f, x0, y0, xn, h, E):
    n = int((xn - x0) / h) + 1
    x = np.linspace(x0, xn, n)
    y = np.zeros(n)
    y[0] = y0
    table = PrettyTable()
    table.field_names = ["i", "x_i", "y_i", "k1", "k2", "k3", "k4", "y_i+1"]
    for i in range(n - 1):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h / 2, y[i] + k1 / 2)
        k3 = h * f(x[i] + h / 2, y[i] + k2 / 2)
        k4 = h * f(x[i] + h, y[i] + k3)
        y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        table.add_row([i, f"{x[i]:.4f}", f"{y[i]:.4f}", f"{k1:.4f}", f"{k2:.4f}", f"{k3:.4f}", f"{k4:.4f}", f"{y[i + 1]:.4f}"])
        if np.abs(y[i + 1] - y[i]) < E:
            x = x[:i+2]
            y = y[:i+2]
            break
    print(table)
    return x, y

# Метод Адамса
def adams_method(f, x0, y0, xn, h, E):
    x_rk, y_rk = rk4_method(f, x0, y0, x0 + 3 * h, h, E)
    x = np.arange(x0, xn + h, h)
    y = np.zeros(len(x))
    y[:4] = y_rk
    table = PrettyTable()
    table.field_names = ["i", "x_i", "y_i", "f(x_i, y_i)", "y_i+1"]
    for i in range(3, len(x) - 1):
        y[i + 1] = y[i] + h * (55 * f(x[i], y[i]) - 59 * f(x[i - 1], y[i - 1]) + 37 * f(x[i - 2], y[i - 2]) - 9 * f(x[i - 3], y[i - 3])) / 24
        table.add_row([i, f"{x[i]:.4f}", f"{y[i]:.4f}", f"{f(x[i], y[i]):.4f}", f"{y[i + 1]:.4f}"])
        if np.abs(y[i + 1] - y[i]) < E:
            x = x[:i+2]
            y = y[:i+2]
            break
    x = np.clip(x, x0, xn)
    print(table)
    return x, y



def exact_solution_solve_ivp(f, x0, y0, xn, x_points):
    sol = solve_ivp(lambda x, y: f(x, y), [x0, xn], [y0], t_eval=x_points)
    return sol.y[0]


print("Выберите функцию для решения:")
print("1 - y' = y + (1 + x) * y^2")
# 0 0.01
# 1
# 0.01
# 1e-6
print("2 - y' = x^2 * y")
# 0 01
# 2
# 0.1
# 1e-6

print("3 - y' = 3 * x ** 2 * y + x^2 * exp(x^3)")
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

x, y = method(f, x0, y0, xn, h, E)
y_exact_values = exact_solution_solve_ivp(f, x0, y0, xn, x)

E_max = max(np.abs(y_exact_values - y))
print(f"\n{method_name}:")

table = PrettyTable()
table.field_names = ["x", "y (численное)", "y (точное)", "delta"]
for x_i, y_i, y_exact_i in zip(x, y, y_exact_values):
    table.add_row([f"{x_i:.4f}", f"{y_i:.4f}", f"{y_exact_i:.4f}", f"{np.abs(y_i - y_exact_i):.4f}"])

print(table)
print(f"Оценка точности: E_max = {E_max:.4f}")

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