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


def euler_step(f, x, y, h):
    return y + h * f(x, y)

def improved_euler_step(f, x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h, y + k1)
    return y + (k1 + k2) / 2

def rk4_step(f, x, y, h):
    k1 = h * f(x, y)
    k2 = h * f(x + h / 2, y + k1 / 2)
    k3 = h * f(x + h / 2, y + k2 / 2)
    k4 = h * f(x + h, y + k3)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

def euler_method(f, x0, y0, xn, h, E):
    p = 1  # Порядок метода Эйлера
    x = [x0]
    y = [y0]
    table = PrettyTable()
    table.field_names = ["i", "x_i", "y_i", "f(x_i, y_i)", "y_i+1"]
    i = 0
    iterations = 0
    while x[-1] < xn:
        x_new = x[-1] + h
        if x_new > xn:
            x_new = xn
        y_new = euler_step(f, x[-1], y[-1], h)
        y_half_step = euler_step(f, x[-1], y[-1], h / 2)
        y_half = euler_step(f, x[-1] + h / 2, y_half_step, h / 2)
        
        R = np.abs(y_new - y_half) / (2**p - 1)
        
        if R <= E:
            x.append(x_new)
            y.append(y_new)
            table.add_row([i, f"{x[-1]:.4f}", f"{y[-1]:.4f}", f"{f(x[-1], y[-1]):.4f}", f"{y_new:.4f}"])
            i += 1
        else:
            h /= 2
        iterations += 1

    print(table)
    return np.array(x), np.array(y), iterations

def improved_euler_method(f, x0, y0, xn, h, E):
    p = 2  # Порядок улучшенного метода Эйлера
    x = [x0]
    y = [y0]
    table = PrettyTable()
    table.field_names = ["i", "x_i", "y_i", "k1", "k2", "y_i+1"]
    i = 0
    iterations = 0
    while x[-1] < xn:
        x_new = x[-1] + h
        if x_new > xn:
            x_new = xn
        y_new = improved_euler_step(f, x[-1], y[-1], h)
        y_half_step = improved_euler_step(f, x[-1], y[-1], h / 2)
        y_half = improved_euler_step(f, x[-1] + h / 2, y_half_step, h / 2)
        
        R = np.abs(y_new - y_half) / (2**p - 1)
        
        if R <= E:
            x.append(x_new)
            y.append(y_new)
            table.add_row([i, f"{x[-1]:.4f}", f"{y[-1]:.4f}", f"{h*f(x[-1], y[-1]):.4f}", f"{h*f(x[-1] + h, y[-1] + h*f(x[-1], y[-1])):.4f}", f"{y_new:.4f}"])
            i += 1
        else:
            h /= 2
        iterations += 1

    print(table)
    return np.array(x), np.array(y), iterations


# Метод Рунге-Кутта 4-го порядка

def rk4_method(f, x0, y0, xn, h, E):
    p = 4  # Порядок метода Рунге-Кутта
    x = [x0]
    y = [y0]
    table = PrettyTable()
    table.field_names = ["i", "x_i", "y_i", "k1", "k2", "k3", "k4", "y_i+1"]
    i = 0
    iterations = 0
    while x[-1] < xn:
        x_new = x[-1] + h
        if x_new > xn:
            x_new = xn
        y_new = rk4_step(f, x[-1], y[-1], h)
        y_half_step = rk4_step(f, x[-1], y[-1], h / 2)
        y_half = rk4_step(f, x[-1] + h / 2, y_half_step, h / 2)
        
        R = np.abs(y_new - y_half) / (2**p - 1)
        
        if R <= E:
            x.append(x_new)
            y.append(y_new)
            k1 = h * f(x[-1], y[-1])
            k2 = h * f(x[-1] + h / 2, y[-1] + k1 / 2)
            k3 = h * f(x[-1] + h / 2, y[-1] + k2 / 2)
            k4 = h * f(x[-1] + h, y[-1] + k3)
            table.add_row([i, f"{x[-1]:.4f}", f"{y[-1]:.4f}", f"{k1:.4f}", f"{k2:.4f}", f"{k3:.4f}", f"{k4:.4f}", f"{y_new:.4f}"])
            i += 1
        else:
            h /= 2
        iterations += 1

    print(table)
    return np.array(x), np.array(y), iterations
# Метод Адамса

def adams_method(f, x0, y0, xn, h, E):
    x_rk, y_rk, _ = rk4_method(f, x0, y0, x0 + 3 * h, h, E)
    x = list(x_rk)
    y = list(y_rk)
    p = 4  # Порядок метода Адамса
    table = PrettyTable()
    table.field_names = ["i", "x_i", "y_i", "f(x_i, y_i)", "y_i+1"]
    i = 3
    iterations = 0
    while x[-1] < xn:
        x_new = x[-1] + h
        if x_new > xn:
            x_new = xn
        f_vals = [f(x[j], y[j]) for j in range(i, i-4, -1)]
        y_new = y[-1] + h * (55 * f_vals[0] - 59 * f_vals[1] + 37 * f_vals[2] - 9 * f_vals[3]) / 24
        y_half_step = rk4_step(f, x[-1], y[-1], h / 2)
        y_half = rk4_step(f, x[-1] + h / 2, y_half_step, h / 2)
        
        R = np.abs(y_new - y_half) / (2**p - 1)
        
        if R <= E:
            x.append(x_new)
            y.append(y_new)
            table.add_row([i, f"{x[-1]:.4f}", f"{y[-1]:.4f}", f"{f(x[-1], y[-1]):.4f}", f"{y_new:.4f}"])
            i += 1
        else:
            h /= 2
        iterations += 1

    print(table)
    return np.array(x), np.array(y), iterations

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
# 0 0.1
# 2
# 0.1
# 1e-6

print("3 - y' = 3 * x ** 2 * y + x^2 * exp(x^3)")
# 0 1
# 2
# 0.01
# 1e-6


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

x, y, iterations = method(f, x0, y0, xn, h, E)
y_exact_values = exact_solution_solve_ivp(f, x0, y0, xn, x)

E_max = max(np.abs(y_exact_values - y))
print(f"\n{method_name}:")

table = PrettyTable()
table.field_names = ["x", "y (численное)", "y (точное)", "delta"]
for x_i, y_i, y_exact_i in zip(x, y, y_exact_values):
    table.add_row([f"{x_i:.4f}", f"{y_i:.4f}", f"{y_exact_i:.4f}", f"{np.abs(y_i - y_exact_i):.4f}"])

print(table)
print(f"Оценка точности: E_max = {E_max}")
print(f"Количество итераций: {iterations}")

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