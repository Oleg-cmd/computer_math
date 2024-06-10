import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, odeint
from prettytable import PrettyTable

def f1(x, y):
    return x ** 2 + x - 2 * y

def f1_ac(x):
    return x ** 2 / 2

def f2(x, y):
    return 2 * x - y + x ** 2

def f2_ac(x):
    return x ** 2

def f3(x, y):
    if x != 0:
        return 5 * x ** 2 - 2 * y / x
    return 0

def f3_ac(x):
    return x ** 3

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

def euler_method(f, x0, y0, x_nodes, h, E):
    p = 1
    x = [x0]
    y = [y0]
    table_x = PrettyTable()
    table_x.field_names = ["Узел", "Шаг", "F_h", "F_h/2", "R", "E"]
    iterations = 0

    for node in x_nodes[1:]:
        while True:
            y_new = euler_step(f, x[-1], y[-1], h)
            y_half_1 = euler_step(f, x[-1], y[-1], h / 2)
            y_half_2 = euler_step(f, x[-1] + h / 2, y_half_1, h / 2)
            R = np.abs(y_new - y_half_2) / (2**p - 1)
            table_x.add_row([node, f"{h:.4f}", f"{y_half_1:.4f}", f"{y_half_2:.4f}", f"{R:.4f}", f"{E:.4f}"])
            if R <= E:
                x.append(node)
                y.append(y_new)
                break
            else:
                h /= 2
        iterations += 1

    print(table_x)
    return np.array(x), np.array(y), iterations

def improved_euler_method(f, x0, y0, x_nodes, h, E):
    p = 2
    x = [x0]
    y = [y0]
    table_x = PrettyTable()
    table_x.field_names = ["Узел", "Шаг", "F_h", "F_h/2", "R", "E"]
    iterations = 0

    for node in x_nodes[1:]:
        while True:
            y_new = improved_euler_step(f, x[-1], y[-1], h)
            y_half_1 = improved_euler_step(f, x[-1], y[-1], h / 2)
            y_half_2 = improved_euler_step(f, x[-1] + h / 2, y_half_1, h / 2)
            R = np.abs(y_new - y_half_2) / (2**p - 1)
            table_x.add_row([node, f"{h:.4f}", f"{y_half_1:.4f}", f"{y_half_2:.4f}", f"{R:.4f}", f"{E:.4f}"])
            if R <= E:
                x.append(node)
                y.append(y_new)
                break
            else:
                h /= 2
        iterations += 1

    print(table_x)
    return np.array(x), np.array(y), iterations

def rk4_method(f, x0, y0, x_nodes, h, E):
    p = 4
    x = [x0]
    y = [y0]
    table_x = PrettyTable()
    table_x.field_names = ["Узел", "Шаг", "F_h", "F_h/2", "R", "E"]
    iterations = 0

    for node in x_nodes[1:]:
        while True:
            y_new = rk4_step(f, x[-1], y[-1], h)
            y_half_step = rk4_step(f, x[-1], y[-1], h / 2)
            y_half = rk4_step(f, x[-1] + h / 2, y_half_step, h / 2)
            R = np.abs(y_new - y_half) / (2**p - 1)
            table_x.add_row([node, f"{h:.4f}", f"{y_new:.4f}", f"{y_half:.4f}", f"{R:.4f}", f"{E:.4f}"])
            if R <= E:
                x.append(node)
                y.append(y_new)
                break
            else:
                h /= 2
        iterations += 1

    print(table_x)
    return np.array(x), np.array(y), iterations

def adams_method(f, x0, y0, x_nodes, h, E):
    p = 4
    x_rk, y_rk, _ = rk4_method(f, x0, y0, x_nodes[:4], h, E)
    x = list(x_rk)
    y = list(y_rk)
    table_x = PrettyTable()
    table_x.field_names = ["Узел", "Шаг", "F_h", "Точное значение", "R", "E"]
    iterations = 0

    for node in x_nodes[4:]:
        while True:
            f_vals = [f(x[j], y[j]) for j in range(-1, -5, -1)]
            y_new = y[-1] + h * (55 * f_vals[0] - 59 * f_vals[1] + 37 * f_vals[2] - 9 * f_vals[3]) / 24
            y_exact = f_ac(node)
            R = np.abs(y_new - y_exact)
            
            table_x.add_row([node, f"{h:.4f}", f"{y_new:.4f}", f"{y_exact:.4f}", f"{R:.4f}", f"{E:.4f}"])
            
            if R <= E:
                x.append(node)
                y.append(y_new)
                break
            else:
                h /= 2
        iterations += 1

    print(table_x)
    return np.array(x), np.array(y), iterations

print("Выберите функцию для решения:")
print("1 - y' = x ** 2 + x - 2 * y")
print("2 - y' = 2 * x - y + x ** 2")
print("3 - y' = 5 * x ** 2 - 2 * y / x")

func_id = input("Введите номер функции: ")

if func_id == "1":
    f = f1
    f_ac = f1_ac
elif func_id == "2":
    f = f2
    f_ac = f1_ac
elif func_id == "3":
    f = f3
    f_ac = f1_ac
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
    
    
def exact_solution_solve_ivp(x_nodes, f_acc):
    return np.array([f_ac(x) for x in x_nodes])
    

# Считаем узлы для всех
x_nodes = np.arange(x0, xn + h, h)

print(f"\n{method_name}:")
x, y, iterations = method(f, x0, y0, x_nodes, h, E)
y_exact_values = exact_solution_solve_ivp(x_nodes, f_ac)


jls_extract_var = print
jls_extract_var("\nСравнение с точными значениями")

table = PrettyTable()
table.field_names = ["x", "y (численное)", "y (точное)", "delta"]
for x_i, y_i, y_exact_i in zip(x, y, y_exact_values):
    table.add_row([f"{x_i:.4f}", f"{y_i:.4f}", f"{y_exact_i:.4f}", f"{np.abs(y_i - y_exact_i):.4f}"])

print(table)
if method_id == "4":
    E_max = max(np.abs(y_exact_values - y))
    print(f"Оценка точности: E_max = {E_max:.4f}")
print(f"Количество итераций: {iterations}")

plt.figure(figsize=(10, 6))
plt.plot(x, y, label=method_name)
y_exact_dense = exact_solution_solve_ivp(np.linspace(x0, xn, 100), f_ac)
plt.plot(np.linspace(x0, xn, 100), y_exact_dense, label="Точное решение", linestyle="--")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()