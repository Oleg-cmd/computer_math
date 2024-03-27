import inspect
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp


# Считает количество интервалов
def calculate_intervals(a, b):
    step = abs(b - a) / 10

    if step > 1:
        step = 1

    counter = 0
    for x in np.arange(a, b, step):
        counter += 1

    return counter


# Функция, разбивающая отрезок на интервалы и вызывая соответсвующий метод и выводя результаты (можно задать максимальное количество интервалов)
def work(work_method, print_method, f, f_prime, f_phi, a, b, tol=1e-6, max_iter=100):
    roots = []
    num_intervals = calculate_intervals(a, b)
    print("Выбранное количество интервалов: ", num_intervals)
    for i in range(num_intervals):
        interval_start = a + (b - a) * i / num_intervals
        interval_end = a + (b - a) * (i + 1) / num_intervals

        if f(interval_start) * f(interval_end) < 0:
            print(
                "Найден подинтервал с потенциальным корнем:",
                interval_start,
                interval_end,
            )

            result, table = work_method(
                f, f_prime, f_phi, interval_start, interval_end, tol, max_iter
            )
            if result is not None:
                roots.append(result)
                print("\nТаблица итераций для данного корня:\n")
                print_method(table)
                print(f"\nКорень уравнения: {result}\n\n")

    if len(roots) == 0:
        print("Корни не найдены.")
        return None, []
    else:
        return roots, []


def get_lambda(f, f_prime, a, b):
    f_prime_a = f_prime(a)
    f_prime_b = f_prime(b)

    denominator = max(f_prime_a, f_prime_b)

    if denominator == 0:
        return 0

    if f_prime_a * f_prime_b > 0:
        return -1 / denominator

    else:
        return 1 / denominator


def get_function_string(f):
    source_lines = inspect.getsource(f)
    source_lines = source_lines.split("\n")[1:]
    function_string = "\n".join(source_lines)
    function_string = function_string.replace("return ", "", 1)
    return function_string


def draw_plot(a, b, f_string, f):
    x_values = np.linspace(a - 1, b + 1, 1000)
    # Вычисляем значения функции для каждого значения x
    y_values = f(x_values)

    # Построение графика
    plt.plot(x_values, y_values, label=f_string)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("График функции f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()


def safe_function_call(f, x_values):
    y_values = np.empty_like(x_values)
    for i, x in enumerate(x_values):
        try:
            y_values[i] = f(x)
        except ValueError:  # Или любое другое исключение, которое может возникнуть
            y_values[i] = np.nan
    return y_values


def draw_plot_second(a, b, f_string1, f1, f_string2, f2):
    x_values = np.linspace(a - 1, b + 1, 1000)
    # Вычисляем значения первой функции для каждого значения x
    y_values_f1 = f1(x_values)
    # Вычисляем значения второй функции для каждого значения x
    y_values_f2 = f2(x_values)

    # Построение графиков
    plt.plot(x_values, y_values_f1, label=f_string1, linestyle="-", color="blue")
    plt.plot(x_values, y_values_f2, label=f_string2, linestyle="--", color="red")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Графики функций")
    plt.grid(True)
    plt.legend()
    plt.show()


def print_system(eq_1, eq_2):
    print("\n" + eq_1)
    print(eq_2 + "\n")


def check_lipschitz_condition(phi_prime, variable, interval_start, interval_end):
    interval = np.linspace(
        interval_start, interval_end, 1000
    )  # Генерация точек интервала
    abs_phi_prime = sp.lambdify(variable, abs(phi_prime), "numpy")
    max_value = np.max(abs_phi_prime(interval))
    return max_value < 1


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_value(x):
    if is_float(x):
        return float(x)
    else:
        print("Введено неверное значение")
        exit()


def get_value_int(x):
    if is_float(x):
        return int(x)
    else:
        print("Введено неверное значение")
        exit()
