import numpy as np
import sympy as sp
import math
import inspect
import matplotlib.pyplot as plt

# Define the variable and function for symbolic integration
x = sp.symbols("x")
f = 3 * x**3 - 2 * x**2 - 7 * x - 8


def check_convergence(func, a, b):
    # Определяем функцию в символьной форме
    f_sym = func(x)

    # Проверяем на несобственный интеграл с бесконечным нижним пределом
    if a == -sp.oo:
        lim_a = sp.limit(f_sym, x, a)
        if lim_a.is_infinite or (lim_a.has(sp.oo, -sp.oo)):
            print("Интеграл расходится из-за особенности в нижнем пределе.")
            return False

    # Проверяем на несобственный интеграл с бесконечным верхним пределом
    if b == sp.oo:
        lim_b = sp.limit(f_sym, x, b)
        if lim_b.is_infinite or (lim_b.has(sp.oo, -sp.oo)):
            print("Интеграл расходится из-за особенности в верхнем пределе.")
            return False

    # Для интегралов с конечными пределами пробуем вычислить интеграл
    if a != -sp.oo and b != sp.oo:
        try:
            integral_result = sp.integrate(f_sym, (x, a, b))
            if integral_result.has(sp.oo, -sp.oo):
                print("Интеграл расходится.")
                return False
            else:

                print("Интеграл сходится.")
                return True
        except Exception as e:
            print("Не удалось вычислить интеграл:", e)
            return False

    # Если ни один из пределов не бесконечный, проверяем общую сходимость интеграла
    try:
        integral_result = sp.integrate(f_sym, (x, a, b))
        if integral_result.is_infinite:
            print("Интеграл расходится.")
            return False
        else:
            print("Интеграл сходится.")
            return True
    except Exception as e:
        print("Ошибка при проверке сходимости:", e)
        return False


def runge_rule(I_h, I_h2, k):
    return I_h2 + (I_h2 - I_h) / (2**k - 1)


def function_1(x):
    return 3 * x**3 - 2 * x**2 - 7 * x - 8


def function_2(x):
    return x**2 - 4 * x + 4


def function_3(x):

    if isinstance(x, sp.Symbol):
        return sp.sin(x)
    else:
        return np.sin(x)


def function_4(x):

    # -inf - 0
    return math.e ** (-x)


def function_5(x):

    # 0 - 1
    return 1 / x


def general_transform(func, a, b, method_name, epsilon):
    method = methods[method_name]
    k = k_values[method_name]

    n = 4
    refined_result = 0

    if b == sp.oo or a == -sp.oo:
        # Для несобственных интегралов начинаем с небольшого интервала и увеличиваем его
        b_temp = 1 if b == sp.oo else b
        a_temp = -1 if a == -sp.oo else a
        I_prev = method(func, a_temp, b_temp, n)
        I_curr = I_prev + 1  # Просто чтобы начать цикл
        while abs(I_curr - I_prev) > epsilon:
            n *= 2
            b_temp *= 2 if b == sp.oo else b
            a_temp *= 2 if a == -sp.oo else a
            I_prev = I_curr
            I_curr = method(func, a_temp, b_temp, n)
            refined_result = runge_rule(I_prev, I_curr, k)

    else:
        # Обработка конечных пределов
        I_prev = method(func, a, b, n)
        I_curr = method(func, a, b, n * 2)
        while abs(I_curr - I_prev) > epsilon:
            n *= 2
            I_prev = I_curr
            I_curr = method(func, a, b, n * 2)
        refined_result = runge_rule(I_prev, I_curr, k)

    print("N: ", n)
    return refined_result


def derivative(func, x_val, order):
    x = sp.symbols("x")
    func_sympy = func(x)
    derivative_func = sp.diff(func_sympy, x, order)
    derivative_val = derivative_func.subs(x, x_val).evalf()
    return abs(derivative_val)


def calculate_steps(method_name, a, b, epsilon, k):
    x = sp.symbols("x")
    if a == sp.oo or b == sp.oo:
        raise ValueError("Cannot use calculate_steps with infinite limits.")

    if method_name == "simpsons_rule":
        # Для метода Симпсона используем 4-ю производную
        fourth_derivative_max = max(
            [derivative(function_1, x_val, 4) for x_val in np.linspace(a, b, 100)]
        )
        h = np.power(180 * epsilon / ((b - a) * fourth_derivative_max), 1 / 4)
    elif method_name == "trapezoidal_rule":
        # Для метода трапеций используем 2-ю производную
        second_derivative_max = max(
            [derivative(function_1, x_val, 2) for x_val in np.linspace(a, b, 100)]
        )
        h = np.sqrt(12 * epsilon / ((b - a) * second_derivative_max))
    else:
        # Для остальных методов используем базовую оценку
        h = np.power(epsilon, 1 / k)

    n = int(np.ceil((b - a) / h))
    return max(n, 1)


# List of functions
functions = [function_1, function_2, function_3, function_4, function_5]


# Midpoint rule
def midpoint_rule(func, a, b, n):
    h = (b - a) / n
    total = 0
    for i in range(n):
        x = a + h / 2 + i * h
        total += func(x)
    return h * total


# Trapezoidal rule
def trapezoidal_rule(func, a, b, n):
    h = (b - a) / n
    total = (func(a) + func(b)) / 2.0
    for i in range(1, n):
        x = a + i * h
        total += func(x)
    return h * total


# Simpson's rule
def simpsons_rule(func, a, b, n):
    if n % 2 == 1:  # Симпсон работает только для четного количества сегментов
        n += 1
    h = (b - a) / n
    total = func(a) + func(b)
    for i in range(1, n, 2):
        x = a + i * h
        total += 4 * func(x)
    for i in range(2, n - 1, 2):
        x = a + i * h
        total += 2 * func(x)
    return h * total / 3


# Left rectangle rule
def left_rectangle_rule(func, a, b, n):
    h = (b - a) / n
    total = 0
    for i in range(n):
        x = a + i * h
        total += func(x)
    return h * total


# Right rectangle rule
def right_rectangle_rule(func, a, b, n):
    h = (b - a) / n
    total = 0
    for i in range(n):
        x = a + h + i * h
        total += func(x)
    return h * total


methods = {
    "left": left_rectangle_rule,
    "right": right_rectangle_rule,
    "midpoint": midpoint_rule,
    "trapezoid": trapezoidal_rule,
    "simpson": simpsons_rule,
}

k_values = {
    "left": 1,
    "right": 1,
    "midpoint": 1,
    "trapezoid": 2,
    "simpson": 4,
}


def get_function_string(f):
    source_lines = inspect.getsource(f)
    source_lines = source_lines.split("\n")[1:]
    function_string = "\n".join(source_lines)
    function_string = function_string.replace("return ", "", 1)
    return function_string


def parse_input(value):
    if value == "inf":
        return float("inf")
    elif value == "-inf":
        return float("-inf")
    else:
        try:
            return float(value)
        except ValueError:
            raise ValueError("Invalid input for limit of integration")


print("Welcome")
print(
    get_function_string(function_1),
    get_function_string(function_2),
    get_function_string(function_3),
    get_function_string(function_4),
    get_function_string(function_5),
)

precision = 0.01

func_choice = int(input("Function number (1-5): "))
method_choice = input("Method (left, right, midpoint, trapezoid, simpson): ")
a = input("Lower limit of integration (type 'inf' for infinity): ")
b = input("Upper limit of integration (type 'inf' for infinity): ")
precision = float(input("Desired precision (e.g., 0.01): "))
n = 4


# Handling infinite limits
a = parse_input(a)
b = parse_input(b)


def plot_function(func, a, b, epsilon, method_name):
    # Create an array of x values from a to b
    k = k_values[method_name]
    if a == -sp.oo:
        a = -10  # Example finite range start for plotting
    if b == sp.oo:
        b = 10

    n = calculate_steps(methods[method_name], a, b, epsilon, k)
    x_values = np.linspace(a, b, 300)
    y_values = func(x_values)

    plt.figure(figsize=(10, 5))
    plt.plot(x_values, y_values, label=f"Function: {func.__name__}", color="blue")

    if method_name in ["midpoint", "left", "right", "trapezoid", "simpson"]:
        h = (b - a) / n
        x_rects = np.linspace(a, b, n + 1)
        if method_name == "midpoint":
            x_rects = x_rects[:-1] + h / 2  # Adjust for midpoint rule
        y_rects = func(x_rects)

        for i in range(n):
            plt.bar(
                x_rects[i],
                y_rects[i],
                width=h,
                alpha=0.2,
                align="edge",
                edgecolor="red",
            )

        plt.title(f"Plot of the function with {n} steps using {method_name}")
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.show()


if check_convergence(functions[func_choice - 1], a, b):
    integral_value = general_transform(
        functions[func_choice - 1], a, b, method_choice, precision
    )
    print(f"The approximate value of the integral is {integral_value}")
    x = sp.symbols("x")
    f = functions[func_choice - 1](x)
    exact_integral = sp.integrate(f, (x, a, b))
    print("The exact value of the integral is:")
    print(exact_integral.evalf())
    plot_function(functions[func_choice - 1], a, b, precision, method_choice)
