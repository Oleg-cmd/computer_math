import numpy as np
import sympy as sp
import math

# Define the variable and function for symbolic integration
x = sp.symbols("x")
f = 3 * x**3 - 2 * x**2 - 7 * x - 8


def check_convergence(func, a, b):
    try:
        # Попытка символьного интегрирования для проверки сходимости
        x = sp.symbols("x")
        f = func(x)
        integral_result = sp.integrate(f, (x, a, b))
        # print(integral_result)
        if integral_result.is_infinite:
            print("Интеграл не существует")
            return False
        else:
            print("Интеграл, вероятно, сходится.")
            return True
    except Exception as e:
        print("Ошибка при проверке сходимости:", e)
        return False


# Define a list of functions to choose from
def function_1(x):
    return 3 * x**3 - 2 * x**2 - 7 * x - 8


def function_2(x):
    return x**2 - 4 * x + 4


def function_3(x):
    return np.sin(x)


def function_4(x):
    return math.e ** (-x)


def general_transform(func, a, b, method, n):
    if np.isinf(b):
        # Замена переменной x = 1/t - 1 преобразует [0, inf) в [1, 0)
        transformed_func = lambda t: np.where(t == 0, 0, func(1 / t - 1) * (1 / t**2))
        a_new, b_new = 1, 0 if a == 0 else 1 / a
        result = method(transformed_func, b_new, a_new, n)
    elif np.isinf(a):
        # Замена переменной x = 1/t + 1 преобразует (-inf, 0] в [1, 0)
        transformed_func = lambda t: np.where(t == 0, 0, func(1 / t + 1) * (1 / t**2))
        a_new, b_new = 1, 0 if b == 0 else 1 / b
        result = method(transformed_func, a_new, b_new, n)
    else:
        result = method(func, a, b, n)
    return result


# List of functions
functions = [function_1, function_2, function_3, function_4]


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


# Define a mapping from user input to the actual functions
methods = {
    "left": left_rectangle_rule,
    "right": right_rectangle_rule,
    "midpoint": midpoint_rule,
    "trapezoid": trapezoidal_rule,
    "simpson": simpsons_rule,
}


# User interaction to choose function and method
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


func_choice = int(input("Function number (1-4): "))
method_choice = input("Method (left, right, midpoint, trapezoid, simpson): ")
a = input("Lower limit of integration (type 'inf' for infinity): ")
b = input("Upper limit of integration (type 'inf' for infinity): ")
n = int(input("Initial number of subintervals (n): "))

# Handling infinite limits

a = parse_input(a)
b = parse_input(b)


if check_convergence(functions[func_choice - 1], a, b):
    integral_value = general_transform(
        functions[func_choice - 1], a, b, methods[method_choice], n
    )
    print(f"The approximate value of the integral is {integral_value}")
    x = sp.symbols("x")
    f = functions[func_choice - 1](x)
    exact_integral = sp.integrate(f, (x, a, b))
    print("The exact value of the integral is:")
    print(exact_integral.evalf())
