import numpy as np
import sympy as sp

# Define the variable and function for symbolic integration
x = sp.symbols("x")
f = 3 * x**3 - 2 * x**2 - 7 * x - 8

# Perform the exact integration


def runge_rule(method, func, a, b, n, p):
    I_n = method(func, a, b, n)
    I_2n = method(func, a, b, 2 * n)
    E_n = abs(I_2n - I_n) / (2**p - 1)
    return I_n, E_n


# Define a list of functions to choose from
def function_1(x):
    return 3 * x**3 - 2 * x**2 - 7 * x - 8


def function_2(x):
    # Replace with another function
    return x**2 - 4 * x + 4


def function_3(x):
    # Replace with another function
    return np.sin(x)


# List of functions
functions = [function_1, function_2, function_3]


# Midpoint rule
def midpoint_rule(func, a, b, n):
    h = (b - a) / n
    x = np.linspace(a + h / 2, b - h / 2, n)
    y = func(x)
    return h * np.sum(y)


# Trapezoidal rule
def trapezoidal_rule(func, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    return h * (np.sum(y) - (y[0] + y[-1]) / 2)


# Simpson's rule
def simpsons_rule(func, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-2:2]))


# Left rectangle rule


def left_rectangle_rule(func, a, b, n):

    h = (b - a) / n

    x = np.linspace(a, b - h, n)

    y = func(x)

    return h * np.sum(y)


# Right rectangle rule


def right_rectangle_rule(func, a, b, n):

    h = (b - a) / n

    x = np.linspace(a + h, b, n)

    y = func(x)

    return h * np.sum(y)


# Define a mapping from user input to the actual functions
methods = {
    "left": left_rectangle_rule,
    "right": right_rectangle_rule,
    "midpoint": midpoint_rule,
    "trapezoid": trapezoidal_rule,
    "simpson": simpsons_rule,
}

# User interaction to choose function and method
# print("Choose a function:")
# for idx, func in enumerate(functions, 1):
#     print(f"{idx}. {func.__name__}")
func_choice = int(input("Function number (1-3): "))
method_choice = input("Method (left, right, midpoint, trapezoid, simpson): ")
a = float(input("Lower limit of integration: "))
b = float(input("Upper limit of integration: "))
n = int(input("Initial number of subintervals (n): "))

# Calculate and print the integral using the chosen method
integral_value = methods[method_choice](functions[func_choice - 1], a, b, n)
print(f"The approximate value of the integral is {integral_value}")

# Perform the exact integration using sympy
x = sp.symbols("x")
f = functions[func_choice - 1](x)
exact_integral = sp.integrate(f, (x, a, b))
print("The exact value of the integral is:")
print(exact_integral.evalf())
