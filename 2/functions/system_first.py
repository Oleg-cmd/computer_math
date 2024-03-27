import numpy as np
import sympy as sp
from utils import check_lipschitz_condition

x, y = sp.symbols("x y")
phi1 = sp.cos(y)
phi2 = sp.sqrt((1 - 0.8 * x**2) / 2)

phi1_prime_y = sp.diff(phi1, y)
phi2_prime_x = sp.diff(phi2, x)


eq_first = "x=cos(y)"
eq_second = "0.8x^2 + 2y^2 = 1"


def first(y):
    return np.cos(y)


def second(x):
    expr = (1 - 0.8 * x**2) / 2

    if expr >= 0:
        return np.sqrt(expr)
    else:
        return np.sqrt(-expr)


def first_np(x):
    return np.arccos(x)


def sec_np(x):
    expr = (1 - 0.8 * x**2) / 2
    return np.sqrt(expr)


def condition(x0, y0):
    # Оценка производных в точке (x0, y0)
    condition_y = check_lipschitz_condition(phi1_prime_y, y, y0 - 0.1, y0 + 0.1)
    condition_x = check_lipschitz_condition(phi2_prime_x, x, x0 - 0.1, x0 + 0.1)

    if not (condition_y and condition_x):
        print("Условия Липшица не выполнены, сходимость не гарантирована.")
        return False
    else:
        print("Условия Липшица выполнены, можно продолжать с итерациями.")
        return True


system = [first, second]
