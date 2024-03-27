from utils import get_lambda
from sympy import sin, cos
import numpy as np


def f(x):
    return 0.5 - sin(x + 0.5)


def f_np(x):
    return 0.5 - np.sin(x + 0.5)


def f_prime(x):
    return -cos(x + 0.5)


# phi(x) = x + lambda*f(x)
def phi(x, a, b):
    return x + get_lambda(f, f_prime, a, b) * f(x)
