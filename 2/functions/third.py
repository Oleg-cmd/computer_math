from utils import get_lambda


def f(x):
    return -1.38 * (x**3) - (5.42 * (x**2)) + 2.57 * x + 10.95


def f_prime(x):
    return -4.14 * (x**2) - 10.84 * x + 2.57


# phi(x) = x + lambda*f(x)
def phi(x, a, b):
    return x + get_lambda(f, f_prime, a, b) * f(x)
