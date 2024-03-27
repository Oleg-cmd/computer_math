from utils import get_lambda


def f(x):
    return 3 * (x**3) + 1.7 * (x**2) - 15.42 * x + 6.89


def f_prime(x):
    return 9 * (x**2) + 3.4 * x - 15.42


# phi(x) = x + lambda*f(x)
def phi(x, a, b):
    return x + get_lambda(f, f_prime, a, b) * f(x)
