from sympy import diff, symbols
import numpy as np

x = symbols("x")


def check_convergence(f, phi, a, b):
    # Оставим вашу проверку сходимости без изменений
    derivative_phi = diff(phi(x, a, b), x)
    derivative_values = [derivative_phi.subs(x, a), derivative_phi.subs(x, b)]
    print(derivative_values[0], derivative_values[1])

    max_derivative = max(abs(value) for value in derivative_values)
    if max_derivative >= 1:
        return False
    else:
        return True


def simple_iteration_method(f, f_prime, phi, a, b, tol, max_iter):
    iteration_table = []

    if not check_convergence(f, phi, a, b):
        # Если не сходится на начальном интервале, разделим его пополам
        while not check_convergence(f, phi, a, b):
            mid = (a + b) / 2
            if check_convergence(f, phi, a, mid):
                b = mid
            else:
                a = mid

    x_prev = (a + b) / 2  # Начальное приближение

    for n in range(1, max_iter + 1):
        x_next = phi(x_prev, a, b)
        f_x_prev = f(x_prev)
        iteration_table.append(
            [n, x_prev, x_next, f_x_prev, f(x_next), abs(x_next - x_prev)]
        )

        if abs(x_next - x_prev) < tol:
            if np.real(x_next) >= a and np.real(x_next) <= b:
                return x_next, iteration_table

        x_prev = x_next

    return None, iteration_table


def print_simple_iteration_table(iteration_table):
    # Вывод заголовка таблицы
    print(
        "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "n", "x_prev", "x_next", "f(x_prev)", "f(x_next)", "|x_next - x_prev|"
        )
    )
    # Вывод данных об итерациях
    for row in iteration_table:
        print(
            "{:<10d} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}".format(
                row[0], row[1], row[2], row[3], row[4], row[5]
            )
        )
