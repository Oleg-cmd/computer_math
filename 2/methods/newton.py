# Метод Ньютона
def newton_method(f, f_prime, f_phi, a, b, tol, max_iter):
    x = (a + b) / 2
    iteration_table = []
    for n in range(1, max_iter + 1):
        f_x = f(x)
        f_prime_x = f_prime(x)
        iteration_table.append([n, x, f_x, f_prime_x])

        if abs(f_x) < tol:
            return x, iteration_table

        delta_x = f_x / f_prime_x
        x -= delta_x

    return None, iteration_table


def print_newton_iteration_table(iteration_table):
    # Вывод заголовка таблицы
    print("{:<10} {:<10} {:<10} {:<10}".format("N", "x", "f(x)", "f'(x)"))
    # Вывод данных об итерациях
    for row in iteration_table:
        print(
            "{:<10d} {:<10.6f} {:<10.6f} {:<10.6f}".format(
                row[0], row[1], row[2], row[3]
            )
        )
