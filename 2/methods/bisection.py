def bisection_method(f, f_prime, f_phi, a, b, tol, max_iter):
    iteration_table = []
    for n in range(1, max_iter + 1):
        c = (a + b) / 2
        f_c = f(c)
        iteration_table.append([n, a, b, c, f(a), f_c, abs(b - a)])

        if abs(f_c) < tol:
            return c, iteration_table

        if f(a) * f_c < 0:
            b = c
        else:
            a = c

    return None, iteration_table


def print_bisection_iteration_table(iteration_table):
    # Вывод заголовка таблицы
    print(
        "{:<10} {:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "N", "a", "b", "c", "f(a)", "f(c)", "|b - a|"
        )
    )
    # Вывод данных об итерациях
    for row in iteration_table:
        print(
            "{:<10d} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}".format(
                row[0], row[1], row[2], row[3], row[4], row[5], row[6]
            )
        )
