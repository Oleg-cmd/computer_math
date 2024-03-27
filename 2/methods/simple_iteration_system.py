import numpy as np


def simple_iteration_system(x0, y0, system, epsilon=1e-5, max_iterations=1000):
    x = x0
    y = y0

    iteration_table = []
    for iteration in range(max_iterations):
        x_prev = x  # Сохраняем предыдущее значение x для проверки изменения
        x = system[0](y)
        y_new = system[1](x)
        iteration_table.append([iteration + 1, x, y, y_new])

        # Проверка сходимости по изменению y и условной проверке изменения x
        if np.abs(y - y_new) < epsilon and np.abs(x - x_prev) < epsilon:
            return (x, y_new), iteration_table

        y = y_new
    return None, iteration_table


def print_iteration_system_table(iteration_table):
    # Вывод заголовка таблицы
    print("{:<10} {:<15} {:<15} {:<15}".format("Iteration", "x", "y (old)", "y (new)"))
    # Вывод данных об итерациях
    for row in iteration_table:
        print(
            "{:<10d} {:<15.6f} {:<15.6f} {:<15.6f}".format(
                row[0], row[1], row[2], row[3]
            )
        )
