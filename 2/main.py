import sympy as sp
import numpy as np

from utils import get_function_string, work, draw_plot, print_system, draw_plot_second

from functions.first import f as f_1, f_prime as f_1_prime, phi as phi_1
from functions.second import f as f_2, f_prime as f_2_prime, phi as phi_2, f_np
from functions.third import f as f_3, f_prime as f_3_prime, phi as phi_3

from methods.bisection import bisection_method, print_bisection_iteration_table
from methods.newton import newton_method, print_newton_iteration_table
from methods.simple_iteration import (
    simple_iteration_method,
    print_simple_iteration_table,
)


from functions.system_first import (
    system,
    eq_first,
    eq_second,
    condition,
    first_np as np_01,
    sec_np as np_02,
)

from functions.system_second import (
    system as system_2,
    eq_first as eq_first_2,
    eq_second as eq_second_2,
    condition as condition_2,
    first_np as np_11,
    sec_np as np_12,
)

from methods.simple_iteration_system import (
    simple_iteration_system,
    print_iteration_system_table,
)


def main():
    print("Добро пожаловать!")
    choice = input(
        "Вы хотите решить нелинейное уравнение (введите 1) или систему нелинейных уравнений (введите 2): "
    )

    a, b = 0, 1
    tol = 0.001
    max_iter = 100

    if choice == "1":
        print("Доступные уравнения:")

        print("1) " + get_function_string(f_1))
        print("2) " + get_function_string(f_2))
        print("3) " + get_function_string(f_3))

        eq_case = input("Выберите уравнение (1, 2, 3): ")

        print("Выберите метод решения нелинейного уравнения:")
        print("1. Метод половинного деления")
        print("2. Метод Ньютона")
        print("3. Метод простой итерации")

        method_choice = input("Введите номер метода (1, 2 или 3): ")

        if eq_case == "1":
            f = f_1
            f_prime = f_1_prime
            f_phi = phi_1
            fnp = f
        elif eq_case == "2":
            f = f_2
            f_prime = f_2_prime
            f_phi = phi_2
            fnp = f_np
        elif eq_case == "3":
            f = f_3
            f_prime = f_3_prime
            f_phi = phi_3
            fnp = f
        else:
            print("Некорректный ввод")
            exit()

        if method_choice == "1":
            method = bisection_method
            print_method = print_bisection_iteration_table
        elif method_choice == "2":
            method = newton_method
            print_method = print_newton_iteration_table
        elif method_choice == "3":
            method = simple_iteration_method
            print_method = print_simple_iteration_table
        else:
            print("Некорректный ввод")
            exit()

        work(method, print_method, f, f_prime, f_phi, a, b, tol, max_iter)
        draw_plot(a, b, get_function_string(f), fnp)

    elif choice == "2":
        print("Выберите систему (1 или 2)")

        print("1.")
        print_system(eq_first, eq_second)

        print("2.")
        print_system(eq_first_2, eq_second_2)

        system_choise = input()

        if system_choise == "1":
            condition(a, b)
            result, table = simple_iteration_system(a, b, system, tol, max_iter)
            print_iteration_system_table(table)
            print("\nОтвет: ", result)
            draw_plot_second(-10, 10, eq_first, np_01, eq_second, np_02)

        elif system_choise == "2":
            condition(a, b)
            result, table = simple_iteration_system(a, b, system_2, tol, max_iter)
            print_iteration_system_table(table)
            print("\nОтвет: ", result)
            draw_plot_second(-10, 10, eq_first_2, np_11, eq_second_2, np_12)
        else:
            print("Некорректный ввод")
            exit()

    else:
        print("Некорректный ввод")
        exit()


if __name__ == "__main__":
    main()
