def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


# Приветственное сообщение и выбор метода решения нелинейного уравнения
print("Добро пожаловать!")
eq_or_system = input(
    "Вы хотите решить нелинейное уравнение (введите 1) или систему нелинейных уравнений (введите 2): "
)
if (eq_or_system != "1") and (eq_or_system != "2"):
    print("Неверный ввод.")
    exit()
else:
    eq_or_system = int(eq_or_system)


if eq_or_system == 1:
    print("Выберите уравнение:")
    print("1) ", f1_text)
    print("2) ", f2_text)
    print("3) ", f3_text)

    eq = input("Введите номер уравнения (1, 2 или 3): ")
    if (eq != "1") and (eq != "2") and (eq != "3"):
        print("Неверный ввод номера уравнения.")
        exit()
    elif eq == "2":

        f = f2
        f_prime = f2_prime
        phi1 = phil2_1
        f1_text = f2_text
        fnp = f
    elif eq == "3":
        f = f3
        f_prime = f3_prime
        phi1 = phil3_1
        f1_text = f3_text
    else:
        fnp = f

    print("Выберите метод решения нелинейного уравнения:")
    print("1. Метод половинного деления")
    print("2. Метод Ньютона")
    print("3. Метод простой итерации")

    method_choice = input("Введите номер метода (1, 2 или 3): ")
    if (method_choice != "1") and (method_choice != "2") and (method_choice != "3"):
        print("Неверный ввод номера метода.")
        exit()

    # Запрос пользователю на задание интервала [a, b] и fault
    a = input("Введите значение a: ")
    b = input("Введите значение b: ")
    fault = input("Введите значение погрешности: ")

    wanna_find_all = input(
        "Хотите найти все корни на заданном интервале? (да/нет/y/n): "
    )
    if (
        wanna_find_all.lower() == "да"
        or wanna_find_all.lower() == "y"
        or wanna_find_all.lower() == "yes"
    ):
        wanna_find_all = True
    elif (
        wanna_find_all.lower() == "нет"
        or wanna_find_all.lower() == "n"
        or wanna_find_all.lower() == "no"
    ):
        wanna_find_all = False
    else:
        print("Неверный ввод")
        exit()

    max_iter = input(
        "Введите максимальное количество итераций (можете пропустить этот шаг): "
    )
    if not max_iter.isdigit() or max_iter == "":
        max_iter = 100
    else:
        max_iter = int(max_iter)

    max_roots = 200

    if not (is_float(a) and is_float(b) and is_float(fault)):
        print("Ошибка: Введены неверные значения. Пожалуйста, введите числа.")
        exit()

    a = float(a)
    b = float(b)
    fault = float(fault)

    a_copy = a
    b_copy = b

    if method_choice == "1":
        roots, _ = bisection_method(f, a, b, fault, max_iter)

        if roots is not None:
            print("Все корни подряд:")
            for root in roots:
                print(f"\nКорень уравнения: {root}\n\n\n")
        else:
            print("Корни или новые корни не были найдены")

    elif method_choice == "2":
        init_guesses = [a, b, (a + b) / 2, random.uniform(a, b), random.uniform(a, b)]
        # тоже по интервалам
        index = 0
        roots = []

        if wanna_find_all is False:
            index = 2

        while index < len(init_guesses):
            root, table = newton_method(
                f, f_prime, init_guesses[index], fault, max_iter
            )
            if root is not None:
                rounded_root = round(root, round(-math.log10(fault)))
                if rounded_root not in roots:
                    print("\nМетод Ньютона сошелся к корню")
                    print("\nТаблица итераций:\n")
                    print_newton_iteration_table(table)
                    print(f"\nКорень уравнения: {root}\n\n\n")
                    roots.append(rounded_root)
                    print(
                        f"Попытка найти другие корни c другим начальным приближением... P.s [{init_guesses[index]}]"
                    )
                else:
                    print("Корни или новые корни c данным приближением не были найдены")

                index += 1

                if wanna_find_all is False:
                    break
            else:
                print("Метод не сходится к корню.\n\n\n")
                break

    if method_choice == "3":
        num_intervals = calculate_intervals(a, b)
        roots = []

        for i in range(num_intervals):
            interval_start = a + (b - a) * i / num_intervals
            interval_end = a + (b - a) * (i + 1) / num_intervals

            # print(interval_start, interval_end)

            # Попробуйте первую функцию phi
            root, table = simple_iteration_method(
                f, phi1, interval_start, interval_end, fault, max_iter
            )
            if root is not None:
                if isinstance(root, complex):  # Если root комплексное число
                    rounded_real = round(root.real, round(-math.log10(fault)))
                    rounded_imag = round(root.imag, round(-math.log10(fault)))
                    rounded_root = rounded_real + rounded_imag * 1j
                else:
                    rounded_root = round(root, round(-math.log10(fault)))
                if rounded_root not in roots:

                    print("\nМетод простой итерации сошелся к корню.")

                    print("\nТаблица итераций:\n")
                    print_simple_iteration_table(table)

                    print("\nНайденный корень:", root)
                    roots.append(rounded_root)

                    if wanna_find_all is False:
                        break

                else:
                    # print("Корень с данным начальным приближением уже найден.")
                    pass

            root, table = simple_iteration_method(
                f, phi2, interval_start, interval_end, fault, max_iter
            )
            if root is not None:
                if isinstance(root, complex):  # Если root комплексное число
                    rounded_real = round(root.real, round(-math.log10(fault)))
                    rounded_imag = round(root.imag, round(-math.log10(fault)))
                    rounded_root = rounded_real + rounded_imag * 1j
                else:
                    rounded_root = round(root, round(-math.log10(fault)))

                if rounded_root not in roots:
                    print("\nМетод простой итерации сошелся к корню.")

                    print("\nТаблица итераций:\n")
                    print_simple_iteration_table(table)

                    print("\nНайденный корень:", root)
                    roots.append(rounded_root)

                    if wanna_find_all is False:
                        break
                else:
                    # print("Корень с данным начальным приближением уже найден.")
                    pass
        if len(roots) == 0:
            print("Метод простой итерации не сходится к корню")
            pass

    # Вывод графика функции

    x_values = numpy.linspace(a - 1, b + 1, 1000)

    # Вычисляем значения функции для каждого значения x
    y_values = fnp(x_values)

    # Построение графика
    plt.plot(x_values, y_values, label=f1_text)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("График функции f(x)")
    plt.grid(True)
    plt.legend()
    plt.show()

else:
    equation1 = "cos(x)-1"
    equation2 = "0.8x^2 + 2y^2 = 1"

    equation3 = "sin(x+1) - 1.1x = 0.1"
    equation4 = "x^2 + y^2 = 1"

    print("Выберите cистему уравнений:")
    print("1)")
    print(equation1)
    print(equation2)
    print("2)")
    print(equation3)
    print(equation4)

    system = input("Введите номер системы (1, 2 или 3): ")
    if system != "1" and system != "2" and system != "3":
        print("Неверно выбранная система")
        exit()
    else:
        # Запрос пользователю на задание интервала [a, b] и fault
        a = input("Введите значение a: ")
        b = input("Введите значение b: ")
        fault = input("Введите значение погрешности: ")

        if not (is_float(a) and is_float(b) and is_float(fault)):
            print("Ошибка: Введены неверные значения. Пожалуйста, введите числа.")
            exit()

        a = float(a)
        b = float(b)
        fault = float(fault)

        max_iter = input(
            "Введите максимальное количество итераций (можете пропустить этот шаг): "
        )
        if not max_iter.isdigit() or max_iter == "":
            max_iter = 100
        else:
            max_iter = int(max_iter)

        def simple_iteration_method_system(
            equations, phi_x, phi_y, initial_guess, tol=1e-6, max_iter=100
        ):
            x_prev, y_prev = initial_guess[0], initial_guess[1]
            iteration_table = []

            for n in range(1, max_iter + 1):
                x_next = phi_x(x_prev, y_prev)
                y_next = phi_y(x_prev, y_prev)

                f_values = np.array(equations(x_next, y_next))
                iteration_table.append([n, x_prev, y_prev, *f_values])

                if np.all(np.abs(f_values) < tol):
                    print("\nМетод простых итераций сошелся к корню.")
                    return x_next, y_next, iteration_table

                x_prev, y_prev = x_next, y_next

            print("Достигнуто максимальное количество итераций.")
            return None, None, iteration_table

        def equations(x, y):
            equation1 = np.cos(x) - 1
            equation2 = 0.8 * x**2 + 2 * y**2 - 1
            return equation1, equation2

        def phi_x(x, y):
            return np.cos(x)

        def phi_y(x, y):
            return np.sqrt((1 - 0.8 * x**2) / 2)

        # Начальное приближение
        initial_guess = (a, b)

        root_x, root_y, table = simple_iteration_method_system(
            equations, phi_x, phi_y, initial_guess, fault, max_iter
        )
        if root_x is not None and root_y is not None:
            print("\nТаблица итераций:\n")
            print(
                "{:<10} {:<10} {:<10} {:<10} {:<10}".format(
                    "N", "x", "y", "f1(x, y)", "f2(x, y)"
                )
            )
            for row in table:
                print("{:<10d} {:<10.6f} {:<10.6f} {:<10.6f} {:<10.6f}".format(*row))
            print(f"\nКорни уравнения: x = {root_x}, y = {root_y}")
        else:
            print("Метод простых итераций не сошелся к корню.")

        if system == "2":
            pass
