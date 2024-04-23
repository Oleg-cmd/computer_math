import numpy as np
import matplotlib.pyplot as plt
from math import log, exp, sqrt

FILE_IN = "input_2.txt"
FILE_OUT = "output.txt"


def solve_minor(matrix, i, j):
    """Найти минор элемента матрицы"""
    n = len(matrix)
    return [
        [matrix[row][col] for col in range(n) if col != j]
        for row in range(n)
        if row != i
    ]


def solve_det(matrix):
    """Найти определитель матрицы"""
    n = len(matrix)
    if n == 1:
        return matrix[0][0]
    det = 0
    sgn = 1
    for j in range(n):
        det += sgn * matrix[0][j] * solve_det(solve_minor(matrix, 0, j))
        sgn *= -1
    return det


def calc_s(dots, f):
    """Найти меру отклонения"""
    n = len(dots)
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]

    return sum([(f(x[i]) - y[i]) ** 2 for i in range(n)])


def calc_stdev(dots, f):
    """Найти среднеквадратичное отклонение"""
    n = len(dots)

    return sqrt(calc_s(dots, f) / n)


def lin_func(dots):
    """Линейная аппроксимация"""
    data = {}

    n = len(dots)
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]

    sx = sum(x)
    sx2 = sum([xi**2 for xi in x])
    sy = sum(y)
    sxy = sum([x[i] * y[i] for i in range(n)])

    d = solve_det([[sx2, sx], [sx, n]])
    d1 = solve_det([[sxy, sx], [sy, n]])
    d2 = solve_det([[sx2, sxy], [sx, sy]])

    try:
        a = d1 / d
        b = d2 / d
    except ZeroDivisionError:
        return None
    data["a"] = a
    data["b"] = b

    f = lambda z: a * z + b
    data["f"] = f

    data["str_f"] = "fi = a*x + b"

    data["s"] = calc_s(dots, f)

    data["stdev"] = calc_stdev(dots, f)

    return data


def sqrt_func(dots):
    """Квадратичная аппроксимация"""
    data = {}

    n = len(dots)
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]

    sx = sum(x)
    sx2 = sum([xi**2 for xi in x])
    sx3 = sum([xi**3 for xi in x])
    sx4 = sum([xi**4 for xi in x])
    sy = sum(y)
    sxy = sum([x[i] * y[i] for i in range(n)])
    sx2y = sum([(x[i] ** 2) * y[i] for i in range(n)])

    d = solve_det([[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]])
    d1 = solve_det([[sy, sx, sx2], [sxy, sx2, sx3], [sx2y, sx3, sx4]])
    d2 = solve_det([[n, sy, sx2], [sx, sxy, sx3], [sx2, sx2y, sx4]])
    d3 = solve_det([[n, sx, sy], [sx, sx2, sxy], [sx2, sx3, sx2y]])

    try:
        c = d1 / d
        b = d2 / d
        a = d3 / d
    except ZeroDivisionError:
        return None
    data["c"] = c
    data["b"] = b
    data["a"] = a

    f = lambda z: a * (z**2) + b * z + c
    data["f"] = f

    data["str_f"] = "fi = a*x^2 + b*x + c"

    data["s"] = calc_s(dots, f)

    data["stdev"] = calc_stdev(dots, f)

    return data


def cubic_func(dots):
    """Кубическая аппроксимация"""
    data = {}

    n = len(dots)
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]

    sx = sum(x)
    sx2 = sum(xi**2 for xi in x)
    sx3 = sum(xi**3 for xi in x)
    sx4 = sum(xi**4 for xi in x)
    sx5 = sum(xi**5 for xi in x)
    sx6 = sum(xi**6 for xi in x)
    sy = sum(y)
    sxy = sum(x[i] * y[i] for i in range(n))
    sx2y = sum((x[i] ** 2) * y[i] for i in range(n))
    sx3y = sum((x[i] ** 3) * y[i] for i in range(n))

    d = solve_det(
        [
            [n, sx, sx2, sx3],
            [sx, sx2, sx3, sx4],
            [sx2, sx3, sx4, sx5],
            [sx3, sx4, sx5, sx6],
        ]
    )

    d1 = solve_det(
        [
            [sy, sx, sx2, sx3],
            [sxy, sx2, sx3, sx4],
            [sx2y, sx3, sx4, sx5],
            [sx3y, sx4, sx5, sx6],
        ]
    )

    d2 = solve_det(
        [
            [n, sy, sx2, sx3],
            [sx, sxy, sx3, sx4],
            [sx2, sx2y, sx4, sx5],
            [sx3, sx3y, sx5, sx6],
        ]
    )
    d3 = solve_det(
        [
            [n, sx, sy, sx3],
            [sx, sx2, sxy, sx4],
            [sx2, sx3, sx2y, sx5],
            [sx3, sx4, sx3y, sx6],
        ]
    )
    d4 = solve_det(
        [
            [n, sx, sx2, sy],
            [sx, sx2, sx3, sxy],
            [sx2, sx3, sx4, sx2y],
            [sx3, sx4, sx5, sx3y],
        ]
    )

    try:
        a = d4 / d
        b = d3 / d
        c = d2 / d
        d = d1 / d

    except ZeroDivisionError:
        return None

    f = lambda z: a * z**3 + b * z**2 + c * z + d

    data["c"] = c
    data["b"] = b
    data["a"] = a
    data["d"] = d

    data["f"] = f
    data["str_f"] = "fi = a*x^3 + b*x^2 + c*x + d"

    data["s"] = calc_s(dots, f)
    data["stdev"] = calc_stdev(dots, f)
    # print(data)

    return data


def exp_func(dots):
    """Экспоненциальная аппроксимация"""
    data = {}
    valid_dots = [
        (x, y) for x, y in dots if y > 0
    ]  # Filter dots to include only valid ones

    if any(x <= 0 or y <= 0 for x, y in dots):
        print(
            "Invalid data points found; skipping ecsp function approximation. Экспоненциальная аппроксимация невозможна"
        )
        return None

    n = len(valid_dots)
    x = [dot[0] for dot in valid_dots]
    y = [dot[1] for dot in valid_dots]

    lin_y = [log(yi) for yi in y]
    lin_result = lin_func(list(zip(x, lin_y)))

    if lin_result is None:
        print("Linear approximation failed in exponential function.")
        return None

    a = exp(lin_result["b"])
    b = lin_result["a"]
    data["a"] = a
    data["b"] = b

    f = lambda z: a * exp(b * z)
    data["f"] = f
    data["str_f"] = "fi = a*e^(b*x)"

    data["s"] = calc_s(valid_dots, f)
    data["stdev"] = calc_stdev(valid_dots, f)
    return data


def log_func(dots):
    """Логарифмическая аппроксимация"""
    data = {}
    valid_dots = [(x, y) for x, y in dots if x > 0]

    if any(x <= 0 or y <= 0 for x, y in dots):
        print(
            "Invalid data points found; skipping log function approximation. Логорифмическая аппроксимация невозможна"
        )
        return None

    n = len(valid_dots)
    x = [dot[0] for dot in valid_dots]
    y = [dot[1] for dot in valid_dots]

    lin_x = [log(xi) for xi in x]
    lin_result = lin_func(list(zip(lin_x, y)))

    if lin_result is None:
        print("Linear approximation failed in logarithmic function.")
        return None

    a = lin_result["a"]
    b = lin_result["b"]
    data["a"] = a
    data["b"] = b

    f = lambda z: a * log(z) + b if z > 0 else 0
    data["f"] = f
    data["str_f"] = "fi = a*ln(x) + b"

    data["s"] = calc_s(valid_dots, f)
    data["stdev"] = calc_stdev(valid_dots, f)
    return data


def pow_func(dots):
    """Степенная аппроксимация"""
    data = {}
    # Check all dots at once if any x or y <= 0
    if any(x <= 0 or y <= 0 for x, y in dots):
        print(
            "Invalid data points found; skipping power function approximation. Степенная аппроксимация невозможна"
        )
        return None

    # All data points are valid if we reach here
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]

    # Apply logarithm to x and y
    lin_x = [log(xi) for xi in x]
    lin_y = [log(yi) for yi in y]

    # Perform linear regression on the logged values
    lin_result = lin_func(list(zip(lin_x, lin_y)))
    if lin_result is None:
        print("Linear approximation failed in power function.")
        return None

    # Exponentiate the intercept to get 'a' and use the slope as 'b'
    a = exp(lin_result["b"])
    b = lin_result["a"]

    # Define the power function
    f = lambda z: (
        a * (z**b) if z > 0 else float("inf")
    )  # Still safeguard against z <= 0

    data["a"] = a
    data["b"] = b
    data["f"] = f
    data["str_f"] = "fi = a*x^b"
    data["s"] = calc_s(dots, f)  # Calculate sum of squares
    data["stdev"] = calc_stdev(dots, f)  # Calculate standard deviation

    return data


def plot(x, y, plot_x, plot_y, labels, best_answers):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.scatter(x, y, color="red", label="Исходные точки")

    # Определяем стили и цвета для всех функций
    colors = [
        "blue",
        "green",
        "magenta",
        "cyan",
        "orange",
        "black",
        "brown",
        "purple",
        "pink",
        "olive",
    ]
    line_styles = ["--", "-.", ":"]
    color_index = 0
    style_index = 0

    best_labels = [
        ba["str_f"] for ba in best_answers
    ]  # Создаем список названий лучших функций

    for i, py in enumerate(plot_y):
        line_style = line_styles[style_index % len(line_styles)]
        color = colors[color_index % len(colors)]
        # Увеличиваем индексы для следующей функции
        color_index += 1
        style_index += 1
        if labels[i] in best_labels:
            # Для лучших функций используем более толстые линии
            ax.plot(
                plot_x,
                py,
                label=labels[i],
                linewidth=2.5,
                linestyle=line_style,
                color=color,
            )
        else:
            # Для остальных функций используем стандартную толщину линий
            ax.plot(
                plot_x,
                py,
                label=labels[i],
                linewidth=1.5,
                linestyle="-",
                color=color,
            )
    ax.legend()
    plt.show()


def getdata_file():
    """Получить данные из файла"""
    data = {"dots": []}

    with open(FILE_IN, "rt", encoding="UTF-8") as fin:
        try:
            for line in fin:
                current_dot = tuple(map(float, line.strip().split()))
                if len(current_dot) != 2:
                    raise ValueError
                data["dots"].append(current_dot)
            if len(data["dots"]) < 2:
                raise AttributeError
        except (ValueError, AttributeError):
            return None

    return data


def getdata_input():
    """Получить данные с клавиатуры"""
    data = {"dots": []}

    print("\nВводите координаты через пробел, каждая точка с новой строки.")
    print("Чтобы закончить, введите 'END'.")
    while True:
        try:
            current = input().strip()
            if current == "END":
                if len(data["dots"]) < 2:
                    raise AttributeError
                break
            current_dot = tuple(map(float, current.split()))
            if len(current_dot) != 2:
                raise ValueError
            data["dots"].append(current_dot)
        except ValueError:
            print("Введите точку повторно - координаты некорректны!")
        except AttributeError:
            print("Минимальное количество точек - 2!")

    return data


def pearson_correlation(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(xi**2 for xi in x)
    sum_y2 = sum(yi**2 for yi in y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))

    try:
        correlation = (n * sum_xy - sum_x * sum_y) / sqrt(
            (n * sum_x2 - sum_x**2) * (n * sum_y2 - sum_y**2)
        )
    except ZeroDivisionError:
        correlation = 0  # В случае ошибки деления на ноль возвращаем 0

    return correlation


def main():
    print("\tЛабораторная работа #4 (8)")
    print("\t  Аппроксимация функций")

    print("\nВзять исходные данные из файла (+) или ввести с клавиатуры (-)?")
    inchoice = input("Режим ввода: ")
    while (inchoice != "+") and (inchoice != "-"):
        print("Введите '+' или '-' для выбора способа ввода.")
        inchoice = input("Режим ввода: ")

    if inchoice == "+":
        data = getdata_file()
        if data is None:
            print("\nДанные в файле некорректны!")
            print("Режим ввода переключен на ручной.")
            data = getdata_input()
    else:
        data = getdata_input()

    answers = []
    temp_answers = [
        lin_func(data["dots"]),
        sqrt_func(data["dots"]),
        cubic_func(data["dots"]),
        exp_func(data["dots"]),
        log_func(data["dots"]),
        pow_func(data["dots"]),
    ]

    x = np.array([dot[0] for dot in data["dots"]])
    y = np.array([dot[1] for dot in data["dots"]])

    print(
        "\n\n%20s%20s%20s%20s%20s%20s%20s%20s"
        % (
            "Вид функции",
            "Ср. отклонение",
            "a",
            "b",
            "c",
            "d",
            "Корреляция",
            "Достоверность",
        )
    )
    print("-" * 180)
    for answer in temp_answers:
        if answer is not None:
            answers.append(answer)
            predicted_y = [answer["f"](xi) for xi in x]
            correlation = pearson_correlation(y, predicted_y)
            SS_res = np.sum((y - predicted_y) ** 2)
            SS_tot = np.sum((y - np.mean(y)) ** 2)
            R_squared = 1 - (SS_res / SS_tot)
            print(
                "%20s%20.5f%20.5f%20.5f%20.5f%20.5f%20.5f%20.5f"
                % (
                    answer["str_f"],
                    answer["stdev"],
                    answer.get("a", float("nan")),
                    answer.get("b", float("nan")),
                    answer.get("c", float("nan")),
                    answer.get("d", float("nan")),
                    correlation,
                    R_squared,
                )
            )

    min_stdev = min(answer["stdev"] for answer in answers)
    # Собираем все функции с минимальным отклонением
    best_answers = [answer for answer in answers if answer["stdev"] == min_stdev]
    print("\nНаилучшие аппроксимирующие функции:")

    for best_answer in best_answers:
        print(f"\n{best_answer['str_f']}, где:")
        print(f"  a = {best_answer.get('a', float('nan')):.5f}")
        print(f"  b = {best_answer.get('b', float('nan')):.5f}")
        print(f"  c = {best_answer.get('c', float('nan')):.5f}")
        print(f"  d = {best_answer.get('d', float('nan')):.5f}")
        predicted_y = [best_answer["f"](xi) for xi in x]
        correlation = pearson_correlation(y, predicted_y)
        SS_res = np.sum((y - predicted_y) ** 2)
        SS_tot = np.sum((y - np.mean(y)) ** 2)
        R_squared = 1 - (SS_res / SS_tot)
        print(f" Корреляция = {correlation:.5f}")
        print(f" Достоверность = {R_squared:.5f}")

    plot_x = np.linspace(np.min(x), np.max(x), 100)
    plot_y = []
    labels = []
    for answer in answers:
        plot_y.append([answer["f"](x) for x in plot_x])
        labels.append(answer["str_f"])

    plot(x, y, plot_x, plot_y, labels, best_answers)

    input("\n\nНажмите Enter, чтобы выйти.")


main()
