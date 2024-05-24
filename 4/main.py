import numpy as np
import matplotlib.pyplot as plt
from math import log, exp, sqrt
from sympy import symbols, sqrt, latex



FILE_IN = "input.txt"
FILE_OUT = "output.txt"


def solve_minor(matrix, i, j):
    """Найти минор элемента матрицы"""
    n = len(matrix)
    # print(f"Вычисление минора для элемента в позиции ({i}, {j})")
    minor = [
        [matrix[row][col] for col in range(n) if col != j]
        for row in range(n)
        if row != i
    ]
    # print(f"Минор для элемента в позиции ({i}, {j}) равен: \n{minor}\n")
    return minor

def solve_det(matrix):
    """Найти определитель матрицы"""
    n = len(matrix)
    if n == 1:
        # print(f"Определитель матрицы {matrix} равен: {matrix[0][0]}")
        return matrix[0][0]
    det = 0
    sgn = 1
    for j in range(n):
        det += sgn * matrix[0][j] * solve_det(solve_minor(matrix, 0, j))
        sgn *= -1
    print(f"Определитель матрицы {matrix} равен: {det}\n")
    return det

def calc_s(dots, f):
    """Найти меру отклонения"""
    n = len(dots)
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]

    s = sum([(f(x[i]) - y[i]) ** 2 for i in range(n)])
    print(f"Мера отклонения для точек {dots} и функции f равна: {s}\n")
    return s



def calc_stdev(dots, f):
    """Найти среднеквадратичное отклонение"""
    n = len(dots)

    # Вывод формулы среднеквадратичного отклонения
    print("Формула среднеквадратичного отклонения:")
    print("σ = sqrt(∑(y - f(x))^2 / n)")

    s = sqrt(calc_s(dots, f) / n)
    print(f"Среднеквадратичное отклонение для точек {dots} и функции f равно: {s}\n")
    return s


def lin_func(dots):
    """Линейная аппроксимация"""
    print("Считаем ЛИНЕЙНУЮ АППРОКСИМАЦИЮ\n")
    data = {}

    n = len(dots)
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]

    # Вычисление сумм для линейной аппроксимации
    sx = sum(x)
    sx2 = sum([xi**2 for xi in x])
    sy = sum(y)
    sxy = sum([x[i] * y[i] for i in range(n)])

    print(f"\nВычисление сумм для линейной аппроксимации:\n")
    print(f"sx = {sx}")
    print(f"sx2 = {sx2}")
    print(f"sy = {sy}")
    print(f"sxy = {sxy}")

    # Вычисление определителей для линейной аппроксимации
    d = solve_det([[sx2, sx], [sx, n]])
    d1 = solve_det([[sxy, sx], [sy, n]])
    d2 = solve_det([[sx2, sxy], [sx, sy]])

    print(f"\nВычисление определителей для линейной аппроксимации:\n")
    print(f"d = {d}")
    print(f"d1 = {d1}")
    print(f"d2 = {d2}")

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

    print(f"\nЛинейная аппроксимация:\n")
    print(f"fi = {a}*x + {b}")

    data["s"] = calc_s(dots, f)

    print(f"\nМера отклонения:\n")
    print(f"s = {data['s']}")

    data["stdev"] = calc_stdev(dots, f)

    print(f"\nСреднеквадратичное отклонение:\n\n")
    print(f"σ = {data['stdev']}")

    return data



def sqrt_func(dots):
    """Квадратичная аппроксимация"""
    print("Считаем КВАДРАТИЧНУЮ АППРОКСИМАЦИЮ\n")
    data = {}

    n = len(dots)
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]

    # Вычисление сумм для квадратичной аппроксимации
    sx = sum(x)
    sx2 = sum([xi**2 for xi in x])
    sx3 = sum([xi**3 for xi in x])
    sx4 = sum([xi**4 for xi in x])
    sy = sum(y)
    sxy = sum([x[i] * y[i] for i in range(n)])
    sx2y = sum([(x[i] ** 2) * y[i] for i in range(n)])

    print(f"\nВычисление сумм для квадратичной аппроксимации:\n")
    print(f"sx = {sx}")
    print(f"sx2 = {sx2}")
    print(f"sx3 = {sx3}")
    print(f"sx4 = {sx4}")
    print(f"sy = {sy}")
    print(f"sxy = {sxy}")
    print(f"sx2y = {sx2y}")

    # Вычисление определителей для квадратичной аппроксимации
    d = solve_det([[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]])
    d1 = solve_det([[sy, sx, sx2], [sxy, sx2, sx3], [sx2y, sx3, sx4]])
    d2 = solve_det([[n, sy, sx2], [sx, sxy, sx3], [sx2, sx2y, sx4]])
    d3 = solve_det([[n, sx, sy], [sx, sx2, sxy], [sx2, sx3, sx2y]])

    print(f"\nВычисление определителей для квадратичной аппроксимации:\n")
    print(f"d = {d}")
    print(f"d1 = {d1}")
    print(f"d2 = {d2}")
    print(f"d3 = {d3}")

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

    print(f"\nКвадратичная аппроксимация:\n")
    print(f"fi = {a}*x^2 + {b}*x + {c}")

    data["s"] = calc_s(dots, f)

    print(f"\nМера отклонения:\n")
    print(f"s = {data['s']}")

    data["stdev"] = calc_stdev(dots, f)

    print(f"\nСреднеквадратичное отклонение:\n")
    print(f"σ = {data['stdev']}")

    return data


def cubic_func(dots):
    """Кубическая аппроксимация"""
    print("Считаем КУБИЧЕСКУЮ АППРОКСИМАЦИЮ\n")
    data = {}

    n = len(dots)
    x = [dot[0] for dot in dots]
    y = [dot[1] for dot in dots]

    # Вычисление сумм для кубической аппроксимации
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

    print(f"\nВычисление сумм для кубической аппроксимации:\n")
    print(f"sx = {sx}")
    print(f"sx2 = {sx2}")
    print(f"sx3 = {sx3}")
    print(f"sx4 = {sx4}")
    print(f"sx5 = {sx5}")
    print(f"sx6 = {sx6}")
    print(f"sy = {sy}")
    print(f"sxy = {sxy}")
    print(f"sx2y = {sx2y}")
    print(f"sx3y = {sx3y}")

    # Вычисление определителей для кубической аппроксимации
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

    print(f"\nВычисление определителей для кубической аппроксимации:\n")
    print(f"d = {d}")
    print(f"d1 = {d1}")
    print(f"d2 = {d2}")
    print(f"d3 = {d3}")
    print(f"d4 = {d4}")

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

    print(f"\nКубическая аппроксимация:\n")
    print(f"fi = {a}*x^3 + {b}*x^2 + {c}*x + {d}")

    data["s"] = calc_s(dots, f)

    print(f"\nМера отклонения:\n")
    print(f"s = {data['s']}")

    data["stdev"] = calc_stdev(dots, f)

    print(f"\nСреднеквадратичное отклонение:\n")
    print(f"σ = {data['stdev']}")

    return data


def exp_func(dots):
    """Экспоненциальная аппроксимация"""
    print("Считаем ЭКСПОНЕНЦИАЛЬНУЮ АППРОКСИМАЦИЮ\n")
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

    # Применение логарифмической функции к y для линеаризации
    lin_y = [log(yi) for yi in y]

    # Линейная аппроксимация линеаризованных данных
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

    print(f"\nЭкспоненциальная аппроксимация:\n")
    print(f"fi = {a}*e^({b}*x)")

    data["s"] = calc_s(valid_dots, f)

    print(f"\nМера отклонения:\n")
    print(f"s = {data['s']}")

    data["stdev"] = calc_stdev(valid_dots, f)

    print(f"\nСреднеквадратичное отклонение:\n")
    print(f"σ = {data['stdev']}")

    return data


def log_func(dots):
    """Логарифмическая аппроксимация"""
    print("Считаем ЛОГАРИФМИЧЕСКУЮ АППРОКСИМАЦИЮ\n")
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

    # Применение логарифмической функции к x для линеаризации
    lin_x = [log(xi) for xi in x]

    # Линейная аппроксимация линеаризованных данных
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

    print(f"\nЛогарифмическая аппроксимация:\n")
    print(f"fi = {a}*ln(x) + {b}")

    data["s"] = calc_s(valid_dots, f)

    print(f"\nМера отклонения:\n")
    print(f"s = {data['s']}")

    data["stdev"] = calc_stdev(valid_dots, f)

    print(f"\nСреднеквадратичное отклонение:\n")
    print(f"σ = {data['stdev']}")

    return data



def pow_func(dots):
    """Степенная аппроксимация"""
    print("Считаем СТЕПЕННУЮ АППРОКСИМАЦИЮ\n")
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

    # Применение логарифмической функции к x и y для линеаризации
    lin_x = [log(xi) for xi in x]
    lin_y = [log(yi) for yi in y]

    # Линейная аппроксимация линеаризованных данных
    lin_result = lin_func(list(zip(lin_x, lin_y)))
    

    if lin_result is None:
        print("Linear approximation failed in power function.")
        return None

    # Возведение в степень коэффициента пересечения для получения 'a' и использование углового коэффициента в качестве 'b'
    a = exp(lin_result["b"])
    b = lin_result["a"]

    # Определение степенной функции
    f = lambda z: (
        a * (z**b) if z > 0 else float("inf")
    )  # Still safeguard against z <= 0

    data["a"] = a
    data["b"] = b
    data["f"] = f
    data["str_f"] = "fi = a*x^b"
    data["s"] = calc_s(dots, f)  # Calculate sum of squares
    data["stdev"] = calc_stdev(dots, f)  # Calculate standard deviation

    print(f"\nСтепенная аппроксимация:\n")
    print(f"fi = {a}*x^{b}")

    print(f"\nМера отклонения:\n")
    print(f"s = {data['s']}")

    print(f"\nСреднеквадратичное отклонение:\n")
    print(f"σ = {data['stdev']}")

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
