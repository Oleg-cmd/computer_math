import numpy as np
import matplotlib.pyplot as plt
from math import sin, sqrt, factorial


def lagrange_polynomial(dots, x):
    """ Многочлен Лагранжа """
    result = 0

    n = len(dots)
    for i in range(n):
        c1 = c2 = 1
        for j in range(n):
            if i != j:
                c1 *= x - dots[j][0]
                c2 *= dots[i][0] - dots[j][0]
        result += dots[i][1] * c1 / c2

    return result


def t_calc(t, n, forward=True):
    """ Вычислить параметр 't' для многочлена Ньютона """
    result = t

    for i in range(1, n):
        if forward:
            result *= t - i
        else:
            result *= t + i

    return result


def newton_divided_differences(dots):
    """ Многочлен Ньютона с разделенными разностями """
    n = len(dots)
    divided_differences = [dots[i][1] for i in range(n)]
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            divided_differences[i] = (divided_differences[i] - divided_differences[i - 1]) / (dots[i][0] - dots[i - j][0])
    return divided_differences

def newton_polynomial(dots, x):
    """ Многочлен Ньютона с разделенными разностями """
    n = len(dots)
    divided_differences = newton_divided_differences(dots)
    result = divided_differences[0]
    for i in range(1, n):
        term = divided_differences[i]
        for j in range(i):
            term *= (x - dots[j][0])
        result += term
    return result


def gauss_polynomial(dots, x):
    """ Многочлен Гаусса """
    result = 0

    n = len(dots)
    for i in range(n):
        term = 1
        for j in range(n):
            if i != j:
                term *= (x - dots[j][0]) / (dots[i][0] - dots[j][0])
        result += dots[i][1] * term

    return result


def plot(x, y, plot_x, plot_y):
    """ Отрисовать график по заданным координатам узлов и точкам многочлена """
    ax = plt.gca()
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.plot(1, 0, marker=">", ms=5, color='k',
            transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, marker="^", ms=5, color='k',
            transform=ax.get_xaxis_transform(), clip_on=False)

    # Отрисовываем график
    plt.plot(x, y, 'o', plot_x, plot_y)
    plt.show(block=False)


def getfunc(func_id):
    """ Получить выбранную функцию """
    if func_id == '1':
        return lambda x: sqrt(x)
    elif func_id == '2':
        return lambda x: x ** 2
    elif func_id == '3':
        return lambda x: sin(x)
    else:
        return None


def make_dots(f, a, b, n):
    dots = []

    h = (b - a) / (n - 1)
    for i in range(n):
        dots.append((a, f(a)))
        a += h

    return dots


def getdata_input():
    """ Получить данные с клавиатуры """
    data = {}

    print("\nВыберите метод интерполяции.")
    print(" 1 — Многочлен Лагранжа")
    print(" 2 — Многочлен Ньютона с разделенными разностями")
    print(" 3 — Многочлен Гаусса")
    while True:
        try:
            method_id = input("Метод решения: ")
            if method_id not in ('1', '2', '3'):
                raise AttributeError
            break
        except AttributeError:
            print("Метода нет в списке.")
    data['method_id'] = method_id

    print("\nВыберите способ ввода исходных данных.")
    print(" 1 — Набор точек")
    print(" 2 — Функция")
    while True:
        try:
            input_method_id = input("Способ: ")
            if input_method_id not in ('1', '2'):
                raise AttributeError
            break
        except AttributeError:
            print("Способа нет в списке.")

    dots = []
    if input_method_id == '1':
        print("Вводите координаты через пробел, каждая точка с новой строки.")
        print("Чтобы закончить, введите 'END'.")
        while True:
            try:
                current = input()
                if current == 'END':
                    if len(dots) < 2:
                        raise AttributeError
                    break
                x, y = map(float, current.split())
                dots.append((x, y))
            except ValueError:
                print("Введите точку повторно - координаты должны быть числами!")
            except AttributeError:
                print("Минимальное количество точек - две!")
    elif input_method_id == '2':
        print("\nВыберите функцию.")
        print(" 1 — √x")
        print(" 2 - x²")
        print(" 3 — sin(x)")
        while True:
            try:
                func_id = input("Функция: ")
                func = getfunc(func_id)
                if func is None:
                    raise AttributeError
                break
            except AttributeError:
                print("Функции нет в списке.")
        print("\nВведите границы отрезка.")
        while True:
            try:
                a, b = map(float, input("Границы отрезка: ").split())
                if a > b:
                    a, b = b, a
                break
            except ValueError:
                print("Границы отрезка должны быть числами, введенными через пробел.")
        print("\nВыберите количество узлов интерполяции.")
        while True:
            try:
                n = int(input("Количество узлов: "))
                if n < 2:
                    raise ValueError
                break
            except ValueError:
                print("Количество узлов должно быть целым числом > 1.")
        dots = make_dots(func, a, b, n)
    data['dots'] = dots

    print("\nВведите значение аргумента для интерполирования.")
    while True:
        try:
            x = float(input("Значение аргумента: "))
            break
        except ValueError:
            print("Значение аргумента должно быть числом.")
    data['x'] = x

    return data


def main():
    print("\tЛабораторная работа #5 (8)")
    print("\t   Интерполяция функций")

    data = getdata_input()
    x = np.array([dot[0] for dot in data['dots']])
    y = np.array([dot[1] for dot in data['dots']])
    plot_x = np.linspace(np.min(x), np.max(x), 100)
    plot_y = None
    if data['method_id'] == '1':
        answer = lagrange_polynomial(data['dots'], data['x'])
        plot_y = [lagrange_polynomial(data['dots'], x) for x in plot_x]
    elif data['method_id'] == '2':
        answer = newton_polynomial(data['dots'], data['x'])
        plot_y = [newton_polynomial(data['dots'], x) for x in plot_x]
    elif data['method_id'] == '3':
        answer = gauss_polynomial(data['dots'], data['x'])
        plot_y = [gauss_polynomial(data['dots'], x) for x in plot_x]
    else:
        answer = None

    if answer is not None:
        plot(x, y, plot_x, plot_y)

    print("\n\nРезультаты вычисления.")
    print(f"Приближенное значение функции: {answer}")

    input("\n\nНажмите Enter, чтобы выйти.")




main()