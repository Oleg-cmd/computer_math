import numpy as np


def gaussian_elimination(matrix_A, vector_b):
    """
    Метод Гаусса для решения системы линейных уравнений Ax = b.

    Параметры:
        matrix_A : Матрица коэффициентов системы уравнений.
        vector_b : Вектор свободных членов.

    Возвращает:
        Вектор решений системы уравнений и вектор неизвестных.
    """
    # Приведение к треугольному виду
    n = len(matrix_A)
    for i in range(n):
        if matrix_A[i][i] == 0:
            raise ValueError("Матрица вырождена")
        for j in range(i + 1, n):
            factor = matrix_A[j][i] / matrix_A[i][i]
            matrix_A[j] -= factor * matrix_A[i]
            vector_b[j] -= factor * vector_b[i]

    # Обратная подстановка
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        if matrix_A[i][i] == 0:
            raise ValueError("Матрица вырождена")
        x[i] = (vector_b[i] - np.dot(matrix_A[i][i + 1 :], x[i + 1 :])) / matrix_A[i][i]

    return x, vector_b


def calculate_determinant(matrix_A):
    """
    Вычисляет определитель матрицы методом Гаусса.

    Параметры:
        matrix_A : Матрица.

    Возвращает:
        Определитель матрицы.
    """
    n = len(matrix_A)
    det = 1.0
    for i in range(n):
        det *= matrix_A[i][i]
    return det


def residual_vector(matrix_A, vector_x, vector_b):
    """
    Вычисляет вектор невязок.

    Параметры:
        matrix_A : Матрица коэффициентов системы уравнений.
        vector_x : Вектор решений системы уравнений.
        vector_b : Вектор свободных членов.

    Возвращает:
       Вектор невязок.
    """
    return vector_b - np.dot(matrix_A, vector_x)


def print_triangular_matrix(matrix_A, vector_b):
    """
    Выводит треугольную матрицу и вектор B.

    Параметры:
        matrix_A : Матрица коэффициентов системы уравнений.
        vector_b : Вектор свободных членов.
    """
    print("Треугольная матрица (A|B):")
    for i in range(len(matrix_A)):
        print("\n")
        print(np.round(matrix_A[i], 3), "|", np.round(vector_b[i], 3))


def input_from_keyboard():
    """
    Ввод матрицы и вектора свободных членов с клавиатуры.
    """
    n = int(input("Введите размерность матрицы: "))
    print("Введите коэффициенты матрицы A:")
    matrix_A = np.zeros((n, n))
    for i in range(n):
        matrix_A[i] = [float(x) for x in input().split()]
    print("Введите вектор свободных членов b:")
    vector_b = np.array([float(x) for x in input().split()])
    return matrix_A, vector_b


def input_from_file(filename):
    """
    Ввод матрицы и вектора свободных членов из файла.
    """
    with open(filename, "r") as file:
        n = int(file.readline())
        matrix_A = np.zeros((n, n))
        for i in range(n):
            matrix_A[i] = [float(x) for x in file.readline().split()]
        vector_b = np.array([float(x) for x in file.readline().split()])
    return matrix_A, vector_b


def print_solution(vector_x):
    """
    Выводит вектор решений и вектор неизвестных.

    Параметры:
        vector_x : Вектор решений системы уравнений.
    """
    print("\n")
    print("Вектор решений (x):", np.round(vector_x, 3))

    print("[", end="")
    for i in range(len(vector_x)):
        my_str = "x" + str(i + 1)
        if i != len(vector_x) - 1:
            my_str += ", "
        print(my_str, end="")
    print("]")


def print_residuals(residuals):
    """
    Выводит вектор невязок.
    """
    print("\n")
    print("Вектор невязок (r):", np.round(residuals, 3))


def realize(matrix_A, vector_b):
    # Решение системы методом Гаусса
    solution, vector_x = gaussian_elimination(matrix_A, vector_b)

    # Вывод результатов
    print_triangular_matrix(matrix_A, vector_b)
    print_solution(vector_x)
    residuals = residual_vector(matrix_A, solution, vector_b)
    print_residuals(residuals)
    determinant = calculate_determinant(matrix_A)
    print("Детерминант:", round(determinant, 3))


# Пример использования:

# Ввод из кода:
print("\n" + "Ввод в коде")
matrix_A = np.array([[2, 1, -1], [-3, -1, 2], [-2, 1, 2]], dtype=float)
vector_b = np.array([8, -11, -3], dtype=float)

realize(matrix_A, vector_b)

# Ввод из файла:
print("\n" + "Ввод в файле")
matrix_A, vector_b = input_from_file("input.txt")
realize(matrix_A, vector_b)

# Ввод с клавиатуры:
print("\n" + "Ручной ввод")
matrix_A, vector_b = input_from_keyboard()
realize(matrix_A, vector_b)
