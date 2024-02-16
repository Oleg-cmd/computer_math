import numpy as np


def lib_det(matrix_A):
    return np.linalg.det(matrix_A)


def is_matrix_singular(matrix_A):
    """
    Проверяет, является ли матрица вырожденной.

    Параметры:
        matrix_A : Матрица.

    Возвращает:
        True, если матрица вырожденная, False в противном случае.
    """

    if matrix_A.shape[0] != matrix_A.shape[1]:
        return True
    # Проверка наличия нулевой строки или столбца
    if any(np.all(row == 0) for row in matrix_A) or any(
        np.all(column == 0) for column in matrix_A.T
    ):
        return True

    det = lib_det(matrix_A)
    if np.isclose(det, 0):
        return True

    return False


def gaussian_elimination(matrix_A, vector_b):
    """
    Метод Гаусса для решения системы линейных уравнений Ax = b.

    Параметры:
        matrix_A : Матрица коэффициентов системы уравнений.
        vector_b : Вектор свободных членов.

    Возвращает:
        Вектор решений системы уравнений.
    """

    if is_matrix_singular(matrix_A):
        raise ValueError("Матрица вырождена")

    # Приведение к треугольному виду
    n = len(matrix_A)
    for i in range(n):
        # Шаг 1: Элиминация
        for j in range(i + 1, n):
            # Вычисляем множитель, на который будем умножать текущую строку,
            # чтобы занулить элемент под главной диагональю.
            factor = matrix_A[j][i] / matrix_A[i][i]
            # Вычитаем из строки j произведение строки i на множитель.
            matrix_A[j] -= factor * matrix_A[i]
            # Изменяем соответствующий элемент вектора свободных членов.
            vector_b[j] -= factor * vector_b[i]

    # Обратная подстановка
    # Создаем вектор решений
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        # Вычисляем скалярное произведение коэффициентов и решений уже найденных переменных
        inner_product = np.dot(matrix_A[i][i + 1 :], x[i + 1 :])

        # Вычитаем скалярное произведение из соответствующего элемента вектора свободных членов
        residual = vector_b[i] - inner_product

        # Делим полученное значение на коэффициент при текущей переменной
        x[i] = residual / matrix_A[i][i]

    return x


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
    # умножение матрицы на ветор
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
    try:
        # Решение системы методом Гаусса
        solution = gaussian_elimination(matrix_A, vector_b)

        # Вывод результатов
        print_triangular_matrix(matrix_A, vector_b)
        print("Вектор решений (x):", solution)
        residuals = residual_vector(matrix_A, solution, vector_b)
        print("Вектор невязок (r):", residuals)
        determinant = calculate_determinant(matrix_A)
        print("Детерминант:", round(determinant, 3))
    except ValueError as e:
        print("Ошибка:", e)


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
