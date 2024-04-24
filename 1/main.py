import numpy as np


def lib_det(matrix_A, num_permutations):
    # Проверяем, является ли матрица квадратной
    if matrix_A.shape[0] != matrix_A.shape[1]:
        raise ValueError("Matrix must be square")

    # Извлекаем размерность матрицы
    n = matrix_A.shape[0]

    # Если количество перестановок четное, определитель остается без изменений
    # Если количество перестановок нечетное, меняем знак определителя
    sign = 1 if num_permutations % 2 == 0 else -1

    # Находим определитель с помощью метода Гаусса
    det = sign * gaussian_det(matrix_A)

    print(np.linalg.det(matrix_A))

    return det


# Функция для вычисления определителя методом Гаусса
def gaussian_det(matrix_A):
    # Создаем копию матрицы, чтобы не изменять оригинал
    A = np.copy(matrix_A)
    n = A.shape[0]
    det = 1

    # Проходим по диагонали матрицы
    for i in range(n):
        # Если элемент на главной диагонали равен нулю, ищем ненулевой элемент ниже
        if A[i, i] == 0:
            for j in range(i + 1, n):
                if A[j, i] != 0:
                    # Меняем строки местами
                    A[[i, j]] = A[[j, i]]
                    # Изменяем знак определителя
                    det *= -1
                    break
            # Если все элементы на этом столбце равны нулю, определитель равен нулю
            else:
                return 0

        # Приводим матрицу к верхнетреугольному виду
        for j in range(i + 1, n):
            coef = A[j, i] / A[i, i]
            A[j, i + 1 :] -= coef * A[i, i + 1 :]

    # Определитель равен произведению элементов на главной диагонали
    det *= np.prod(np.diagonal(A))

    return det


def sort_rows_with_single_nonzero(matrix_A, vector_b):
    num_permutations = 0  # Инициализируем количество перестановок

    # Создаем копию матрицы для сортировки
    sorted_matrix_A = np.copy(matrix_A)

    # Сортируем строки матрицы
    sorted_matrix_A.sort(axis=1)
    print(sorted_matrix_A)

    # Получаем индексы для сортировки матрицы и вектора
    sorted_indices = np.argsort(np.argmax(matrix_A != 0, axis=1))

    # Сортируем матрицу и вектор
    sorted_matrix_A = matrix_A[sorted_indices]
    print(sorted_matrix_A)
    sorted_vector_b = vector_b[sorted_indices]

    # Сравниваем каждую строку отсортированной матрицы с исходной матрицей,
    # чтобы определить количество перестановок
    for row, sorted_row in zip(matrix_A, sorted_matrix_A):
        if not np.array_equal(row, sorted_row):
            num_permutations += 1

    return sorted_matrix_A, sorted_vector_b, num_permutations


def is_matrix_singular(matrix_A, vector_b, num):
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

    det = lib_det(matrix_A, num)
    if np.isclose(det, 0) or np.all(vector_b == 0):
        return True

    return False


def gaussian_elimination(matrix_A, vector_b, num):
    """
    Метод Гаусса для решения системы линейных уравнений Ax = b.

    Параметры:
        matrix_A : Матрица коэффициентов системы уравнений.
        vector_b : Вектор свободных членов.

    Возвращает:
        Вектор решений системы уравнений.
    """

    if is_matrix_singular(matrix_A, vector_b, num):
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


def c(matrix_A, num):
    """
    Вычисляет определитель матрицы методом Гаусса.

    Параметры:
        matrix_A : Матрица.

    Возвращает:
        Определитель матрицы.
    """
    return lib_det(matrix_A, num)


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
        matrix_A, vector_b, num = sort_rows_with_single_nonzero(matrix_A, vector_b)
        solution = gaussian_elimination(matrix_A, vector_b, num)
        # Вывод результатов
        print_triangular_matrix(matrix_A, vector_b)
        print("Вектор решений (x):", solution)
        residuals = residual_vector(matrix_A, solution, vector_b)
        print("Вектор невязок (r):", residuals)
        determinant = lib_det(matrix_A, num)
        print("Количество перестановок", num)
        print("Детерминант:", round(determinant, 3))

    except ValueError as e:
        print("Ошибка:", e)


# Пример использования:

# Ввод из кода:
print("\n" + "Ввод в коде")
matrix_A = np.array([[0, 1, 2], [5, 0, 3], [0, 7, 8]], dtype=float)
vector_b = np.array([3, 4, 10], dtype=float)

realize(matrix_A, vector_b)

# Ввод из файла:
print("\n" + "Ввод в файле")
matrix_A, vector_b = input_from_file("input.txt")
realize(matrix_A, vector_b)

# Ввод с клавиатуры:
print("\n" + "Ручной ввод")
matrix_A, vector_b = input_from_keyboard()
realize(matrix_A, vector_b)