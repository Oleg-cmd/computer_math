def f(x):
    return 3 * x**3 - 2 * x**2 - 7 * x - 8


b = 3
a = 2

k1 = 41 * (b - a) / 840
k2 = 216 * (b - a) / 840
k3 = 27 * (b - a) / 840
k4 = 272 * (b - a) / 840

# print(k1 * f(a) + k2 * f((a + b) / 2) + k3 * f(b) + k4 * f(1))

# print(f(18 / 6))


# Midpoint rule
def midpoint_rule(func, a, b, n):
    h = (b - a) / n
    print("h = (b - a) / n = ", h)
    total = 0
    for i in range(n):
        x = a + h / 2 + i * h
        print(f"a + h / 2 + i * h = {a} + {h}/2 + {i}*{h}")
        print("x", i, " = ", x)
        total += func(x)
        print("Сумма: ", total)
    return h * total


# Trapezoidal rule
def trapezoidal_rule(func, a, b, n):
    h = (b - a) / n
    print("h = (b - a) / n = ", h)
    total = (func(a) + func(b)) / 2.0
    for i in range(1, n):
        x = a + i * h
        print("x", i, " = ", x)
        total += func(x)
        print("Сумма: ", total)
    return h * total


# Simpson's rule
def simpsons_rule(func, a, b, n):
    if n % 2 == 1:  # Симпсон работает только для четного количества сегментов
        n += 1
    h = (b - a) / n
    print("h = (b - a) / n = ", h)

    total = func(a) + func(b)
    print("Сумма: ", total)
    for i in range(1, n, 2):
        x = a + i * h
        print("x", i, " = ", x)
        total += 4 * func(x)
        print("Сумма: ", total)
    for i in range(2, n - 1, 2):
        x = a + i * h
        print("x", i, " = ", x)
        total += 2 * func(x)
        print("Сумма: ", total)

    print(h * total / 3)
    return h * total / 3


# Left rectangle rule
def left_rectangle_rule(func, a, b, n):
    h = (b - a) / n
    total = 0
    for i in range(n):
        x = a + i * h
        total += func(x)
    return h * total


# Right rectangle rule
def right_rectangle_rule(func, a, b, n):
    h = (b - a) / n
    print("h = (b - a) / n = ", h)
    total = 0
    for i in range(n):
        x = a + h + i * h
        print("x", i, " = ", x)
        total += func(x)
        print("Сумма: ", total)
    return h * total


n = 11
epsilon = 0.1
k = 1


# def runge_rule(I_h, I_h2, k):
#     return I_h2 + (I_h2 - I_h) / (2**k - 1)


# I_prev = midpoint_rule(f, a, b, n)
# I_curr = midpoint_rule(f, a, b, n * 2)
# while abs(I_curr - I_prev) > epsilon:
#     n *= 2
#     I_prev = I_curr
#     I_curr = midpoint_rule(f, a, b, n * 2)
# refined_result = runge_rule(I_prev, I_curr, k)
# print(refined_result)

simpsons_rule(f, a, b, n)
