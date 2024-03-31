import sympy as sp

# Define the variable and function for symbolic integration
x = sp.symbols("x")
f = 3 * x**3 - 2 * x**2 - 7 * x - 8

# Perform the exact integration
exact_integral = sp.integrate(f, (x, 2, 3))
print(exact_integral.evalf())
