import numpy as np

def f(x):
    return x**2 / (x + 3)

def lagrange_basis(x, xi, x_points):
    L = 1.0
    for xj in x_points:
        if xj != xi:
            L *= (x - xj) / (xi - xj)
    return L

def lagrange_interpolation(x, x_points, y_points):
    P = 0.0
    for i in range(len(x_points)):
        P += y_points[i] * lagrange_basis(x, x_points[i], x_points)
    return P

print("Task 5 - Adding node x3=2 and constructing P3(x)")
print()

# new node
x3 = 2
y3 = f(x3)
print(f"f({x3}) = {y3}")

# cubic with 4 points
x_cubic = np.array([0, 1.5, 2, 3])
y_cubic = np.array([0, 0.5, y3, 1.5])

print()
print("P3(x) uses 4 points:")
for i in range(len(x_cubic)):
    print(f"({x_cubic[i]}, {y_cubic[i]})")

# testing
print()
print("Testing P3(x):")
test_points = [0.75, 1.5, 2.0, 2.25]
for x_val in test_points:
    P3_val = lagrange_interpolation(x_val, x_cubic, y_cubic)
    f_val = f(x_val)
    print(f"P3({x_val}) = {P3_val:.6f}, f({x_val}) = {f_val:.6f}")

