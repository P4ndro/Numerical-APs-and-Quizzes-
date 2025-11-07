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

# data points from tasks 1 and 2
x_points = np.array([0, 1.5, 3])
y_points = np.array([0, 0.5, 1.5])

# points to evaluate
eval_points = np.array([0.75, 1.5, 2.25])

print("Task 3 - Evaluating P2(x) and computing errors")
print()
print("x\t\tf(x)\t\tP2(x)\t\te(x)")

errors = []
for x_val in eval_points:
    f_val = f(x_val)
    P2_val = lagrange_interpolation(x_val, x_points, y_points)
    error = abs(f_val - P2_val)
    errors.append(error)
    print(f"{x_val}\t\t{f_val:.6f}\t{P2_val:.6f}\t{error:.6f}")

print()
print("Errors:", errors)

