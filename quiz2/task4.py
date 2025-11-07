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

x_points = np.array([0, 1.5, 3])
y_points = np.array([0, 0.5, 1.5])

eval_points = np.array([0.75, 1.5, 2.25])

# calculate errors
errors = []
for x_val in eval_points:
    f_val = f(x_val)
    P2_val = lagrange_interpolation(x_val, x_points, y_points)
    error = abs(f_val - P2_val)
    errors.append(error)

errors = np.array(errors)

print("Task 4 - Computing norms")
print()
print("Errors:")
print(f"e(0.75) = {errors[0]:.6f}")
print(f"e(1.50) = {errors[1]:.6f}")
print(f"e(2.25) = {errors[2]:.6f}")
print()

# L2 norm
L2_norm = np.sqrt(np.sum(errors**2))
print(f"L2 norm = sqrt({errors[0]:.6f}^2 + {errors[1]:.6f}^2 + {errors[2]:.6f}^2)")
print(f"        = {L2_norm:.6f}")
print()

# L_inf norm
Linf_norm = np.max(errors)
print(f"L_inf norm = max of errors = {Linf_norm:.6f}")

