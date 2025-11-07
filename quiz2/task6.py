import numpy as np
import matplotlib.pyplot as plt

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

print("Task 6 - Plotting")

# P2(x) with 3 points
x_quad = np.array([0, 1.5, 3])
y_quad = np.array([0, 0.5, 1.5])

# P3(x) with 4 points
x_cubic = np.array([0, 1.5, 2, 3])
y_cubic = np.array([0, 0.5, 0.8, 1.5])

# plot points
x_plot = np.linspace(0, 3, 300)
f_plot = f(x_plot)

P2_plot = np.array([lagrange_interpolation(x, x_quad, y_quad) for x in x_plot])
P3_plot = np.array([lagrange_interpolation(x, x_cubic, y_cubic) for x in x_plot])

# main plot
plt.figure(figsize=(10, 6))

plt.plot(x_plot, f_plot, 'b-', linewidth=2, label='f(x)')
plt.plot(x_plot, P2_plot, 'r--', linewidth=2, label='P2(x)')
plt.plot(x_plot, P3_plot, 'g-.', linewidth=2, label='P3(x)')

plt.plot(x_quad, y_quad, 'ro', markersize=8, label='P2 nodes')
plt.plot(x_cubic, y_cubic, 'gs', markersize=8, label='P3 nodes')

plt.xlabel('x')
plt.ylabel('y')
plt.title('f(x), P2(x), and P3(x)')
plt.legend()
plt.grid(True)

plt.savefig('task6_plot.png', dpi=200)
print("Plot saved as task6_plot.png")
plt.show()

