import numpy as np

def f(x):
    return np.cos(x)

def trap(f, a, b, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    return h * (0.5*f(x[0]) + 0.5*f(x[-1]) + f(x[1:-1]).sum())

def simpson(f, a, b, n):
    assert n % 2 == 0, "n must be even for Simpson"
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    return (h/3) * (f(x[0]) + f(x[-1]) +
                    4*f(x[1:-1:2]).sum() +
                    2*f(x[2:-1:2]).sum())

a, b = 0.0, np.pi/2
exact = 1.0

for n in [1, 2, 4]:
    T = trap(f, a, b, n)
    print(f"Trapezoid n={n}: approx={T:.10f}, error={abs(T-exact):.10e}")

for n in [2, 4]:
    S = simpson(f, a, b, n)
    print(f"Simpson n={n}:  approx={S:.10f}, error={abs(S-exact):.10e}")
