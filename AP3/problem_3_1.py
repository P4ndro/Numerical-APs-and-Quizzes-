import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from datetime import datetime

# MODIFY THESE FUNCTIONS FOR TA'S REQUIREMENTS
def f(x):
    """1D Function: x^2 * e^x"""
    return x**2 * np.exp(x)

def f_deriv(x):
    """Exact derivative: (x^2 + 2x) * e^x"""
    return (x**2 + 2*x) * np.exp(x)

def g(x, y):
    """2D Function: xy + x^2 + y^2"""
    return x*y + x**2 + y**2

def g_grad(x, y):
    """Exact gradient"""
    fx = y + 2*x
    fy = x + 2*y
    return np.array([fx, fy])

# Evaluation points - MODIFY THESE IF NEEDED
x0 = 0.5
x0_2d, y0_2d = 1.5, 1.0

def central_diff(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def central_diff_2d(f, x, y, h, direction):
    if direction == 'x':
        return (f(x + h, y) - f(x - h, y)) / (2 * h)
    return (f(x, y + h) - f(x, y - h)) / (2 * h)

def analyze_1d(func, deriv, x0, name):
    """Analyze 1D function: tangent line with exact vs FD derivatives + error analysis"""
    h_values = np.logspace(-8, -1, 15)
    exact = deriv(x0)
    
    # Compute errors for different h values
    errors = []
    for h in h_values:
        fd = central_diff(func, x0, h)
        errors.append({'h': h, 'Exact': exact, 'FD': fd, 'Error': abs(fd - exact)})
    df = pd.DataFrame(errors)
    
    # Create figure with 2 plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Function with tangent lines (exact vs FD)
    x = np.linspace(x0 - 2, x0 + 2, 200)
    y0 = func(x0)
    tangent_exact = y0 + exact * (x - x0)
    
    # Use optimal h for FD tangent
    h_optimal = df.loc[df['Error'].idxmin(), 'h']
    fd_optimal = central_diff(func, x0, h_optimal)
    tangent_fd = y0 + fd_optimal * (x - x0)
    
    axes[0].plot(x, func(x), 'b-', linewidth=2, label='f(x)')
    axes[0].plot(x, tangent_exact, 'r-', linewidth=2, label=f"Tangent (exact, f'={exact:.4f})")
    axes[0].plot(x, tangent_fd, 'g--', linewidth=2, label=f'Tangent (FD, h={h_optimal:.1e})')
    axes[0].plot(x0, y0, 'ko', markersize=8)
    axes[0].set_xlabel('x', fontsize=11)
    axes[0].set_ylabel('y', fontsize=11)
    axes[0].set_title(f'{name}\nTangent Line: Exact vs Finite Difference', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Error vs step size
    axes[1].loglog(df['h'], df['Error'], 'o-', linewidth=2, markersize=6, label='FD Error')
    axes[1].loglog(df['h'], df['h']**2, 'k--', alpha=0.5, label='O(h²)')
    axes[1].set_xlabel('Step size h', fontsize=11)
    axes[1].set_ylabel('Absolute Error', fontsize=11)
    axes[1].set_title('Finite Difference Accuracy', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return df, fig

def analyze_2d(func, grad, x0, y0, name, x_range, y_range):
    """Analyze 2D function: tangent plane + normal vector with exact vs FD + error analysis"""
    h_values = np.logspace(-8, -1, 15)
    exact_fx, exact_fy = grad(x0, y0)
    
    # Compute errors for different h values
    errors = []
    for h in h_values:
        fd_fx = central_diff_2d(func, x0, y0, h, 'x')
        fd_fy = central_diff_2d(func, x0, y0, h, 'y')
        error = np.linalg.norm([fd_fx - exact_fx, fd_fy - exact_fy])
        errors.append({
            'h': h,
            'Exact_fx': exact_fx,
            'Exact_fy': exact_fy,
            'FD_fx': fd_fx,
            'FD_fy': fd_fy,
            'Error': error
        })
    df = pd.DataFrame(errors)
    
    # Normal vectors
    normal_exact = np.array([-exact_fx, -exact_fy, 1.0])
    normal_exact = normal_exact / np.linalg.norm(normal_exact)
    
    h_optimal = df.loc[df['Error'].idxmin(), 'h']
    fd_fx_opt = central_diff_2d(func, x0, y0, h_optimal, 'x')
    fd_fy_opt = central_diff_2d(func, x0, y0, h_optimal, 'y')
    normal_fd = np.array([-fd_fx_opt, -fd_fy_opt, 1.0])
    normal_fd = normal_fd / np.linalg.norm(normal_fd)
    
    # Create figure with 2 plots
    fig = plt.figure(figsize=(14, 5))
    
    # Plot 1: Surface with tangent plane and normal vectors
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    
    x = np.linspace(x_range[0], x_range[1], 40)
    y = np.linspace(y_range[0], y_range[1], 40)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)
    z0 = func(x0, y0)
    
    # Tangent plane using exact gradient
    Z_tangent = z0 + exact_fx*(X - x0) + exact_fy*(Y - y0)
    
    ax1.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis', edgecolor='none')
    ax1.plot_surface(X, Y, Z_tangent, alpha=0.3, color='red')
    ax1.scatter([x0], [y0], [z0], color='red', s=100, label='Point')
    
    # Normal vectors
    scale = 1.5
    ax1.quiver(x0, y0, z0, normal_exact[0]*scale, normal_exact[1]*scale, normal_exact[2]*scale,
              color='red', arrow_length_ratio=0.3, linewidth=3, label='Normal (exact)')
    ax1.quiver(x0, y0, z0, normal_fd[0]*scale, normal_fd[1]*scale, normal_fd[2]*scale,
              color='green', arrow_length_ratio=0.3, linewidth=3, label=f'Normal (FD, h={h_optimal:.1e})')
    
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    ax1.set_title(f'{name}\nTangent Plane & Normal Vectors', fontsize=12)
    ax1.legend()
    
    # Plot 2: Error vs step size
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.loglog(df['h'], df['Error'], 'o-', linewidth=2, markersize=6, label='FD Error')
    ax2.loglog(df['h'], df['h']**2, 'k--', alpha=0.5, label='O(h²)')
    ax2.set_xlabel('Step size h', fontsize=11)
    ax2.set_ylabel('Gradient Error (L2 norm)', fontsize=11)
    ax2.set_title('Finite Difference Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return df, fig, normal_exact, normal_fd

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("="*60)
    print("Problem 3.1: Normal Vectors and Tangent Lines/Planes")
    print("Comparing Exact vs Finite Difference Derivatives")
    print(f"Run ID: {timestamp}")
    print("="*60)
    
    # 1D Function
    print(f"\n1D Function: f(x) = x²·e^x at x={x0}")
    df1, fig1 = analyze_1d(f, f_deriv, x0, 'f(x) = x²·e^x')
    fig1.savefig(f'1d_function_{timestamp}.png', dpi=300, bbox_inches='tight')
    df1.to_csv(f'table_1d_{timestamp}.csv', index=False)
    print(f"  Exact derivative: {f_deriv(x0):.6f}")
    print(f"  Min FD error: {df1['Error'].min():.2e} at h={df1.loc[df1['Error'].idxmin(), 'h']:.2e}")
    
    # 2D Function
    print(f"\n2D Function: g(x,y) = xy + x² + y² at ({x0_2d}, {y0_2d})")
    df2, fig2, norm_exact, norm_fd = analyze_2d(g, g_grad, x0_2d, y0_2d,
                                                  'g(x,y) = xy + x² + y²', (-1, 3), (-1, 3))
    fig2.savefig(f'2d_function_{timestamp}.png', dpi=300, bbox_inches='tight')
    df2.to_csv(f'table_2d_{timestamp}.csv', index=False)
    grad = g_grad(x0_2d, y0_2d)
    print(f"  Exact gradient: ({grad[0]:.6f}, {grad[1]:.6f})")
    print(f"  Normal vector (exact): [{norm_exact[0]:.4f}, {norm_exact[1]:.4f}, {norm_exact[2]:.4f}]")
    print(f"  Min FD error: {df2['Error'].min():.2e} at h={df2.loc[df2['Error'].idxmin(), 'h']:.2e}")
    
    # Summary
    summary = pd.DataFrame({
        'Function': ['f(x)', 'g(x,y)'],
        'Point': [f'{x0}', f'({x0_2d},{y0_2d})'],
        'Min_Error': [
            f"{df1['Error'].min():.2e}",
            f"{df2['Error'].min():.2e}"
        ],
        'Optimal_h': [
            f"{df1.loc[df1['Error'].idxmin(), 'h']:.2e}",
            f"{df2.loc[df2['Error'].idxmin(), 'h']:.2e}"
        ]
    })
    summary.to_csv(f'summary_{timestamp}.csv', index=False)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Generated: 2 plots + 2 tables + 1 summary (ID: {timestamp})")
    print("="*60)
    
    plt.show()

if __name__ == "__main__":
    main()
