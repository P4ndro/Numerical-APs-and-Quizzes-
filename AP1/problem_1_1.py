

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle, Rectangle
import matplotlib.patches as mpatches

# Set random seed for reproducibility
np.random.seed(42)

class NormCalculator:
    """Class to handle norm and distance calculations"""
    
    @staticmethod
    def vector_norm_l1(x):
        """Compute L1 norm (Manhattan norm): ||x||_1 = sum(|x_i|)"""
        return np.sum(np.abs(x))
    
    @staticmethod
    def vector_norm_l2(x):
        """Compute L2 norm (Euclidean norm): ||x||_2 = sqrt(sum(x_i^2))"""
        return np.sqrt(np.sum(x**2))
    
    @staticmethod
    def matrix_norm_l1(A):
        """
        Induced L1 matrix norm: max column sum
        ||A||_1 = max_j sum_i |a_ij|
        """
        return np.max(np.sum(np.abs(A), axis=0))
    
    @staticmethod
    def matrix_norm_l2(A):
        """
        Induced L2 matrix norm (spectral norm): largest singular value
        ||A||_2 = sigma_max(A)
        """
        return np.linalg.norm(A, ord=2)
    
    @staticmethod
    def distance(x1, x2, norm_func):
        """Compute distance between two vectors/matrices using given norm"""
        return norm_func(x1 - x2)


def generate_random_vectors():
    """Generate two random 4-vectors"""
    v1 = np.random.randn(4)
    v2 = np.random.randn(4)
    return v1, v2


def vectors_to_matrices(v1, v2):
    """Reshape 4-vectors into 2x2 matrices"""
    M1 = v1.reshape(2, 2)
    M2 = v2.reshape(2, 2)
    return M1, M2


def print_results(v1, v2, M1, M2, calc):
    """Print all computed norms and distances"""
    print("=" * 70)
    print("PROBLEM 1.1: VECTOR AND MATRIX NORMS")
    print("=" * 70)
    
    print("\n1. GENERATED VECTORS:")
    print("-" * 70)
    print(f"Vector 1 (v1): {v1}")
    print(f"Vector 2 (v2): {v2}")
    
    print("\n2. VECTORS AS 2×2 MATRICES:")
    print("-" * 70)
    print(f"Matrix 1 (M1):\n{M1}")
    print(f"\nMatrix 2 (M2):\n{M2}")
    
    # Vector norms
    print("\n3. VECTOR NORMS:")
    print("-" * 70)
    print(f"L1 norm of v1: ||v1||_1 = {calc.vector_norm_l1(v1):.4f}")
    print(f"L1 norm of v2: ||v2||_1 = {calc.vector_norm_l1(v2):.4f}")
    print(f"L2 norm of v1: ||v1||_2 = {calc.vector_norm_l2(v1):.4f}")
    print(f"L2 norm of v2: ||v2||_2 = {calc.vector_norm_l2(v2):.4f}")
    
    # Matrix norms 
    print("\n4. INDUCED MATRIX NORMS:")
    print("-" * 70)
    print(f"L1 matrix norm of M1: ||M1||_1 = {calc.matrix_norm_l1(M1):.4f}")
    print(f"L1 matrix norm of M2: ||M2||_1 = {calc.matrix_norm_l1(M2):.4f}")
    print(f"L2 matrix norm of M1: ||M1||_2 = {calc.matrix_norm_l2(M1):.4f}")
    print(f"L2 matrix norm of M2: ||M2||_2 = {calc.matrix_norm_l2(M2):.4f}")
    
    # Distances between vectors
    print("\n5. DISTANCES BETWEEN VECTORS:")
    print("-" * 70)
    v_diff = v2 - v1
    print(f"Difference vector (v2 - v1): {v_diff}")
    dist_l1_v = calc.distance(v1, v2, calc.vector_norm_l1)
    dist_l2_v = calc.distance(v1, v2, calc.vector_norm_l2)
    print(f"L1 distance: d_1(v1, v2) = ||v2 - v1||_1 = {dist_l1_v:.4f}")
    print(f"L2 distance: d_2(v1, v2) = ||v2 - v1||_2 = {dist_l2_v:.4f}")
    
    # Distances between matrices
    print("\n6. DISTANCES BETWEEN MATRICES:")
    print("-" * 70)
    M_diff = M2 - M1
    print(f"Difference matrix (M2 - M1):\n{M_diff}")
    dist_l1_m = calc.distance(M1, M2, calc.matrix_norm_l1)
    dist_l2_m = calc.distance(M1, M2, calc.matrix_norm_l2)
    print(f"L1 distance: d_1(M1, M2) = ||M2 - M1||_1 = {dist_l1_m:.4f}")
    print(f"L2 distance: d_2(M1, M2) = ||M2 - M1||_2 = {dist_l2_m:.4f}")
    
    print("\n" + "=" * 70)


def visualize_unit_balls_2d(v1):
    """Visualize 2D cross-sections of unit balls centered at origin and at v1"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Unit Ball Visualizations: ||x - xr|| ≤ 1', fontsize=16, fontweight='bold')
    
    # Create grid for visualization
    limit = 2.5
    x = np.linspace(-limit, limit, 1000)
    y = np.linspace(-limit, limit, 1000)
    X, Y = np.meshgrid(x, y)
    
    # --- L1 norm unit ball centered at origin ---
    ax = axes[0, 0]
    # L1 norm: |x| + |y| ≤ 1 (diamond shape)
    Z_l1_origin = np.abs(X) + np.abs(Y)
    contour1 = ax.contour(X, Y, Z_l1_origin, levels=[1], colors='blue', linewidths=2)
    ax.contourf(X, Y, Z_l1_origin, levels=[0, 1], colors=['lightblue'], alpha=0.3)
    ax.plot(0, 0, 'ko', markersize=10, label='Center (origin)')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('L1 Norm Unit Ball\n||x||₁ ≤ 1 (centered at origin)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # --- L2 norm unit ball centered at origin ---
    ax = axes[0, 1]
    # L2 norm: x² + y² ≤ 1 (circle)
    Z_l2_origin = X**2 + Y**2
    contour2 = ax.contour(X, Y, Z_l2_origin, levels=[1], colors='red', linewidths=2)
    ax.contourf(X, Y, Z_l2_origin, levels=[0, 1], colors=['lightcoral'], alpha=0.3)
    ax.plot(0, 0, 'ko', markersize=10, label='Center (origin)')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('L2 Norm Unit Ball\n||x||₂ ≤ 1 (centered at origin)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # --- L1 norm unit ball centered at v1[0:2] ---
    ax = axes[1, 0]
    center_x, center_y = v1[0], v1[1]
    Z_l1_shifted = np.abs(X - center_x) + np.abs(Y - center_y)
    contour3 = ax.contour(X, Y, Z_l1_shifted, levels=[1], colors='blue', linewidths=2)
    ax.contourf(X, Y, Z_l1_shifted, levels=[0, 1], colors=['lightblue'], alpha=0.3)
    ax.plot(center_x, center_y, 'ro', markersize=10, label=f'Center xᵣ=({center_x:.2f}, {center_y:.2f})')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(f'L1 Norm Unit Ball\n||x - xᵣ||₁ ≤ 1 (centered at xᵣ)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    # --- L2 norm unit ball centered at v1[0:2] ---
    ax = axes[1, 1]
    Z_l2_shifted = (X - center_x)**2 + (Y - center_y)**2
    contour4 = ax.contour(X, Y, Z_l2_shifted, levels=[1], colors='red', linewidths=2)
    ax.contourf(X, Y, Z_l2_shifted, levels=[0, 1], colors=['lightcoral'], alpha=0.3)
    ax.plot(center_x, center_y, 'ro', markersize=10, label=f'Center xᵣ=({center_x:.2f}, {center_y:.2f})')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(f'L2 Norm Unit Ball\n||x - xᵣ||₂ ≤ 1 (centered at xᵣ)', fontsize=12, fontweight='bold')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    plt.tight_layout()
    plt.savefig('unit_balls_2d.png', dpi=300, bbox_inches='tight')
    print("\n[OK] Saved: unit_balls_2d.png")
    plt.show()


def visualize_unit_balls_3d(v1):
    """Visualize 3D unit balls using first 3 components of v1"""
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle('3D Unit Ball Visualizations', fontsize=16, fontweight='bold')
    
    # Generate sphere points for L2 ball
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    
    # --- L2 norm unit ball centered at origin ---
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.3, color='red', edgecolor='none')
    ax1.scatter([0], [0], [0], color='black', s=100, label='Center (origin)')
    ax1.set_xlabel('x₁', fontsize=11)
    ax1.set_ylabel('x₂', fontsize=11)
    ax1.set_zlabel('x₃', fontsize=11)
    ax1.set_title('L2 Norm Unit Ball (3D)\n||x||₂ ≤ 1', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.set_box_aspect([1,1,1])
    
    # --- L2 norm unit ball centered at v1[0:3] ---
    ax2 = fig.add_subplot(122, projection='3d')
    center = v1[0:3]
    ax2.plot_surface(x_sphere + center[0], y_sphere + center[1], 
                     z_sphere + center[2], alpha=0.3, color='red', edgecolor='none')
    ax2.scatter([center[0]], [center[1]], [center[2]], color='red', s=100, 
                label=f'Center xᵣ=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})')
    ax2.set_xlabel('x₁', fontsize=11)
    ax2.set_ylabel('x₂', fontsize=11)
    ax2.set_zlabel('x₃', fontsize=11)
    ax2.set_title('L2 Norm Unit Ball (3D)\n||x - xᵣ||₂ ≤ 1', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.set_box_aspect([1,1,1])
    
    plt.tight_layout()
    plt.savefig('unit_balls_3d.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: unit_balls_3d.png")
    plt.show()


def visualize_comparison(v1, v2):
    """Compare L1 and L2 distances between vectors visually"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot only first 2 components for 2D visualization
    p1 = v1[0:2]
    p2 = v2[0:2]
    
    # Plot points
    ax.plot(p1[0], p1[1], 'bo', markersize=15, label='Vector 1 (v1)', zorder=5)
    ax.plot(p2[0], p2[1], 'ro', markersize=15, label='Vector 2 (v2)', zorder=5)
    
    # Draw L2 (Euclidean) distance line
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', linewidth=2, 
            label=f'L2 distance = {np.linalg.norm(p2-p1):.3f}', zorder=3)
    
    # Draw L1 (Manhattan) distance path
    ax.plot([p1[0], p2[0]], [p1[1], p1[1]], 'b-', linewidth=2, alpha=0.7, zorder=3)
    ax.plot([p2[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2, alpha=0.7, 
            label=f'L1 distance = {np.sum(np.abs(p2-p1)):.3f}', zorder=3)
    
    # Draw L1 unit ball around v1
    diamond_x = [p1[0]+1, p1[0], p1[0]-1, p1[0], p1[0]+1]
    diamond_y = [p1[1], p1[1]+1, p1[1], p1[1]-1, p1[1]]
    ax.plot(diamond_x, diamond_y, 'b-', linewidth=1.5, alpha=0.4)
    ax.fill(diamond_x, diamond_y, 'blue', alpha=0.1)
    
    # Draw L2 unit ball around v1
    circle = Circle((p1[0], p1[1]), 1, fill=True, alpha=0.1, 
                    edgecolor='red', linewidth=1.5, facecolor='red')
    ax.add_patch(circle)
    
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Distance Comparison: L1 vs L2 Norms\n(with unit balls around v1)', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('distance_comparison.png', dpi=300, bbox_inches='tight')
    print("[OK] Saved: distance_comparison.png")
    plt.show()


def main():
    """Main execution function"""
    print("\nSTARTING PROBLEM 1.1: VECTOR AND MATRIX NORMS\n")
    
    # Initialize calculator
    calc = NormCalculator()
    
    # Step 1 & 2: Generate random vectors
    v1, v2 = generate_random_vectors()
    
    # Step 2: Convert to matrices
    M1, M2 = vectors_to_matrices(v1, v2)
    
    # Step 3: Print all results
    print_results(v1, v2, M1, M2, calc)
    
    # Step 4: Visualizations
    print("\nGENERATING VISUALIZATIONS...")
    print("-" * 70)
    visualize_unit_balls_2d(v1)
    visualize_unit_balls_3d(v1)
    visualize_comparison(v1, v2)
    
    print("\n" + "=" * 70)
    print("EXECUTION COMPLETE!")
    print("Generated files:")
    print("  • unit_balls_2d.png - 2D cross-sections of unit balls")
    print("  • unit_balls_3d.png - 3D visualization of L2 unit balls")
    print("  • distance_comparison.png - Visual comparison of distances")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

