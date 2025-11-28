
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import integrate, interpolate
from scipy.signal import savgol_filter
from skimage import color, exposure, feature, filters, io, util
from skimage.restoration import denoise_bilateral

# Absolute/relative path to the provided photograph.
IMAGE_PATH = Path(__file__).with_name("photo-1667204651371-5d4a65b8b5a9.jpg")


@dataclass
class ReconstructionResult:
    """Container for downstream consumers or testing."""

    z_pixels: np.ndarray
    r_pixels: np.ndarray
    spline_tck: Tuple[np.ndarray, np.ndarray, int]
    volume_px3: float


def load_and_preprocess(image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the RGB image, convert to grayscale, and apply gentle denoising.
    Returns the original ndarray (for plotting) and a filtered grayscale array.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    rgb = io.imread(image_path)
    if rgb.ndim == 2:  # Already grayscale
        gray = util.img_as_float(rgb)
    else:
        gray = color.rgb2gray(rgb)

    # Slight Gaussian blur suppresses sensor noise before edge detection.
    gray = filters.gaussian(gray, sigma=1.2, preserve_range=True)
    # Bilateral filter keeps object boundaries but damps text/reflection noise.
    gray = denoise_bilateral(gray, sigma_color=0.05, sigma_spatial=3, channel_axis=None)
    gray = exposure.rescale_intensity(gray, in_range="image", out_range=(0.0, 1.0))
    return rgb, gray


def auto_canny(gray: np.ndarray) -> np.ndarray:
    """
    Try several parameter combinations to obtain a non-empty edge map.
    Raises RuntimeError if no combination yields meaningful edges.
    """
    parameter_grid = [
        (1.0, 0.1, 0.3),
        (1.5, 0.08, 0.25),
        (2.0, 0.05, 0.2),
        (2.5, 0.04, 0.15),
    ]
    for sigma, low, high in parameter_grid:
        edges = feature.canny(
            gray,
            sigma=sigma,
            low_threshold=low,
            high_threshold=high,
        )
        edge_density = edges.mean()
        if edge_density > 1e-3:
            return edges

    raise RuntimeError(
        "Edge detection failed. Consider trimming the frame or improving lighting."
    )


def extract_right_profile(edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract the outer right contour (radius) and corresponding vertical positions.
    """
    ys, xs = np.nonzero(edges)
    if ys.size == 0:
        raise ValueError("No edge pixels detected — cannot extract profile.")

    axis_x = np.median(xs)
    right_mask = xs >= axis_x
    ys = ys[right_mask]
    xs = xs[right_mask]
    if ys.size == 0:
        raise ValueError("Right-hand contour could not be located.")

    # Pick the outermost pixel for every scanline.
    line_max = {}
    for y, x in zip(ys, xs):
        if y not in line_max or x > line_max[y]:
            line_max[y] = x

    y_sorted = np.array(sorted(line_max))
    x_sorted = np.array([line_max[y] for y in y_sorted])
    radius = x_sorted - axis_x

    # Remove flat/negative radii and normalize height origin to bottom.
    valid = radius > 0
    if not np.any(valid):
        raise ValueError("Extracted radii are non-positive — check axis selection.")
    y_valid = y_sorted[valid]
    r_valid = radius[valid]

    z_pixels = (edges.shape[0] - 1) - y_valid
    z_pixels = z_pixels - z_pixels.min()
    return z_pixels.astype(float), r_valid.astype(float), float(axis_x)


def fit_spline(z: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Fit a cubic smoothing spline r(z). Regularization is proportional to noise level.
    """
    order = 3
    sort_idx = np.argsort(z)
    z_sorted = z[sort_idx]
    r_sorted = r[sort_idx]

    # Smoothness scaled by the number of points keeps small oscillations in check.
    smoothness = 5e-2 * len(z_sorted)
    tck = interpolate.splrep(z_sorted, r_sorted, k=order, s=smoothness)
    return tck


def compute_volume(tck, z_min: float, z_max: float) -> float:
    """
    Volume of revolution around the z-axis: V = π ∫ r(z)^2 dz.
    """
    integrand = lambda zz: interpolate.splev(zz, tck) ** 2
    integral, _ = integrate.quad(integrand, z_min, z_max, limit=400)
    return np.pi * integral


def build_surface(
    tck, z_min: float, z_max: float, n_z: int = 200, n_theta: int = 120
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a surface of revolution from the spline for visualization.
    """
    z_line = np.linspace(z_min, z_max, n_z)
    r_line = interpolate.splev(z_line, tck)
    r_line = np.clip(r_line, a_min=0.0, a_max=None)

    theta = np.linspace(0.0, 2 * np.pi, n_theta)
    R, Theta = np.meshgrid(r_line, theta)
    Z = np.meshgrid(z_line, theta)[0]

    X = R * np.cos(Theta)
    Y = R * np.sin(Theta)
    return X, Y, Z


def visualize_pipeline(
    original: np.ndarray,
    edges: np.ndarray,
    z: np.ndarray,
    r: np.ndarray,
    tck,
    volume_px3: float,
    axis_x: float,
) -> None:
    """Create 2×2 summary figure for quick inspection."""
    z_norm = z / z.max()
    z_dense = np.linspace(z.min(), z.max(), 500)
    r_dense = interpolate.splev(z_dense, tck)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(original)
    axes[0, 0].axvline(axis_x, color="cyan", linewidth=2, alpha=0.6, label="Symmetry axis")
    axes[0, 0].legend(loc="lower right")
    axes[0, 0].set_title("Original image + assumed axis")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(edges, cmap="gray")
    axes[0, 1].set_title("Canny edges")
    axes[0, 1].axis("off")

    axes[1, 0].plot(r, z_norm, ".", label="Extracted profile")
    axes[1, 0].plot(interpolate.splev(z_dense, tck), z_dense / z.max(), "-", label="Spline")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Radius (pixels)")
    axes[1, 0].set_ylabel("Normalized height")
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].text(
        0.05,
        0.95,
        f"Pixel volume ≈ {volume_px3:.2e}",
        fontsize=14,
        va="top",
    )
    axes[1, 1].text(
        0.05,
        0.65,
        "Units are pixel³.\nReal units need a reference scale.",
        fontsize=11,
        va="top",
    )
    axes[1, 1].axis("off")
    axes[1, 1].set_title("Volume result")

    fig.tight_layout()


def visualize_surface(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
    """Render the reconstructed solid of revolution."""
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.9,
    )
    ax.set_title("3D reconstruction (surface of revolution)")
    ax.set_xlabel("x (radius)")
    ax.set_ylabel("y")
    ax.set_zlabel("z (height)")
    plt.tight_layout()


def smooth_profile(r: np.ndarray, window: int = 17, poly: int = 3) -> np.ndarray:
    """Apply Savitzky-Golay smoothing if enough samples are available."""
    if len(r) < window:
        window = max(5, len(r) | 1)
        if window > len(r):
            return r
    return savgol_filter(r, window_length=window, polyorder=poly)


def run_pipeline(image_path: Path = IMAGE_PATH) -> ReconstructionResult:
    """End-to-end execution for easy reuse."""
    original, gray = load_and_preprocess(image_path)
    edges = auto_canny(gray)
    z_pixels, r_pixels, axis_x = extract_right_profile(edges)
    r_smooth = smooth_profile(r_pixels)
    tck = fit_spline(z_pixels, r_smooth)
    z_min, z_max = z_pixels.min(), z_pixels.max()
    volume_px3 = compute_volume(tck, z_min, z_max)

    X, Y, Z = build_surface(tck, z_min, z_max)
    visualize_pipeline(original, edges, z_pixels, r_smooth, tck, volume_px3, axis_x)
    visualize_surface(X, Y, Z)
    plt.show()

    return ReconstructionResult(z_pixels=z_pixels, r_pixels=r_smooth, spline_tck=tck, volume_px3=volume_px3)


if __name__ == "__main__":
    run_pipeline()

