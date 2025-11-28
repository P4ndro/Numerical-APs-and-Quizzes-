
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from scipy.interpolate import CubicSpline, make_interp_spline, splrep, splev

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

FONT = FontProperties(family="DejaVu Sans")
DIGITS: Sequence[str] = ("2", "3", "8")
RAW_SAMPLE_POINTS = 24
REFERENCE_POINTS = 400
NODE_LEVELS = {
    "full": 24,
    "half": 12,
    "few": 6,
}
NODE_STRATEGIES = ("uniform", "chord")


@dataclass(frozen=True)
class SplineConfig:
    name: str
    method: str  # cubic | make | splrep
    kwargs: Dict


SPLINE_CONFIGS: Tuple[SplineConfig, ...] = (
    SplineConfig("cubic_notaknot", "cubic", {"bc_type": "not-a-knot"}),
    SplineConfig("cubic_natural", "cubic", {"bc_type": "natural"}),
    SplineConfig("cubic_clamped", "cubic", {"bc_type": ((1, 0.0), (1, 0.0))}),
    SplineConfig("bspline_interp", "make", {"k": 3}),
    SplineConfig("bspline_smooth", "splrep", {"k": 3, "s": 1e-4}),
)


def _load_digit_polygons(char: str) -> List[np.ndarray]:
    """Return normalized polygon outlines for the digit."""
    text_path = TextPath((0, 0), char, size=1, prop=FONT)
    polygons = text_path.to_polygons(closed_only=False)
    if not polygons:
        raise ValueError(f"No outline extracted for '{char}'.")

    arrays = [np.asarray(poly, dtype=float) for poly in polygons if len(poly) >= 3]
    stacked = np.vstack(arrays)
    centroid = stacked.mean(axis=0)
    arrays = [poly - centroid for poly in arrays]
    span = np.ptp(stacked, axis=0).max()
    if span > 0:
        arrays = [poly / span for poly in arrays]
    return arrays


def _resample(points: np.ndarray, count: int, close: bool = True) -> np.ndarray:
    """Resample a closed polygon to a requested number of points using arc length."""
    pts = np.asarray(points, dtype=float)
    if close and not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    segment = np.diff(pts, axis=0)
    d = np.linalg.norm(segment, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(d)])
    if cumulative[-1] == 0:
        return pts[:count]

    normalized = cumulative / cumulative[-1]
    targets = np.linspace(0.0, 1.0, count, endpoint=False)
    x = np.interp(targets, normalized, pts[:, 0])
    y = np.interp(targets, normalized, pts[:, 1])
    resampled = np.column_stack([x, y])
    return resampled


def _select_nodes(points: np.ndarray, count: int, strategy: str) -> np.ndarray:
    """Pick a subset of nodes using a specified strategy."""
    pts = np.asarray(points)
    count = min(count, len(pts))
    if count <= 2:
        return pts[:count]

    if strategy == "uniform":
        idx = np.linspace(0, len(pts) - 1, count, dtype=int)
        return pts[idx]

    if strategy == "chord":
        diffs = np.diff(pts, axis=0, append=pts[:1])
        d = np.linalg.norm(diffs, axis=1)
        cumulative = np.cumsum(d)
        cumulative = np.insert(cumulative, 0, 0.0)
        total = cumulative[-1]
        if total == 0:
            return pts[:count]
        targets = np.linspace(0.0, total, count, endpoint=False)
        x = np.interp(targets, cumulative, np.concatenate([pts[:, 0], [pts[0, 0]]]))
        y = np.interp(targets, cumulative, np.concatenate([pts[:, 1], [pts[0, 1]]]))
        return np.column_stack([x, y])

    raise ValueError(f"Unknown strategy '{strategy}'.")


def _remove_duplicate_nodes(nodes: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    """Drop consecutive duplicate nodes that break spline construction."""
    if len(nodes) <= 1:
        return nodes
    diffs = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    keep = np.concatenate([[True], diffs > tol])
    filtered = nodes[keep]
    return filtered


def _parameterize(points: np.ndarray) -> np.ndarray:
    """Compute chord-length parameter in [0, 1]."""
    if len(points) == 1:
        return np.array([0.0])
    diffs = np.diff(points, axis=0)
    d = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(d)])
    total = cumulative[-1]
    if total == 0:
        return np.linspace(0.0, 1.0, len(points))
    return cumulative / total


def _build_spline(
    t_nodes: np.ndarray, coord: np.ndarray, config: SplineConfig
) -> object:
    """Return spline representation for a single coordinate."""
    if config.method == "cubic":
        spline = CubicSpline(t_nodes, coord, bc_type=config.kwargs["bc_type"])
        return spline

    if config.method == "make":
        spline = make_interp_spline(t_nodes, coord, **config.kwargs)
        return spline

    if config.method == "splrep":
        tck = splrep(t_nodes, coord, **config.kwargs)
        return tck

    raise ValueError(f"Unsupported spline method '{config.method}'.")


def _evaluate_spline(
    t_nodes: np.ndarray,
    nodes: np.ndarray,
    t_eval: np.ndarray,
    config: SplineConfig,
) -> np.ndarray:
    """Evaluate parametric spline at dense parameter values."""
    x_nodes, y_nodes = nodes[:, 0], nodes[:, 1]
    if config.method in {"cubic", "make"}:
        sx = _build_spline(t_nodes, x_nodes, config)
        sy = _build_spline(t_nodes, y_nodes, config)
        x = sx(t_eval)
        y = sy(t_eval)
    else:
        tck_x = _build_spline(t_nodes, x_nodes, config)
        tck_y = _build_spline(t_nodes, y_nodes, config)
        x = splev(t_eval, tck_x)
        y = splev(t_eval, tck_y)
    return np.column_stack([x, y])


def _compute_errors(reference: np.ndarray, approx: np.ndarray) -> Tuple[float, float]:
    """Return RMS and max Euclidean errors."""
    if len(reference) != len(approx):
        raise ValueError("Reference and approximation must share sample count.")
    diff = reference - approx
    dist = np.sqrt(np.sum(diff**2, axis=1))
    rms = float(np.sqrt(np.mean(dist**2)))
    emax = float(np.max(dist))
    return rms, emax


def _plot_digit_results(
    digit: str,
    strategy: str,
    node_frames: Dict[str, Dict[str, Dict[str, np.ndarray]]],
    raw_polygons: Sequence[np.ndarray],
) -> None:
    """Plot original samples and spline fits for a digit/strategy combination."""
    levels = list(NODE_LEVELS.keys())
    fig, axes = plt.subplots(1, len(levels), figsize=(5 * len(levels), 5), sharex=True, sharey=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(SPLINE_CONFIGS)))

    for ax, level in zip(axes, levels):
        for poly in raw_polygons:
            ax.plot(poly[:, 0], poly[:, 1], "o", ms=4, alpha=0.4, label="raw points")
        for color, config in zip(colors, SPLINE_CONFIGS):
            curves = node_frames[level][config.name]["curve"]
            for curve in curves:
                ax.plot(curve[:, 0], curve[:, 1], "-", color=color, lw=2, alpha=0.9)
        ax.set_title(f"{digit}: {strategy} – {level}")
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], color=color, lw=3, label=config.name)
        for color, config in zip(colors, SPLINE_CONFIGS)
    ]
    axes[-1].legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    fig.suptitle(f"Digit {digit} – {strategy} node selection", fontsize=16)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(OUTPUT_DIR / f"digit_{digit}_{strategy}.png", dpi=200)
    plt.close(fig)


def run_experiments() -> List[Dict[str, str]]:
    """Main experiment loop for all digits."""
    summary: List[Dict[str, str]] = []

    for digit in DIGITS:
        polygons = _load_digit_polygons(digit)
        raw_polys = [_resample(poly, RAW_SAMPLE_POINTS) for poly in polygons]
        reference_polys = [_resample(poly, REFERENCE_POINTS) for poly in polygons]
        t_ref = np.linspace(0.0, 1.0, REFERENCE_POINTS, endpoint=False)

        for strategy in NODE_STRATEGIES:
            node_frames: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
            for level_name, count in NODE_LEVELS.items():
                node_frames[level_name] = {}
                for config in SPLINE_CONFIGS:
                    node_frames[level_name][config.name] = {"curve": [], "errors": []}

                for raw_poly, ref_poly in zip(raw_polys, reference_polys):
                    nodes = _select_nodes(raw_poly, count, strategy)
                    nodes = _remove_duplicate_nodes(nodes)
                    if len(nodes) <= 3:
                        continue
                    t_nodes = _parameterize(nodes)
                    for config in SPLINE_CONFIGS:
                        curve = _evaluate_spline(t_nodes, nodes, t_ref, config)
                        rms, emax = _compute_errors(ref_poly, curve)
                        node_frames[level_name][config.name]["curve"].append(curve)
                        node_frames[level_name][config.name]["errors"].append((rms, emax))

                for config in SPLINE_CONFIGS:
                    errors = node_frames[level_name][config.name]["errors"]
                    rms_avg = float(np.mean([e[0] for e in errors]))
                    emax_avg = float(np.mean([e[1] for e in errors]))
                    summary.append(
                        {
                            "digit": digit,
                            "strategy": strategy,
                            "level": level_name,
                            "spline": config.name,
                            "rms": f"{rms_avg:.4f}",
                            "emax": f"{emax_avg:.4f}",
                        }
                    )

            _plot_digit_results(digit, strategy, node_frames, raw_polys)

    return summary


def print_summary(summary: Iterable[Dict[str, str]]) -> None:
    """Pretty-print the experiment summary table."""
    header = f"{'digit':>5} | {'strategy':>8} | {'nodes':>5} | {'spline':<15} | {'RMS':>8} | {'Emax':>8}"
    print(header)
    print("-" * len(header))
    for row in summary:
        print(
            f"{row['digit']:>5} | "
            f"{row['strategy']:>8} | "
            f"{row['level']:>5} | "
            f"{row['spline']:<15} | "
            f"{row['rms']:>8} | "
            f"{row['emax']:>8}"
        )


def main() -> None:
    summary = run_experiments()
    print_summary(summary)


if __name__ == "__main__":
    main()

