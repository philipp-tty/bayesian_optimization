#!/usr/bin/env python3
"""Run Bayesian Optimization on 2D benchmark functions."""

import argparse
import importlib.util
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithms.bayesian_optimization import BayesianOptimization


def ensure_interactive_backend() -> bool:
    """
    Ensure a GUI backend is selected for matplotlib.

    Returns:
        bool: True if an interactive backend is available, False otherwise.
    """
    current_backend = matplotlib.get_backend()

    # Check if already interactive
    if current_backend and "agg" not in current_backend.lower():
        return True

    # Try common interactive backends
    gui_backends = ["QtAgg", "Qt5Agg", "TkAgg", "GTKAgg", "WXAgg", "MacOSX"]

    for backend in gui_backends:
        try:
            matplotlib.use(backend, force=True)
            print(f"Switched to {backend} backend for plotting")
            return True
        except (ImportError, ValueError):
            continue

    print("Warning: No interactive matplotlib backend available. Plots will not be shown.")
    return False


def setup_plot_style() -> None:
    """Configure matplotlib with a modern, scientific style."""
    # Try different seaborn style names for compatibility
    for style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]:
        try:
            plt.style.use(style)
            break
        except OSError:
            continue

    # Custom style parameters
    matplotlib.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "axes.edgecolor": "#333333",
        "axes.labelsize": 10,
        "axes.titlesize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "font.size": 10,
        "figure.figsize": (8, 6)
    })


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_benchmarks_module():
    """Dynamically load the 2d_minimization.py module."""
    mod_path = Path(__file__).resolve().parent / "test_functions" / "2d_minimization.py"

    if not mod_path.exists():
        raise FileNotFoundError(f"Benchmark module not found at {mod_path}")

    spec = importlib.util.spec_from_file_location("bench2d", str(mod_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {mod_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["bench2d"] = module
    spec.loader.exec_module(module)

    return module


def parse_arguments(bench_mod) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization on a 2D benchmark function.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default="himmelblau",
        choices=bench_mod.available_benchmarks(),
        help="Benchmark function to optimize"
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Number of BO iterations (acquisitions)"
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=8,
        help="Number of initial Sobol samples"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Standard deviation of i.i.d. Gaussian noise on outputs"
    )
    parser.add_argument(
        "--normalized",
        action="store_true",
        default=True,
        help="Work in normalized [0,1]^2 space"
    )
    parser.add_argument(
        "--no-normalized",
        dest="normalized",
        action="store_false",
        help="Work in raw bounds space"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Show live optimization plot"
    )

    return parser.parse_args()


def build_grid(
        bounds: torch.Tensor,
        resolution: int = 220,
        dtype: torch.dtype = torch.double
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build a 2D grid for plotting.

    Returns:
        Tuple of (X_grid, Y_grid, XY_flat)
    """
    lb = bounds[0].to(dtype=dtype).cpu()
    ub = bounds[1].to(dtype=dtype).cpu()

    xs = torch.linspace(float(lb[0]), float(ub[0]), resolution, dtype=dtype)
    ys = torch.linspace(float(lb[1]), float(ub[1]), resolution, dtype=dtype)

    X_grid, Y_grid = torch.meshgrid(xs, ys, indexing="ij")
    XY_flat = torch.stack([X_grid.reshape(-1), Y_grid.reshape(-1)], dim=-1)

    return X_grid, Y_grid, XY_flat


def normalize_points(X: torch.Tensor, raw_bounds: torch.Tensor) -> torch.Tensor:
    """Normalize points from raw space to [0,1]^d."""
    lb = raw_bounds[0].to(dtype=X.dtype, device=X.device)
    ub = raw_bounds[1].to(dtype=X.dtype, device=X.device)
    return (X - lb) / (ub - lb + 1e-12)


class LivePlotter:
    """Handle live plotting during optimization."""

    def __init__(
            self,
            objective,
            bounds_plot_space: torch.Tensor,
            bench,
            normalized: bool,
            raw_bounds: torch.Tensor
    ):
        self.objective = objective
        self.bench = bench
        self.normalized = normalized
        self.raw_bounds = raw_bounds
        self.fig = None
        self.ax = None
        self.artists = {}

        if not ensure_interactive_backend():
            self.enabled = False
            return

        self.enabled = True
        setup_plot_style()
        self._initialize_plot(bounds_plot_space)

    def _initialize_plot(self, bounds_plot_space: torch.Tensor) -> None:
        """Set up the initial plot with contours and markers."""
        # Generate contour data
        X_grid, Y_grid, XY_flat = build_grid(bounds_plot_space, resolution=220)

        with torch.no_grad():
            Z = self.objective(XY_flat.to(dtype=torch.double))
            Z = Z.view(X_grid.shape[0], X_grid.shape[1]).cpu().numpy()

        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

        # Plot contours
        contour = self.ax.contourf(
            X_grid.numpy(), Y_grid.numpy(), Z,
            levels=30, cmap="magma", alpha=0.95
        )
        cbar = self.fig.colorbar(contour, ax=self.ax, pad=0.01)
        cbar.set_label("f(x)", fontsize=10)

        # Plot known global minima
        mins_raw = self.bench.global_minima.to(dtype=torch.double)
        mins_plot = normalize_points(mins_raw, self.raw_bounds) if self.normalized else mins_raw

        self.ax.scatter(
            mins_plot[:, 0].cpu(), mins_plot[:, 1].cpu(),
            marker="X", s=90, c="#06d6a0",
            edgecolors="#1b1e23", linewidths=0.6,
            zorder=5, label="Known minima"
        )

        # Initialize dynamic artists
        self.artists['points'] = self.ax.scatter(
            [], [], c=[], cmap="viridis",
            vmin=0.0, vmax=1.0, s=36,
            edgecolors="#ffffff", linewidths=0.6, zorder=4
        )
        self.artists['path'], = self.ax.plot(
            [], [], color="#ffffff",
            lw=1.1, alpha=0.55, zorder=3
        )
        self.artists['best'] = self.ax.scatter(
            [], [], marker="*", s=150, c="#FFD166",
            edgecolors="#1b1e23", linewidths=0.7,
            zorder=6, label="Current best"
        )

        # Configure axes
        lb = bounds_plot_space[0].cpu().numpy()
        ub = bounds_plot_space[1].cpu().numpy()

        self.ax.set_xlim(lb[0], ub[0])
        self.ax.set_ylim(lb[1], ub[1])
        self.ax.set_xlabel("x₁ (normalized)" if self.normalized else "x₁")
        self.ax.set_ylabel("x₂ (normalized)" if self.normalized else "x₂")
        self.ax.set_title("Bayesian Optimization")
        self.ax.legend(loc="upper right", fontsize=9)

        # Show the plot window
        plt.ion()  # Turn on interactive mode
        plt.show()

    def update(self, X_train: torch.Tensor, Y_train: torch.Tensor, iteration: int, total: int) -> None:
        """Update the plot with new data."""
        if not self.enabled or self.fig is None:
            return

        # Convert to plot space
        X_plot = X_train.detach().to(dtype=torch.double).cpu()

        xs = X_plot[:, 0].numpy()
        ys = X_plot[:, 1].numpy()
        n = len(xs)

        if n > 0:
            # Update sampled points
            colors = np.linspace(0, 1, n)
            self.artists['points'].set_offsets(np.column_stack([xs, ys]))
            self.artists['points'].set_array(colors)

            # Update path
            self.artists['path'].set_data(xs, ys)

            # Update best point
            y_vals = Y_train.detach().view(-1)
            best_idx = int(torch.argmin(y_vals))
            best_x = float(xs[best_idx])
            best_y = float(ys[best_idx])
            self.artists['best'].set_offsets(np.array([[best_x, best_y]]))

            # Update title with progress
            best_f = float(y_vals[best_idx].item())
            regret = best_f - float(self.bench.optimum_value)
            self.ax.set_title(
                f"Bayesian Optimization • iter {iteration}/{total} • "
                f"best f = {best_f:.4g} • regret = {regret:.3g}"
            )

        # Refresh the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def close(self) -> None:
        """Close the plot window."""
        if self.enabled and self.fig is not None:
            plt.close(self.fig)


def main():
    """Main execution function."""
    # Load benchmark module
    try:
        bench_mod = load_benchmarks_module()
    except (FileNotFoundError, ImportError) as e:
        print(f"Error loading benchmarks: {e}")
        sys.exit(1)

    # Parse arguments
    args = parse_arguments(bench_mod)

    # Set random seeds
    set_random_seeds(args.seed)

    # Build objective and bounds
    objective, bounds, bench = bench_mod.make_objective(
        name=args.benchmark,
        normalized=args.normalized,
        noise_std=float(args.noise_std)
    )

    # Generate initial points using Sobol sequence
    d = 2
    sobol = torch.quasirandom.SobolEngine(d, scramble=True, seed=args.seed)
    X_unit = sobol.draw(args.n_init).to(dtype=torch.double)

    if args.normalized:
        X0 = X_unit
    else:
        X0 = bench_mod.unnormalize(X_unit, bench.bounds.to(dtype=torch.double))

    Y0 = objective(X0)

    # Initialize Bayesian Optimization
    bo = BayesianOptimization(
        x_train=X0,
        y_train=Y0,
        bounds=bounds,
        maximize=False,  # All benchmarks are minimization problems
        use_outcome_transform=True
    )

    # Initialize live plotter if requested
    plotter = None
    if args.plot:
        plotter = LivePlotter(
            objective, bounds, bench,
            args.normalized,
            raw_bounds=bench.bounds.to(dtype=torch.double)
        )

    # Print starting message
    print(f"\n{'=' * 60}")
    print(f"Starting Bayesian Optimization")
    print(f"Benchmark: {bench.name}")
    print(f"Iterations: {args.iters} (+ {args.n_init} initial points)")
    print(f"Normalized space: {args.normalized}")
    print(f"Noise std: {args.noise_std}")
    print(f"{'=' * 60}\n")

    # Main optimization loop
    for i in range(1, args.iters + 1):
        # Get next point to evaluate
        x_new = bo.get_next_data_points(q=1)
        y_new = objective(x_new)

        # Update the model
        bo.update_model(x_new, y_new, use_outcome_transform=True)

        # Get current best
        y_vals = bo.y_train.detach().cpu().view(-1)
        best_idx = int(torch.argmin(y_vals))
        x_best = bo.x_train[best_idx]
        y_best = float(y_vals[best_idx].item())

        # Convert to raw space for display
        if args.normalized:
            x_best_raw = bench_mod.unnormalize(x_best, bench.bounds)
        else:
            x_best_raw = x_best

        # Print progress
        print(f"[{i:03d}/{args.iters}] y_new = {float(y_new.item()):+.6g} | "
              f"best = {y_best:.6g} at x* = [{x_best_raw[0]:.4f}, {x_best_raw[1]:.4f}]")

        # Update plot
        if plotter is not None:
            plotter.update(bo.x_train, bo.y_train, i, args.iters)

    # Final summary
    y_all = bo.y_train.detach().cpu().view(-1)
    final_idx = int(torch.argmin(y_all))
    x_final = bo.x_train[final_idx]
    y_final = float(y_all[final_idx].item())

    if args.normalized:
        x_final_raw = bench_mod.unnormalize(x_final, bench.bounds)
    else:
        x_final_raw = x_final

    # Calculate metrics
    dist_to_min = torch.cdist(
        x_final_raw.view(1, -1),
        bench.global_minima.to(dtype=torch.double)
    ).min().item()
    regret = y_final - float(bench.optimum_value)

    # Print summary
    print(f"\n{'=' * 60}")
    print("OPTIMIZATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Benchmark: {bench.name}")
    print(f"Best value found: {y_final:.6g}")
    print(f"Best location (raw): [{x_final_raw[0]:.6f}, {x_final_raw[1]:.6f}]")
    print(f"Known optimum: {bench.optimum_value:.6g}")
    print(f"Final regret: {regret:.6g}")
    print(f"Distance to nearest global minimum: {dist_to_min:.6g}")
    print(f"{'=' * 60}\n")

    # Keep plot open if it exists
    if plotter is not None and plotter.enabled:
        try:
            print("Plot window is open. Press Enter to close and exit...")
            input()
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            plotter.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nOptimization interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nError during optimization: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)