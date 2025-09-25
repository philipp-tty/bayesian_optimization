#!/usr/bin/env python3
"""Run Bayesian Optimization on 2D benchmark functions with tidy comments and PEP-compliant style.

This script can optimize a single benchmark or all available ones, optionally
showing live plots of the true objective, posterior mean, and posterior
uncertainty during the run.

Usage examples
--------------
Single benchmark (normalized, with plotting):
    python bayesopt_2d_cleaned.py --benchmark himmelblau --iters 25 --plot

All benchmarks (no plots):
    python bayesopt_2d_cleaned.py --all --iters 40
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch

from algorithms.bayesian_optimization import BayesianOptimization


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------

def ensure_interactive_backend() -> bool:
    """Ensure a GUI backend is selected for matplotlib.

    Returns
    -------
    bool
        True if an interactive backend is available, False otherwise.
    """
    current_backend = matplotlib.get_backend()
    if current_backend and "agg" not in current_backend.lower():
        return True

    for backend in ["QtAgg", "Qt5Agg", "TkAgg", "GTKAgg", "WXAgg", "MacOSX"]:
        try:
            matplotlib.use(backend, force=True)
            print(f"Switched to {backend} backend for plotting")
            return True
        except (ImportError, ValueError):
            continue

    print("Warning: No interactive matplotlib backend available. Plots will not be shown.")
    return False


def setup_plot_style() -> None:
    """Configure matplotlib with a modern, readable style."""
    for style in ["seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"]:
        try:
            plt.style.use(style)
            break
        except OSError:
            continue

    matplotlib.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "axes.edgecolor": "#333333",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.size": 10,
            "figure.figsize": (8, 6),
        }
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_random_seeds(seed: int) -> None:
    """Seed NumPy and PyTorch for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_benchmarks_module():
    """Dynamically import the 2D benchmark module (test_functions/2d_minimization.py)."""
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
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run Bayesian Optimization on a 2D benchmark function.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--benchmark",
        type=str,
        default="himmelblau",
        choices=bench_mod.available_benchmarks(),
        help="Benchmark function to optimize.",
    )
    parser.add_argument("--iters", type=int, default=20, help="Number of BO iterations (acquisitions).")
    parser.add_argument("--n-init", type=int, default=8, help="Number of initial Sobol samples.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help="Standard deviation of i.i.d. Gaussian noise on outputs.",
    )
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument(
        "--normalized",
        action="store_true",
        default=True,
        help="Work in normalized [0,1]^2 space.",
    )
    norm_group.add_argument("--no-normalized", dest="normalized", action="store_false", help="Work in raw bounds.")
    parser.add_argument("--plot", action="store_true", help="Show live optimization plots.")
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Run all benchmarks (ignores --benchmark). If --plot is set, show them together in one grid figure."
        ),
    )
    parser.add_argument(
        "--save-plot",
        type=str,
        default=None,
        metavar="PATH",
        help="Save final plot (grid or live figure) to this file. Creates parent dirs.",
    )

    return parser.parse_args()


def build_grid(
    bounds: torch.Tensor, *, resolution: int = 220, dtype: torch.dtype = torch.double
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a 2D grid over the rectangular bounds for contour plotting.

    Parameters
    ----------
    bounds
        Tensor of shape (2, 2) with lower/upper bounds.
    resolution
        Number of grid steps per axis.
    dtype
        Torch dtype used for the grid.

    Returns
    -------
    (X_grid, Y_grid, XY_flat)
        Meshgrids and the flattened N×2 coordinates in row-major order.
    """
    lb = bounds[0].to(dtype=dtype).cpu()
    ub = bounds[1].to(dtype=dtype).cpu()

    xs = torch.linspace(float(lb[0]), float(ub[0]), resolution, dtype=dtype)
    ys = torch.linspace(float(lb[1]), float(ub[1]), resolution, dtype=dtype)

    X_grid, Y_grid = torch.meshgrid(xs, ys, indexing="ij")
    XY_flat = torch.stack([X_grid.reshape(-1), Y_grid.reshape(-1)], dim=-1)
    return X_grid, Y_grid, XY_flat


def normalize_points(X: torch.Tensor, raw_bounds: torch.Tensor) -> torch.Tensor:
    """Map points from raw space to [0,1]^d given raw bounds."""
    lb = raw_bounds[0].to(dtype=X.dtype, device=X.device)
    ub = raw_bounds[1].to(dtype=X.dtype, device=X.device)
    return (X - lb) / (ub - lb + 1e-12)


# Reusable red→green colormap for sample chronology (old→new)
RED_GREEN_CMAP = mcolors.LinearSegmentedColormap.from_list("red_green", ["#ff0000", "#06d6a0"])


# ---------------------------------------------------------------------------
# Static grid plot for multiple benchmarks
# ---------------------------------------------------------------------------

def plot_benchmarks_grid(results, normalized: bool, bench_mod, save_path: Optional[str] = None, show: bool = True) -> None:
    """Plot benchmark surfaces with sampled points and best locations in one figure.

    Parameters
    ----------
    results
        Iterable of dicts with keys: ``name``, ``bench``, ``X``, ``Y``.
    normalized
        Whether the coordinates in ``X`` are already normalized.
    bench_mod
        Benchmark module with ``unnormalize`` helper.
    save_path
        Optional path to save the figure (PNG/PDF etc.).
    show
        Whether to display the plot interactively.
    """
    if not ensure_interactive_backend() and not save_path:
        print("Warning: No interactive matplotlib backend available. Skipping plot.")
        return

    setup_plot_style()

    n = len(results)
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows), constrained_layout=True)
    axes = np.array(axes).reshape(-1)

    for idx, res in enumerate(results):
        ax = axes[idx]
        bench = res["bench"]
        X_hist = res["X"].detach().to(dtype=torch.double).cpu()
        Y_hist = res["Y"].detach().to(dtype=torch.double).cpu().view(-1)

        bounds_plot = (
            torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.double) if normalized else bench.bounds.to(torch.double)
        )

        Xg, Yg, XY = build_grid(bounds_plot, resolution=220)
        with torch.no_grad():
            XY_raw = bench_mod.unnormalize(XY, bench.bounds.to(torch.double)) if normalized else XY
            Z = bench.fn(XY_raw).view(Xg.shape[0], Xg.shape[1]).cpu().numpy()

        cont = ax.contourf(Xg.numpy(), Yg.numpy(), Z, levels=30, cmap="cividis", alpha=0.95)
        cbar = fig.colorbar(cont, ax=ax, pad=0.01, shrink=0.9)
        cbar.set_label("f(x)", fontsize=9)

        mins_raw = bench.global_minima.to(dtype=torch.double)
        mins_plot = normalize_points(mins_raw, bench.bounds.to(torch.double)) if normalized else mins_raw
        ax.scatter(
            mins_plot[:, 0].cpu(),
            mins_plot[:, 1].cpu(),
            marker="X",
            s=70,
            c="#06d6a0",
            edgecolors="#1b1e23",
            linewidths=0.6,
            zorder=5,
        )

        if X_hist.numel() > 0:
            xs, ys = X_hist[:, 0].numpy(), X_hist[:, 1].numpy()
            colors = np.linspace(0, 1, len(xs))
            ax.scatter(
                xs,
                ys,
                c=colors,
                cmap=RED_GREEN_CMAP,
                vmin=0.0,
                vmax=1.0,
                s=26,
                edgecolors="#ffffff",
                linewidths=0.5,
                zorder=4,
            )
            ax.plot(xs, ys, color="#ffffff", lw=1.0, alpha=0.55, zorder=3)

            best_idx = int(torch.argmin(Y_hist))
            ax.scatter(
                [xs[best_idx]],
                [ys[best_idx]],
                marker="*",
                s=120,
                c="#FFD166",
                edgecolors="#1b1e23",
                linewidths=0.6,
                zorder=6,
            )

            regret = float(Y_hist[best_idx].item()) - float(bench.optimum_value)
            ax.set_title(f"{bench.name} • best={float(Y_hist[best_idx]):.4g} • regret={regret:.3g}")
        else:
            ax.set_title(f"{bench.name}")

        lb, ub = bounds_plot[0].cpu().numpy(), bounds_plot[1].cpu().numpy()
        ax.set_xlim(lb[0], ub[0])
        ax.set_ylim(lb[1], ub[1])
        ax.set_xlabel("x1 (normalized)" if normalized else "x1")
        ax.set_ylabel("x2 (normalized)" if normalized else "x2")

    for j in range(idx + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Bayesian Optimization across Benchmarks", fontsize=12)
    if save_path:
        try:
            out_path = Path(save_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, bbox_inches="tight")
            print(f"Saved benchmark grid figure to: {out_path}")
        except Exception as e:
            print(f"Failed to save figure to {save_path}: {e}")

    if show and ensure_interactive_backend():
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Live plotting during optimization
# ---------------------------------------------------------------------------

class LivePlotter:
    """Handle live plotting of objective and GP posterior during optimization."""

    def __init__(
        self,
        objective,
        bounds_plot_space: torch.Tensor,
        bench,
        normalized: bool,
        raw_bounds: torch.Tensor,
        posterior_fn: Optional[Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        self.objective = objective
        self.bench = bench
        self.normalized = normalized
        self.raw_bounds = raw_bounds
        self.fig = None
        self.ax = None
        self.ax_mean = None
        self.ax_std = None
        self.artists: dict = {}
        # plotting handles
        self.artists["obj_contour"] = None
        self.artists["f_cbar"] = None  # shared colorbar for objective + mean
        self.artists["mean_contour"] = None
        self.artists["std_img"] = None
        self.artists["std_cbar"] = None
        # shared normalization/levels for objective + mean
        self.norm_f = None
        self.obj_levels = None
        self.posterior_fn = posterior_fn
        self.surr_enabled = False
        self.save_path = save_path

        if not ensure_interactive_backend():
            self.enabled = False
            return

        self.enabled = True
        setup_plot_style()
        self._initialize_plot(bounds_plot_space)
        if self.posterior_fn is not None:
            try:
                self._initialize_surrogate_plots()
                self.surr_enabled = True
            except Exception as e:  # pragma: no cover – plotting failure shouldn't crash runs
                warnings.warn(f"Could not initialize mean/uncertainty plots: {e}")
                self.surr_enabled = False

    def _initialize_plot(self, bounds_plot_space: torch.Tensor) -> None:
        """Set up the initial figure with the true objective on the first axis."""
        X_grid, Y_grid, XY_flat = build_grid(bounds_plot_space, resolution=220)
        self.X_grid, self.Y_grid, self.XY_flat = X_grid, Y_grid, XY_flat

        with torch.no_grad():
            Z = self.objective(XY_flat.to(dtype=torch.double)).view(X_grid.shape[0], X_grid.shape[1]).cpu().numpy()

        self.fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        self.ax, self.ax_mean, self.ax_std = axes[0], axes[1], axes[2]

        # Objective: draw with fixed norm and store levels for reuse in mean plot
        vmin, vmax = float(np.nanmin(Z)), float(np.nanmax(Z))
        self.norm_f = mcolors.Normalize(vmin=vmin, vmax=vmax)
        contour = self.ax.contourf(
            X_grid.numpy(), Y_grid.numpy(), Z, levels=30, cmap="cividis", alpha=0.95, norm=self.norm_f
        )
        self.obj_levels = contour.levels
        self.artists["obj_contour"] = contour

        # One shared colorbar for objective + mean
        f_cbar = self.fig.colorbar(contour, ax=[self.ax, self.ax_mean], pad=0.01)
        f_cbar.set_label("f(x) and GP mean", fontsize=10)
        self.artists["f_cbar"] = f_cbar

        mins_raw = self.bench.global_minima.to(dtype=torch.double)
        mins_plot = normalize_points(mins_raw, self.raw_bounds) if self.normalized else mins_raw
        self.ax.scatter(
            mins_plot[:, 0].cpu(),
            mins_plot[:, 1].cpu(),
            marker="X",
            s=90,
            c="#06d6a0",
            edgecolors="#1b1e23",
            linewidths=0.6,
            zorder=5,
            label="Known minima",
        )

        self.artists["points"] = self.ax.scatter(
            [], [], c=[], cmap=RED_GREEN_CMAP, vmin=0.0, vmax=1.0, s=36, edgecolors="#ffffff", linewidths=0.6, zorder=4
        )
        (self.artists["path"],) = self.ax.plot([], [], color="#ffffff", lw=1.1, alpha=0.55, zorder=3)
        self.artists["best"] = self.ax.scatter(
            [], [], marker="*", s=150, c="#FFD166", edgecolors="#1b1e23", linewidths=0.7, zorder=6, label="Current best"
        )

        lb = bounds_plot_space[0].cpu().numpy()
        ub = bounds_plot_space[1].cpu().numpy()
        for ax in (self.ax, self.ax_mean, self.ax_std):
            ax.set_xlim(lb[0], ub[0])
            ax.set_ylim(lb[1], ub[1])
            ax.set_xlabel("x1 (normalized)" if self.normalized else "x1")
            ax.set_ylabel("x2 (normalized)" if self.normalized else "x2")

        self.ax_mean.grid(False)
        self.ax_std.grid(False)

        self.ax.set_title("Objective")
        self.ax.legend(loc="upper right", fontsize=9)
        self.ax_mean.set_title("GP Posterior Mean")
        self.ax_std.set_title("GP Posterior Uncertainty")

        plt.ion()
        plt.show()

    def _initialize_surrogate_plots(self) -> None:
        """Populate posterior mean and std subplots."""
        if self.posterior_fn is None or self.X_grid is None:
            return

        with torch.no_grad():
            mean, std = self.posterior_fn(self.XY_flat)
            M = mean.view(self.X_grid.shape[0], self.X_grid.shape[1]).detach().cpu().numpy()
            S = std.view(self.X_grid.shape[0], self.X_grid.shape[1]).detach().cpu().numpy()

        # Posterior mean: identical style, shared norm/levels with objective
        mean_contour = self.ax_mean.contourf(
            self.X_grid.numpy(),
            self.Y_grid.numpy(),
            M,
            levels=(self.obj_levels if self.obj_levels is not None else 30),
            cmap="cividis",
            alpha=0.95,
            norm=self.norm_f,
        )
        self.artists["mean_contour"] = mean_contour

        # Posterior std: keep imshow + its own colorbar
        x_min, x_max = float(self.X_grid.min()), float(self.X_grid.max())
        y_min, y_max = float(self.Y_grid.min()), float(self.Y_grid.max())
        extent = [x_min, x_max, y_min, y_max]

        std_img = self.ax_std.imshow(S.T, extent=extent, origin="lower", cmap="magma", aspect="auto")
        std_cbar = self.fig.colorbar(std_img, ax=self.ax_std, pad=0.01)
        std_cbar.set_label("Posterior std (uncertainty)", fontsize=10)
        self.artists["std_img"] = std_img
        self.artists["std_cbar"] = std_cbar

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, X_train: torch.Tensor, Y_train: torch.Tensor, iteration: int, total: int) -> None:
        """Update dynamic artists and (if available) the GP posterior plots."""
        if not self.enabled or self.fig is None:
            return

        X_plot = X_train.detach().to(dtype=torch.double).cpu()
        xs, ys = X_plot[:, 0].numpy(), X_plot[:, 1].numpy()
        n = len(xs)

        if n > 0:
            colors = np.linspace(0, 1, n)
            self.artists["points"].set_offsets(np.column_stack([xs, ys]))
            self.artists["points"].set_array(colors)

            self.artists["path"].set_data(xs, ys)

            y_vals = Y_train.detach().view(-1)
            best_idx = int(torch.argmin(y_vals))
            self.artists["best"].set_offsets(np.array([[float(xs[best_idx]), float(ys[best_idx])]]))

            best_f = float(y_vals[best_idx].item())
            regret = best_f - float(self.bench.optimum_value)
            self.ax.set_title(f"Objective • iter {iteration}/{total} • best f = {best_f:.4g} • regret = {regret:.3g}")

        # Initialize surrogate plots if not present yet
        needs_init = (
            self.posterior_fn is not None
            and self.XY_flat is not None
            and (not self.surr_enabled or self.artists.get("mean_contour") is None or self.artists.get("std_img") is None)
        )
        if needs_init:
            try:
                self._initialize_surrogate_plots()
                self.surr_enabled = True
            except Exception:
                pass  # try again on a later iteration

        if self.surr_enabled and self.posterior_fn is not None and self.XY_flat is not None:
            try:
                with torch.no_grad():
                    mean, std = self.posterior_fn(self.XY_flat)
                    M = mean.view(self.X_grid.shape[0], self.X_grid.shape[1]).detach().cpu().numpy()
                    S = std.view(self.X_grid.shape[0], self.X_grid.shape[1]).detach().cpu().numpy()

                # Update posterior mean: robustly remove old and redraw with shared norm/levels
                old = self.artists.get("mean_contour")
                if old is not None:
                    try:
                        for coll in getattr(old, "collections", []):
                            coll.remove()
                    except Exception:
                        # Fallback: clear axis and re-apply formatting
                        self.ax_mean.cla()
                        x_min, x_max = float(self.X_grid.min()), float(self.X_grid.max())
                        y_min, y_max = float(self.Y_grid.min()), float(self.Y_grid.max())
                        self.ax_mean.set_xlim(x_min, x_max)
                        self.ax_mean.set_ylim(y_min, y_max)
                        self.ax_mean.set_xlabel("x1 (normalized)" if self.normalized else "x1")
                        self.ax_mean.set_ylabel("x2 (normalized)" if self.normalized else "x2")
                        self.ax_mean.grid(False)

                levels = self.obj_levels if self.obj_levels is not None else 30
                new_mean_contour = self.ax_mean.contourf(
                    self.X_grid.numpy(),
                    self.Y_grid.numpy(),
                    M,
                    levels=levels,
                    cmap="cividis",
                    alpha=0.95,
                    norm=self.norm_f,
                )
                self.artists["mean_contour"] = new_mean_contour
                self.ax_mean.set_title(f"GP Posterior Mean • iter {iteration}/{total}")

                # Update posterior std (imshow) + its own colorbar
                std_img = self.artists.get("std_img")
                if std_img is not None:
                    std_img.set_data(S.T)
                    s_min, s_max = float(np.nanmin(S)), float(np.nanmax(S))
                    if s_min != s_max:
                        std_img.set_clim(s_min, s_max)
                    if self.artists.get("std_cbar") is not None:
                        self.artists["std_cbar"].update_normal(std_img)
                    self.ax_std.set_title(f"GP Posterior Uncertainty • iter {iteration}/{total}")
            except Exception as e:  # pragma: no cover
                warnings.warn(f"Failed to update mean/uncertainty plots: {e}")

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)

    def save(self, path: Optional[str] = None) -> None:
        """Save current figure to disk."""
        if self.fig is None:
            return
        target = path or self.save_path
        if not target:
            return
        try:
            p = Path(target)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(p, bbox_inches="tight")
            print(f"Saved live optimization figure to: {p}")
        except Exception as e:
            print(f"Failed to save live plot to {target}: {e}")

    def close(self) -> None:
        # Before closing, auto-save if requested
        if self.save_path:
            self.save(self.save_path)
        if self.enabled and self.fig is not None:
            plt.close(self.fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for running experiments."""
    try:
        bench_mod = load_benchmarks_module()
    except (FileNotFoundError, ImportError) as e:
        print(f"Error loading benchmarks: {e}")
        sys.exit(1)

    args = parse_arguments(bench_mod)

    # ----------------------------- ALL BENCHMARKS ----------------------------
    if args.all:
        set_random_seeds(args.seed)
        results = []
        names = bench_mod.available_benchmarks()
        print(f"\n{'=' * 60}")
        print(f"Running ALL benchmarks: {', '.join(names)}")
        print(f"Iterations: {args.iters} (+ {args.n_init} initial)")
        print(f"Normalized space: {args.normalized} | Noise std: {args.noise_std}")
        print(f"{'=' * 60}\n")

        for name in names:
            objective, bounds, bench = bench_mod.make_objective(
                name=name, normalized=args.normalized, noise_std=float(args.noise_std)
            )

            d = 2
            sobol = torch.quasirandom.SobolEngine(d, scramble=True, seed=args.seed)
            X_unit = sobol.draw(args.n_init).to(dtype=torch.double)
            X0 = X_unit if args.normalized else bench_mod.unnormalize(X_unit, bench.bounds.to(torch.double))
            Y0 = objective(X0)

            bo = BayesianOptimization(
                x_train=X0,
                y_train=Y0,
                bounds=bounds,
                maximize=False,
                use_outcome_transform=True,
            )

            for i in range(1, args.iters + 1):
                x_new = bo.get_next_data_points(q=1)
                y_new = objective(x_new)
                bo.update_model(x_new, y_new, use_outcome_transform=True)

                y_vals = bo.y_train.detach().cpu().view(-1)
                best_idx = int(torch.argmin(y_vals))
                x_best = bo.x_train[best_idx]
                y_best = float(y_vals[best_idx].item())
                x_best_raw = bench_mod.unnormalize(x_best, bench.bounds) if args.normalized else x_best

                print(
                    f"[{name:>16s} {i:03d}/{args.iters}] y_new={float(y_new.item()):+.6g} | "
                    f"best={y_best:.6g} at x*=[{x_best_raw[0]:.4f}, {x_best_raw[1]:.4f}]"
                )

            results.append({"name": bench.name, "bench": bench, "X": bo.x_train, "Y": bo.y_train})

            y_all = bo.y_train.detach().cpu().view(-1)
            final_idx = int(torch.argmin(y_all))
            x_final = bo.x_train[final_idx]
            y_final = float(y_all[final_idx].item())
            x_final_raw = bench_mod.unnormalize(x_final, bench.bounds) if args.normalized else x_final
            dist_to_min = (
                torch.cdist(x_final_raw.view(1, -1), bench.global_minima.to(dtype=torch.double)).min().item()
            )
            regret = y_final - float(bench.optimum_value)
            print(
                f"Summary [{name}]: best={y_final:.6g}, x*=[{x_final_raw[0]:.6f}, {x_final_raw[1]:.6f}], "
                f"f*={bench.optimum_value:.6g}, regret={regret:.6g}, dist-to-min={dist_to_min:.6g}\n"
            )

        if args.plot or args.save_plot:
            plot_benchmarks_grid(
                results,
                args.normalized,
                bench_mod,
                save_path=args.save_plot,
                show=args.plot,
            )
        return

    # ------------------------------ SINGLE BENCH ------------------------------
    set_random_seeds(args.seed)

    objective, bounds, bench = bench_mod.make_objective(
        name=args.benchmark, normalized=args.normalized, noise_std=float(args.noise_std)
    )

    d = 2
    sobol = torch.quasirandom.SobolEngine(d, scramble=True, seed=args.seed)
    X_unit = sobol.draw(args.n_init).to(dtype=torch.double)
    X0 = X_unit if args.normalized else bench_mod.unnormalize(X_unit, bench.bounds.to(torch.double))
    Y0 = objective(X0)

    bo = BayesianOptimization(
        x_train=X0,
        y_train=Y0,
        bounds=bounds,
        maximize=False,  # All benchmarks are minimization problems
        use_outcome_transform=True,
    )

    def make_posterior_fn(bo_inst):
        """Return a function that maps X -> (posterior_mean, posterior_std)."""

        def posterior_fn(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            device = getattr(bo_inst.x_train, "device", torch.device("cpu"))
            dtype = getattr(bo_inst.x_train, "dtype", torch.double)
            Xd = X.to(device=device, dtype=dtype)
            with torch.no_grad():
                model = getattr(bo_inst, "model", None) or getattr(bo_inst, "gp", None)
                if model is not None:
                    try:
                        post = model.posterior(Xd)
                        mean = post.mean.squeeze(-1)
                        std = post.variance.clamp_min(0.0).sqrt().squeeze(-1)
                        return mean.to(torch.double).cpu(), std.to(torch.double).cpu()
                    except Exception:
                        pass
                predict = getattr(bo_inst, "predict", None)
                if callable(predict):
                    out = predict(Xd)
                    if isinstance(out, (tuple, list)) and len(out) >= 2:
                        m, v = out[0], out[1]
                        std = v.clamp_min(0.0).sqrt()
                        return m.to(torch.double).cpu().squeeze(-1), std.to(torch.double).cpu().squeeze(-1)
                    if isinstance(out, dict) and "mean" in out:
                        m = out["mean"]
                        if "std" in out:
                            s = out["std"]
                        else:
                            v = out.get("var", out.get("variance"))
                            s = v.clamp_min(0.0).sqrt()
                        return m.to(torch.double).cpu().squeeze(-1), s.to(torch.double).cpu().squeeze(-1)
                get_post = getattr(bo_inst, "get_posterior", None)
                if callable(get_post):
                    post = get_post(Xd)
                    if hasattr(post, "mean") and hasattr(post, "variance"):
                        m = post.mean.squeeze(-1)
                        s = post.variance.clamp_min(0.0).sqrt().squeeze(-1)
                        return m.to(torch.double).cpu(), s.to(torch.double).cpu()
                    if isinstance(post, (tuple, list)) and len(post) >= 2:
                        m, v = post[0], post[1]
                        return (
                            m.to(torch.double).cpu().squeeze(-1),
                            v.clamp_min(0.0).sqrt().to(torch.double).cpu().squeeze(-1),
                        )
            raise RuntimeError("Cannot compute posterior from the current BO model.")

        return posterior_fn

    plotter = None
    if args.plot:
        plotter = LivePlotter(
            objective,
            bounds,
            bench,
            args.normalized,
            raw_bounds=bench.bounds.to(dtype=torch.double),
            posterior_fn=make_posterior_fn(bo),
            save_path=args.save_plot,
        )

    print(f"\n{'=' * 60}")
    print("Starting Bayesian Optimization")
    print(f"Benchmark: {bench.name}")
    print(f"Iterations: {args.iters} (+ {args.n_init} initial points)")
    print(f"Normalized space: {args.normalized}")
    print(f"Noise std: {args.noise_std}")
    print(f"{'=' * 60}\n")

    for i in range(1, args.iters + 1):
        x_new = bo.get_next_data_points(q=1)
        y_new = objective(x_new)
        bo.update_model(x_new, y_new, use_outcome_transform=True)

        y_vals = bo.y_train.detach().cpu().view(-1)
        best_idx = int(torch.argmin(y_vals))
        x_best = bo.x_train[best_idx]
        y_best = float(y_vals[best_idx].item())
        x_best_raw = bench_mod.unnormalize(x_best, bench.bounds) if args.normalized else x_best

        print(
            f"[{i:03d}/{args.iters}] y_new = {float(y_new.item()):+.6g} | "
            f"best = {y_best:.6g} at x* = [{x_best_raw[0]:.4f}, {x_best_raw[1]:.4f}]"
        )

        if plotter is not None:
            plotter.update(bo.x_train, bo.y_train, i, args.iters)

    y_all = bo.y_train.detach().cpu().view(-1)
    final_idx = int(torch.argmin(y_all))
    x_final = bo.x_train[final_idx]
    y_final = float(y_all[final_idx].item())
    x_final_raw = bench_mod.unnormalize(x_final, bench.bounds) if args.normalized else x_final

    dist_to_min = torch.cdist(x_final_raw.view(1, -1), bench.global_minima.to(dtype=torch.double)).min().item()
    regret = y_final - float(bench.optimum_value)

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

    if plotter is not None and plotter.enabled:
        if args.save_plot:
            # Ensure final state is saved before user closes window
            plotter.save(args.save_plot)
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
