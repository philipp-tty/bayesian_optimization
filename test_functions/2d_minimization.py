from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple, List

import torch
from torch import Tensor


@dataclass(frozen=True)
class Benchmark2D:
    """
    Definition of a 2D benchmark function for minimization.
    - fn expects inputs X with shape (..., 2) in RAW (un-normalized) bounds and returns (..., 1).
    - bounds is a 2 x 2 tensor: [lower; upper] in RAW units.
    - global_minima is k x 2 in RAW units; optimum_value is the known global minimum value.
    """
    name: str
    fn: Callable[[Tensor], Tensor]
    bounds: Tensor  # shape (2, 2)
    global_minima: Tensor  # shape (k, 2)
    optimum_value: float


# ------------- helpers ------------- #

def _ensure_last_dim2(X: Tensor) -> Tensor:
    if X.shape[-1] != 2:
        raise ValueError(f"Expected last dim=2, got {X.shape}")
    return X


def _as_like(X: Tensor, v) -> Tensor:
    return torch.as_tensor(v, dtype=X.dtype, device=X.device)


def unnormalize(X01: Tensor, bounds: Tensor) -> Tensor:
    """
    Map X in [0,1]^2 to raw domain given bounds (2 x 2).
    """
    X01 = _ensure_last_dim2(X01)
    lower, upper = bounds[0], bounds[1]
    return lower + (upper - lower) * X01


def normalize(Xraw: Tensor, bounds: Tensor) -> Tensor:
    """
    Map X in raw domain to [0,1]^2 given bounds (2 x 2).
    """
    Xraw = _ensure_last_dim2(Xraw)
    lower, upper = bounds[0], bounds[1]
    return (Xraw - lower) / (upper - lower)


def _return_column(y: Tensor) -> Tensor:
    # Ensure shape (..., 1)
    if y.ndim == 0:
        return y.view(1, 1)
    if y.shape[-1] != 1:
        return y.unsqueeze(-1)
    return y


# ------------- benchmark functions (RAW domain) ------------- #

def branin(X: Tensor) -> Tensor:
    """
    Branin-Hoo function (2D), bounds: x in [-5, 10], y in [0, 15].
    Global minima f* ≈ 0.397887 at:
      (-π, 12.275), (π, 2.275), (3π, 2.475).
    """
    X = _ensure_last_dim2(X)
    x = X[..., 0]
    y = X[..., 1]
    a = _as_like(x, 1.0)
    b = _as_like(x, 5.1 / (4.0 * torch.pi**2))
    c = _as_like(x, 5.0 / torch.pi)
    r = _as_like(x, 6.0)
    s = _as_like(x, 10.0)
    t = _as_like(x, 1.0 / (8.0 * torch.pi))
    val = a * (y - b * x**2 + c * x - r) ** 2 + s * (1 - t) * torch.cos(x) + s
    return _return_column(val)


def rosenbrock_2d(X: Tensor) -> Tensor:
    """
    Rosenbrock function (2D), bounds: [-2, 2]^2, min at (1,1) with f*=0.
    """
    X = _ensure_last_dim2(X)
    x = X[..., 0]
    y = X[..., 1]
    val = 100.0 * (y - x**2) ** 2 + (1.0 - x) ** 2
    return _return_column(val)


def ackley_2d(X: Tensor) -> Tensor:
    """
    Ackley function (2D), bounds: [-5, 5]^2, min at (0,0) with f*=0.
    """
    X = _ensure_last_dim2(X)
    x = X[..., 0]
    y = X[..., 1]
    a = _as_like(x, 20.0)
    b = _as_like(x, 0.2)
    c = _as_like(x, 2.0 * torch.pi)
    term1 = -a * torch.exp(-b * torch.sqrt(0.5 * (x**2 + y**2)))
    term2 = -torch.exp(0.5 * (torch.cos(c * x) + torch.cos(c * y)))
    val = term1 + term2 + a + _as_like(x, torch.e)
    return _return_column(val)


def beale(X: Tensor) -> Tensor:
    """
    Beale function (2D), bounds: [-4.5, 4.5]^2, min at (3, 0.5) with f*=0.
    """
    X = _ensure_last_dim2(X)
    x = X[..., 0]
    y = X[..., 1]
    val = (1.5 - x + x * y) ** 2 + (2.25 - x + x * y**2) ** 2 + (2.625 - x + x * y**3) ** 2
    return _return_column(val)


def himmelblau(X: Tensor) -> Tensor:
    """
    Himmelblau function (2D), bounds: [-6, 6]^2, four global minima with f*=0:
      (3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
    """
    X = _ensure_last_dim2(X)
    x = X[..., 0]
    y = X[..., 1]
    val = (x**2 + y - 11.0) ** 2 + (x + y**2 - 7.0) ** 2
    return _return_column(val)


def goldstein_price(X: Tensor) -> Tensor:
    """
    Goldstein-Price function (2D), bounds: [-2, 2]^2, min at (0, -1) with f*=3.
    """
    X = _ensure_last_dim2(X)
    x = X[..., 0]
    y = X[..., 1]
    term1 = 1.0 + (x + y + 1.0) ** 2 * (19.0 - 14.0 * x + 3.0 * x**2 - 14.0 * y + 6.0 * x * y + 3.0 * y**2)
    term2 = 30.0 + (2.0 * x - 3.0 * y) ** 2 * (18.0 - 32.0 * x + 12.0 * x**2 + 48.0 * y - 36.0 * x * y + 27.0 * y**2)
    val = term1 * term2
    return _return_column(val)


def styblinski_tang_2d(X: Tensor) -> Tensor:
    """
    Styblinski–Tang (2D), bounds: [-5, 5]^2, min at (-2.903534, -2.903534) with f*≈ -78.332.
    f(x) = 0.5 * sum_i (x_i^4 - 16 x_i^2 + 5 x_i)
    """
    X = _ensure_last_dim2(X)
    x = X[..., 0]
    y = X[..., 1]
    g = lambda z: z**4 - 16.0 * z**2 + 5.0 * z
    val = 0.5 * (g(x) + g(y))
    return _return_column(val)


# ------------- registry and metadata ------------- #

def _t2(v: List[List[float]]) -> Tensor:
    return torch.tensor(v, dtype=torch.double)


BENCHMARKS: Dict[str, Benchmark2D] = {
    "branin": Benchmark2D(
        name="branin",
        fn=branin,
        bounds=_t2([[-5.0, 0.0], [10.0, 15.0]]),
        global_minima=_t2([
            [-float(torch.pi), 12.275],
            [float(torch.pi), 2.275],
            [3.0 * float(torch.pi), 2.475],
        ]),
        optimum_value=0.39788735772973816,
    ),
    "rosenbrock": Benchmark2D(
        name="rosenbrock",
        fn=rosenbrock_2d,
        bounds=_t2([[-2.0, -2.0], [2.0, 2.0]]),
        global_minima=_t2([[1.0, 1.0]]),
        optimum_value=0.0,
    ),
    "ackley": Benchmark2D(
        name="ackley",
        fn=ackley_2d,
        bounds=_t2([[-5.0, -5.0], [5.0, 5.0]]),
        global_minima=_t2([[0.0, 0.0]]),
        optimum_value=0.0,
    ),
    "beale": Benchmark2D(
        name="beale",
        fn=beale,
        bounds=_t2([[-4.5, -4.5], [4.5, 4.5]]),
        global_minima=_t2([[3.0, 0.5]]),
        optimum_value=0.0,
    ),
    "himmelblau": Benchmark2D(
        name="himmelblau",
        fn=himmelblau,
        bounds=_t2([[-6.0, -6.0], [6.0, 6.0]]),
        global_minima=_t2([
            [3.0, 2.0],
            [-2.805118, 3.131312],
            [-3.779310, -3.283186],
            [3.584428, -1.848126],
        ]),
        optimum_value=0.0,
    ),
    "goldstein_price": Benchmark2D(
        name="goldstein_price",
        fn=goldstein_price,
        bounds=_t2([[-2.0, -2.0], [2.0, 2.0]]),
        global_minima=_t2([[0.0, -1.0]]),
        optimum_value=3.0,
    ),
    "styblinski_tang": Benchmark2D(
        name="styblinski_tang",
        fn=styblinski_tang_2d,
        bounds=_t2([[-5.0, -5.0], [5.0, 5.0]]),
        global_minima=_t2([[-2.903534, -2.903534]]),
        optimum_value=-78.332,
    ),
}


# ------------- public API ------------- #

def available_benchmarks() -> List[str]:
    return sorted(BENCHMARKS.keys())


def get_benchmark(name: str) -> Benchmark2D:
    key = name.lower()
    if key not in BENCHMARKS:
        raise KeyError(f"Unknown benchmark '{name}'. Available: {available_benchmarks()}")
    return BENCHMARKS[key]


def make_objective(
    name: str,
    normalized: bool = True,
    noise_std: float = 0.0,
) -> Tuple[Callable[[Tensor], Tensor], Tensor, Benchmark2D]:
    """
    Create a BoTorch-ready objective function.
    - If normalized=True, the returned fn expects X in [0,1]^2 and raw bounds are mapped internally.
      Bounds returned will be [[0,0],[1,1]].
    - If normalized=False, fn expects X in RAW bounds and bounds returned are the raw bounds.
    - noise_std adds iid Gaussian noise to outputs (useful for robustness tests).
    Returns: (fn, bounds, benchmark_metadata)
    """
    bench = get_benchmark(name)
    raw_bounds = bench.bounds.clone()

    if normalized:
        bounds = torch.stack([torch.zeros(2, dtype=raw_bounds.dtype),
                              torch.ones(2, dtype=raw_bounds.dtype)], dim=0)

        def fn(X: Tensor) -> Tensor:
            X_raw = unnormalize(X, raw_bounds.to(device=X.device, dtype=X.dtype))
            y = bench.fn(X_raw)
            if noise_std and noise_std > 0:
                y = y + noise_std * torch.randn_like(y)
            return y

        return fn, bounds, bench

    else:
        def fn(X: Tensor) -> Tensor:
            y = bench.fn(X)
            if noise_std and noise_std > 0:
                y = y + noise_std * torch.randn_like(y)
            return y

        return fn, raw_bounds, bench


__all__ = [
    "Benchmark2D",
    "available_benchmarks",
    "get_benchmark",
    "make_objective",
    # raw functions
    "branin",
    "rosenbrock_2d",
    "ackley_2d",
    "beale",
    "himmelblau",
    "goldstein_price",
    "styblinski_tang_2d",
    # helpers
    "normalize",
    "unnormalize",
]
