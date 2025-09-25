# Bayesian Optimization on 2D Benchmark Functions

Lightweight framework to run Bayesian Optimization (BO) experiments on classic 2D test functions (Branin, Himmelblau, Ackley, etc.) with:
- Normalized or raw search space
- Optional Gaussian observation noise
- Live visualization (objective + GP posterior mean + uncertainty)
- Batch runs across all benchmarks
- Reproducible Sobol initialization
- Clean, PEP-compliant code

---

## Benchmarks Included

branin, rosenbrock, ackley, beale, himmelblau, goldstein_price, styblinski_tang, booth, rastrigin

All are minimization problems with known optima stored in `test_functions/2d_minimization.py`.

---

## Installation

Minimal environment (Python 3.10+):

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install torch botorch gpytorch numpy matplotlib
```

(Install CUDA-enabled PyTorch from https://pytorch.org if you have a GPU.)

---

## Quick Start

Single benchmark (normalized, with plotting):
```bash
python main.py --benchmark himmelblau --iters 25 --plot
```

All benchmarks (headless):
```bash
python main.py --all --iters 30
```

Save plots to files (grid for all benchmarks, single live figure):
```bash
# All benchmarks, save grid (no GUI needed)
python main.py --all --iters 30 --save-plot outputs/all_benchmarks_grid.png

# Single benchmark with live plotting + save final panel figure
python main.py --benchmark himmelblau --iters 25 --plot --save-plot outputs/himmelblau_run.png
```

More examples:
```bash
# Different initial design size + seed
python main.py --benchmark branin --iters 40 --n-init 12 --seed 123

# Add output noise
python main.py --benchmark rastrigin --iters 50 --noise-std 0.01

# Work in raw coordinate space
python main.py --benchmark ackley --iters 30 --no-normalized

# Dense run with visualization
python main.py --benchmark styblinski_tang --iters 60 --plot
```

Interrupt safely with Ctrl+C.

---

## CLI Arguments (core)

| Flag | Meaning |
|------|---------|
| --benchmark NAME | One benchmark (ignored if --all) |
| --all | Run every benchmark sequentially |
| --iters N | BO acquisition iterations (excludes initial Sobol points) |
| --n-init N | Initial Sobol samples |
| --normalized / --no-normalized | Toggle [0,1]^2 parameterization |
| --noise-std S | Add iid Gaussian noise to observations |
| --plot | Enable live plotting (single) or final grid (multi-run) |
| --seed SEED | Reproducibility |
| --save-plot PATH | Save final plot (grid or live figure) to PATH (PNG/PDF). |

---

## Algorithm

Each iteration:
1. Fit (or refit) a `SingleTaskGP` to all collected data.
2. Use `LogExpectedImprovement` (analytic, q=1) to pick the next point.
3. Append observation and repeat.

Refitting each step ensures outcome transforms stay calibrated.

---

## Output

Console logs:
- Iteration, new y value
- Current best value + location (raw coordinates if normalized mode)

Summary includes best value, regret, and distance to nearest known global minimum.

When `--plot`:
- Panel 1: True objective with sample chronology (red → green) + known minima
- Panel 2: GP posterior mean (shared color scale)
- Panel 3: GP posterior standard deviation (uncertainty)

When `--save-plot PATH`:
- Multi-run (`--all`): saves the benchmark grid to PATH (shown only if also `--plot`).
- Single run (`--plot`): saves the final live figure automatically (also on close / Ctrl+C).

---

## Reproducibility

Sobol initialization and PyTorch/NumPy RNGs are seeded via `--seed`. For full determinism on GPU you may also set:

```bash
export CUBLAS_WORKSPACE_CONFIG=:16:8
export PYTHONHASHSEED=0
```

---

## Adding a New Benchmark

1. Implement a raw-domain function in `test_functions/2d_minimization.py` taking `Tensor (...,2)` → `(...,1)`.
2. Add metadata to `BENCHMARKS`:
```python
"myfunc": Benchmark2D(
    name="myfunc",
    fn=myfunc_fn,
    bounds=torch.tensor([[x1_low, x2_low],[x1_high, x2_high]], dtype=torch.double),
    global_minima=torch.tensor([[x1*, x2*]], dtype=torch.double),
    optimum_value=KNOWN_F_STAR,
),
```
3. It becomes available automatically via `--benchmark myfunc`.

---

## Tips

- Use `--no-normalized` if you want to visualize behavior in raw units directly.
- Increase `--n-init` on highly multi-modal surfaces (e.g., Rastrigin).
- Add slight noise (e.g., 0.01) to test robustness of surrogate modeling.
- GPU usage is automatic if CUDA is available.

---

## License

Add a license of your choice (e.g., MIT) here.

---

## Citation

If you extend this for research, cite BoTorch:
https://botorch.org

---

Happy optimizing!
