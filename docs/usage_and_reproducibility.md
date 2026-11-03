# Usage and reproducibility

## Local setup

Python >=3.10 and a CPU version of PyTorch are sufficient. Install locally in editable mode with `python -m pip install -e '.[dev,viz]'`. No network access or dataset download is required **after dependencies are installed**. Do not upload or publish this draft without authorization.

## Code, tests and computation

```bash
python examples/quickstart_torch.py
python -m pytest -q
python scripts/make_figures.py --out figures
```

The PyTorch implementation uses raw embeddings, shared trainable class proxies, squared Euclidean distances, Cauchy affinity logits, and population variance over unordered distinct pairs. Tests use float64, independent numerical calculations, known exact configurations, analytic gradients, and finite differences.

## Figure policy

All current figures are **synthetic toy calculations or analytic constructions**. They are not MNIST, CUB, Cars, or experimentally measured test-set results.

- Figure 1: two **matched** objective runs on fixed 3D toy data. Same data, initial class proxies, optimizer, steps, seed, and 3D plot limits. MES coefficient is the only optimized-condition difference. Proxies move; embeddings do not.
- Figure 2: explanatory vector flowchart, no experimental data.
- Figure 3: ideal four-point regular tetrahedron with known coordinates and six equal squared distances.
- Figure 4: six unordered squared pair distances computed from the two Figure 1 proxy sets; values match the MES formula.
- Figure 5: analytic tetrahedron and illustrative octahedron configurations in 3D with explicit distinct pairwise squared distances. Not claimed as a learned optimum.

Regenerate by running `scripts/make_figures.py`. Fixed seed/optimizer values are embedded in that script and also written to `figures/manifest.json`. SVG files are accompanied by high-resolution PNG alternatives.

## Optional MNIST cluster-film protocol — script available, real-data run pending

A self-contained, scientifically controlled educational video protocol is now
implemented in `scripts/cluster_movie.py`. It records 3D encoder outputs and
proxy weights throughout genuine paired training, using the same seeded MNIST
*train* holdout and minibatch schedule in both arms. It saves raw NPZ
snapshots, a metric history, the latest live preview, and optionally an MP4.
The full protocol is in [cluster movie guide](cluster_movie_guide.md).

Its complete offline procedural-shapes path has been tested end-to-end, but
no downloaded MNIST training has been executed in this isolated environment.
Dataset access, permissions, training convergence, and reproducibility on
the eventual target machine require local validation. Results must not be
transplanted or inferred from private research runs.

## Limits of reproduction

Floating-point results can vary slightly across PyTorch versions and CPUs. The synthetic tests use wide enough tolerances for numerically stable float64 calculations. Actual benchmark parity, TensorFlow parity, and pretrained checkpoints are explicitly out of scope for this first draft.