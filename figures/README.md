# Reproducible figure inventory

Figures 01–05 are **synthetic, analytic, or conceptual**. Figures **06–07 are derived from actual trained four-class MNIST runs provided by the author**, not from synthetic idealized tetrahedra. Do not confuse their training protocol with the quickstart preset.

~~~bash
python scripts/reproduce.py figures
~~~

That command calls the deterministic generator at scripts/make_figures.py, producing five SVGs, five PNGs, and figures/manifest.json. The manifest records the toy seed, optimizer parameters, numeric center positions and measured squared pair distances.

| File | Type | Provenance |
|---|---|---|
| 01_toy_comparison.svg | Synthetic | Fixed generated embeddings; optimize proxies under APML vs APML+MES |
| 02_objective_pipeline.svg | Conceptual | Shared-proxy APML/MES architecture |
| 03_ideal_tetrahedron.svg | Analytic | Four coordinates with all six squared distances equal to 8 |
| 04_six_squared_distances.svg | Synthetic | Distances from the same two runs used for 01 |
| 05_dimensional_limit.svg | Analytic | Four-point simplex and six-point illustrative octahedron |
| 06_mnist_trained_proxies.svg | **Real trained MNIST** | Seed 42, two methods, true 3D proxies + 220 displayed test embeddings per arm, six full-3D distances |
| 07_mnist_c4_three_seed_regularity.svg | **Real trained MNIST** | Ratio of min/mean Euclidean proxy distance for seeds 42/43/44 |

**Figures 06–07 are committed visual evidence, not outputs of `scripts/make_figures.py`.** Their exact measurements, method settings, original ZIP SHA-256 checksums and caveats are listed in [experimental provenance](../docs/mnist_c4_evidence.md).

**Interpretation:** a regular tetrahedron is feasible for four distinct class proxies in 3D. MES is the variance of squared distances, and zero variance also admits coincident centers. Do not portray the mathematical construction as a learned configuration or infer retrieval performance from a visually attractive synthetic plot.

For additional real training data, run the separately documented MNIST experiment and preserve the raw snapshots and configuration. Note that the runnable demonstration uses a **different** eight-epoch beta=0.2 preset, so it does not reproduce the archived figures 06–07 exactly. Do not copy images or unpublished data from the private research lab.
