# Reproducible figure inventory

All figures here are **computed or analytic** and must not be described as benchmark or real MNIST outcomes. The committed branch contains five SVG files. The PNG exports and provenance manifest are **generated locally**, not necessarily committed.

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

**Interpretation:** a regular tetrahedron is feasible for four distinct class proxies in 3D. MES is the variance of squared distances, and zero variance also admits coincident centers. Do not portray the mathematical construction as a learned configuration or infer retrieval performance from a visually attractive synthetic plot.

For real training data, run the separately documented MNIST experiment, preserve the raw snapshots and its configuration, and mark the figure as **measured MNIST training/holdout** with the actual seed and protocol. Do not copy images or unpublished data from the private research lab.
