# Figure inventory and provenance

Generate all figures locally using `python scripts/make_figures.py --out figures` with a fixed synthetic seed. **None are benchmark results or publisher figure reproductions.** Both SVG and PNG exports and `manifest.json` are created by the same script.

| ID | Figure | Provenance |
|---|---|---|
| 01 | `01_toy_comparison.svg` | Synthetic fixed embeddings; trainable class proxies |
| 02 | `02_objective_pipeline.svg` | Conceptual objective flow |
| 03 | `03_ideal_tetrahedron.svg` | Analytic tetrahedron with six squared distances equal to 8 |
| 04 | `04_six_squared_distances.svg` | Synthetic distances from figure 01 |
| 05 | `05_dimensional_limit.svg` | Analytic tetrahedron and illustrative octahedron |

The images in the source snapshot were generated locally, but the binary/image assets are not yet committed to this review branch. Run the script to generate them before reviewing the README layout or approving public release. Any real-data illustration requires a separately verified training provenance.
