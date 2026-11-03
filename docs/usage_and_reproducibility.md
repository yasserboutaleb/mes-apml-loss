# Reproducing MES-loss: one entry point, three levels

This repository implements the original 2023 APML+MES objective in PyTorch, with a small educational reproduction workflow. The code is distributed under the [MIT license](../LICENSE).

## Install

Python 3.10+ and a compatible PyTorch/torchvision combination are required. In the environment of your choice, from the repository root:

~~~bash
python -m pip install -e ".[dev,movie]"
~~~

CPU is sufficient for tests, figures and the small offline shapes demo. MP4 requires FFmpeg on PATH; it is not required for mathematical tests.

## Reproduction matrix

| Goal | Command | Required data | Output |
|---|---|---|---|
| Validate mathematics | python scripts/reproduce.py check | None | Pytest |
| Test training API | python scripts/reproduce.py quickstart | Generated toy vectors | Printed loss values |
| Recreate illustrations | python scripts/reproduce.py figures | Generated vectors | SVG, PNG, manifest |
| Offline actual training | python scripts/reproduce.py smoke --out runs/shapes_smoke_42 | Generated procedural images | Actual embeddings/proxy snapshots |
| MNIST 4-digit training | python scripts/reproduce.py mnist --out runs/mnist_1234_seed42 --video | MNIST official TRAIN split | Two trained models, snapshots, MP4 |

Each run requires a **new, empty output path**. The script will not delete or replace prior training runs.

## Exact commands

### A. Verify the loss and its gradients

~~~bash
python scripts/reproduce.py check
python scripts/reproduce.py quickstart
~~~

The tests independently check the Cauchy-affinity probabilities, population variance of unique **squared** distances, analytical and finite-difference gradients, tetrahedron, scale/translation invariances, collapse and the C=2 case.

### B. Regenerate deterministic figures

~~~bash
python scripts/reproduce.py figures
~~~

Five figures come from a deterministic analytic/synthetic generator. The generator writes the seed and synthetic run outcomes to figures/manifest.json. Committed SVGs are available immediately; PNGs and the manifest are created by this command. **Synthetic fixed embeddings are not MNIST evidence.**

### C. Record real, paired MNIST 3D training

~~~bash
python scripts/reproduce.py mnist --out runs/mnist_1234_seed42 --video
~~~

- The educational preset uses seed 42, digits 1/2/3/4, dimension 3, eight epochs and MES beta=0.2.
- **APML alone** is compared with **APML+MES**, not with Softmax. Any Softmax comparison needs a separate actual trained baseline.
- Both arms start with the same network weights and process the same minibatches.
- The visualized examples are fixed, disjoint holdout examples selected only from the MNIST **training** split.
- Snapshots are true model states. A video may duplicate frames, but not interpolate learned coordinates.
- The command is a **new educational protocol**, not the 2023 paper's benchmark training or open-set retrieval experiment.

For video configuration and live monitoring, see [the cluster movie guide](cluster_movie_guide.md). Without FFmpeg, omit --video to collect snapshots, or use the underlying render command to generate PNGs.

## Scientific provenance

Preserve the command, commit SHA, seeds, software versions, input split hashes, configuration, exact saved NPZ snapshots, and measured metrics. Distinguish **analytic**, **synthetic**, **actual educational MNIST**, and **published benchmark** materials. Cross-hardware bit-for-bit numerical identity is not assumed.

MES regularizes relative homogeneity of squared distances but is scale-sensitive, and zero MES includes collapsed class proxies. It does not itself guarantee positive max-min separation or retrieval generalization.

**License and citations:** source code is under the [MIT license](../LICENSE). Cite the [2023 paper](https://doi.org/10.1016/j.patrec.2023.06.005) for the scientific method. MNIST data and third-party software keep their own licensing terms.
