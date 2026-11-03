# Filming genuine APML + MES training in three dimensions

**Status: scientific preview; NOT approved for public publication.**

This is an honest and reproducible animation of *learned embeddings and trainable
class proxies*, not an artist's trajectory. It contains two simultaneous paired
training conditions:

| Arm | Loss | Everything else |
| --- | --- | --- |
| Left | $\mathcal L_{\rm APML}$ | same model, same initialization, same minibatches, same optimizer |
| Right | $\mathcal L_{\rm APML}+\beta\mathcal L_{\rm MES}$ | only the MES weight differs |

Exactly four classes are shown. Encoder outputs are directly 3D: **no UMAP, t-SNE,
PCA, normalization, animation tweening, or post-hoc projection**. The six thin
lines connect every pair of the four class proxies. The six bar-chart values
are **squared Euclidean distances**, and the time-series plots their **population
variance**, the original MES objective. The display uses the same fixed camera
and axes for both arms and all frames in the saved MP4.

## Installation

From the unpacked preview repository root:

```bash
python -m pip install -e '.[movie,dev]'
# ffmpeg is needed to encode MP4 (install from the operating-system package manager).
python -m pytest -q
```

Choose a compatible PyTorch/torchvision build for your installed CPU or GPU.
For NVIDIA GPU, follow the official PyTorch instructions for the driver/CUDA
combination; do not force a GPU build blindly.

## 1. Test locally, even without the internet (synthetic SHAPES)

```bash
python scripts/cluster_movie.py train \
  --dataset shapes --epochs 3 --train-per-class 100 \
  --val-per-class 20 --display-per-class 20 \
  --snapshot-every 4 --out runs/shapes_demo \
  --device cpu --video
```

This is a **real training run on generated images of lines and circles**, not
MNIST and not a result of the publication. The same loss definitions and video
code are exercised; sample shapes are explicitly identified in the animation.

## 2. Film the requested MNIST four-class experiment

```bash
python scripts/cluster_movie.py train \
  --dataset mnist --digits 1,2,3,4 \
  --epochs 8 --train-per-class 1200 \
  --val-per-class 200 --display-per-class 120 \
  --batch-size 128 --snapshot-every 5 \
  --seed 42 --beta 0.2 \
  --out runs/mnist_movie_seed42 \
  --device auto --video --dpi 150 --fps 8
```

**The first MNIST invocation downloads MNIST automatically.** This script uses
*only the official MNIST training split*. For each digit, it takes disjoint
seeded training and holdout subsets. The fixed displayed examples are selected
from that holdout and do **not** participate in parameter updates. It never
loads the official MNIST test split and never touches CUB/Cars.

At every snapshot, each model generates fresh 3D embeddings for exactly the
same held-out images. Accuracy in the frame subtitle is computed on the
**displayed holdout subset**, not a benchmark test set. Train/validation
checkpoint selection is not performed: each arm runs the same fixed number of
steps. In particular, the chosen $\beta=0.2$ is a visual demonstration
parameter, **not** an experimentally justified optimum.

Outputs:

```text
runs/mnist_movie_seed42/
  config.json               # parameters, dataset-selection hashes, software
  history.jsonl             # step-wise metrics (shown validation subset)
  checkpoints/apml.pt       # final model state (not used to fake frames)
  checkpoints/apml_mes.pt   # final model state, for later verification
  snapshots/step_000000.npz # real model embeddings and weights at step 0
  snapshots/step_000005.npz # real model embeddings and weights at step 5
  ...
  live/latest.png           # replaced atomically during training
  live/index.html           # optional local browser auto-refresh viewer
  movie.mp4                 # H.264 1920×1080 when --dpi 150
```

All points in a movie frame come from the specified snapshot; frames may be
held for a configurable number of **identical** video frames (and the first
and final frames get extra holds). Models have
no artificial center-interpolation or simulated geometry.

## 3. Watch while the models train

The image `runs/.../live/latest.png` is recreated at each capture. You can
view it in JupyterLab using `examples/monitor_training.ipynb` in a second
notebook tab. That monitor polls the image without exposing any additional
network service or requiring the Docker socket.

On a local desktop with an interactive Matplotlib backend you may add
`--live-window`. Do **not** use it in a headless/Jupyter server terminal.

If you prefer a local browser, an optional static server can display the
self-refreshing `live/index.html`:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory runs/mnist_movie_seed42/live
```

Then open `http://127.0.0.1:8765`. The localhost URL is intentionally not
exposed to the internet. If you train on a remote server, the Jupyter notebook
monitor is simpler than opening another firewall or VPN port.

## 4. Re-render at a different speed or quality

```bash
python scripts/cluster_movie.py render --run runs/mnist_movie_seed42 --fps 6 --dpi 150 --frames-per-snapshot 4
python scripts/cluster_movie.py render --run runs/mnist_movie_seed42 --fps 8 --dpi 150 --save-frames
# No ffmpeg available? Render numbered PNG images instead:
python scripts/cluster_movie.py render --run runs/mnist_movie_seed42 --frames-only
```

Use these output files in an editor to add a title, narration, references, and
footnotes. Changing video FPS or `--frames-per-snapshot` repeats unchanged captured
frames, and does not change any recorded training positions. If the live preview axis crops dots, increase
`--preview-axis-limit`; the final video independently calculates **one fixed**
axis bound from **all** saved snapshots, so positions are never normalized per
frame or per condition.

## 5. What the animation means — and what it does not

- APML uses $-\log(1+\|z_i-w_k\|^2)$ class logits and ordinary cross-entropy.
- MES is the **variance of the unique squared inter-proxy distances** and is
  applied to **trainable class weights**, not batch centroids.
- MES is not scale invariant and can be zero for coincident centers; the APML
  term is the data-dependent discriminative part of the coupled objective.
- We measure geometry, rather than force a tetrahedron: the results need not
  form one, and the MES loss need not decrease monotonically in every run.
- A prettier cluster plot is not evidence of improved metric-learning
  generalization. The paper's benchmark results require their own protocols.
- We do not use the private research server paths, CUB test data, experimental
  relative-MES objectives, or any private checkpoint.

## Quality control before making a YouTube claim

1. Record the exact environment and command with the seed and beta.
2. Review `history.jsonl` and the six distance trajectories, not just 3D aesthetics.
3. Verify that the displayed holdout accuracy has **not** been described as
   official test accuracy.
4. Run at least three additional seeds before describing an effect as robust.
5. State clearly that this is an **educational four-class MNIST illustration**
   unless a full controlled benchmark protocol is run.

The preview has no redistribution license. Obtain explicit clearance before
publishing the source, video, or edited experimental material on a public site.