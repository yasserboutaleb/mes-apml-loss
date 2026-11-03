"""Train two matched 3D classifiers and record REAL APML/MES trajectories.

This is an educational experiment. It uses only original APML and MES, does
not reproduce benchmark metrics, and does not use an official test set.

All images/embeddings/centers in saved training snapshots come from actual
forward passes and parameter updates (no fabricated animation trajectories).
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

from mes_apml.torch_backend import apml_loss, mes_loss

DEFAULT_DIGITS = (1, 2, 3, 4)
SHAPE_NAMES = ("vertical", "horizontal", "diagonal", "ring")
COLORS = ("#52D1DC", "#FFA857", "#C5A1FF", "#8FE388")


@dataclass(frozen=True)
class MovieConfig:
    dataset: str = "mnist"
    digits: tuple[int, ...] = DEFAULT_DIGITS
    epochs: int = 6
    train_per_class: int = 900
    val_per_class: int = 120
    batch_size: int = 128
    snapshot_every: int = 20
    display_per_class: int = 90
    lr: float = 1e-3
    beta: float = 0.2
    seed: int = 42
    proxy_init_std: float = 0.6
    device: str = "auto"
    data_dir: str = "data"
    preview: bool = True
    preview_axis_limit: float = 4.5
    num_threads: int = 4

    def validate(self):
        if self.dataset not in ("mnist", "shapes"):
            raise ValueError("dataset must be 'mnist' or 'shapes'")
        if len(self.digits) != 4 or len(set(self.digits)) != 4 or any(d not in range(10) for d in self.digits):
            raise ValueError("Exactly four distinct MNIST digit IDs in 0..9 are required")
        for name in ("epochs", "train_per_class", "val_per_class", "batch_size", "snapshot_every", "display_per_class", "num_threads"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.display_per_class > self.val_per_class:
            raise ValueError("display_per_class cannot exceed val_per_class")
        for name in ("lr", "proxy_init_std", "preview_axis_limit"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and strictly positive")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")


class TinyCNN3D(nn.Module):
    """Deliberately small model; the output is *directly* in R^3."""

    def __init__(self, num_classes: int = 4, proxy_init_std: float = 0.6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 7 * 7, 64), nn.ReLU(), nn.Linear(64, 3),
        )
        self.class_weights = nn.Parameter(torch.randn(num_classes, 3) * proxy_init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


def set_seed(seed: int, num_threads: int = 4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(num_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_shapes(count_per_class: int, seed: int) -> TensorDataset:
    """Offline toy dataset: genuine learned embeddings of PROCEDURAL shapes.

    NEVER label this dataset as MNIST or use its metrics as research evidence.
    """
    rng = np.random.default_rng(seed)
    images = []
    labels = []
    yy, xx = np.mgrid[0:28, 0:28]
    for class_id in range(4):
        for _ in range(count_per_class):
            dy, dx = rng.integers(-3, 4, size=2)
            u, v = yy - (14 + dy), xx - (14 + dx)
            width = float(rng.uniform(1.0, 2.0))
            if class_id == 0:  # vertical
                ink = (np.abs(v) < width) & (np.abs(u) < 10)
            elif class_id == 1:  # horizontal
                ink = (np.abs(u) < width) & (np.abs(v) < 10)
            elif class_id == 2:  # diagonal
                ink = (np.abs(u - v) < width * 1.5) & (np.abs(u) < 10)
            else:  # ring
                ink = np.abs(np.sqrt(u * u + v * v) - 8) < width
            img = np.clip(ink.astype(np.float32) + rng.normal(0, 0.13, (28, 28)), 0, 1)
            images.append(img[None, :, :].astype(np.float32))
            labels.append(class_id)
    # Keep class balance and permute order reproducibly.
    order = rng.permutation(len(labels))
    images_arr = np.asarray(images, dtype=np.float32)[order]
    labels_arr = np.asarray(labels, dtype=np.int64)[order]
    return TensorDataset(torch.from_numpy(images_arr), torch.from_numpy(labels_arr))


class MNISTSelection(Dataset):
    def __init__(self, base, indices: Sequence[int], mapping: dict[int, int]):
        self.base, self.indices, self.mapping = base, tuple(int(i) for i in indices), mapping

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, y = self.base[self.indices[i]]
        return img, self.mapping[int(y)]


def make_mnist_splits(data_dir: str, digits: tuple[int, ...], train_per_class: int,
                      val_per_class: int, seed: int):
    """Stratified, disjoint split of *official MNIST TRAIN* only. Never load test."""
    try:
        from torchvision.datasets import MNIST
        from torchvision.transforms import ToTensor
    except (ImportError, RuntimeError) as exc:
        raise RuntimeError("MNIST requires compatible torchvision and torch installs") from exc
    base = MNIST(root=data_dir, train=True, download=True, transform=ToTensor())
    targets = np.asarray(base.targets)
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for d in digits:
        options = np.flatnonzero(targets == d)
        need = train_per_class + val_per_class
        if len(options) < need:
            raise ValueError(f"Digit {d} has {len(options)} examples in MNIST train; requested {need}")
        shuffled = rng.permutation(options)
        val_indices.extend(shuffled[:val_per_class])
        train_indices.extend(shuffled[val_per_class:need])
    assert not set(train_indices).intersection(val_indices)
    mapping = {d: i for i, d in enumerate(digits)}
    return MNISTSelection(base, train_indices, mapping), MNISTSelection(base, val_indices, mapping)


def build_splits(cfg: MovieConfig):
    if cfg.dataset == "mnist":
        return make_mnist_splits(cfg.data_dir, cfg.digits,
                                 cfg.train_per_class, cfg.val_per_class, cfg.seed)
    return (make_shapes(cfg.train_per_class, cfg.seed + 1000),
            make_shapes(cfg.val_per_class, cfg.seed + 2000))


def selection_digest(ds: Dataset) -> str:
    """Fingerprint exact dataset selection, without adding private image data to logs."""
    h = hashlib.sha256()
    if isinstance(ds, MNISTSelection):
        h.update(np.asarray(ds.indices, dtype=np.int64).tobytes())
        h.update(np.asarray(sorted(ds.mapping.items()), dtype=np.int64).tobytes())
    elif isinstance(ds, TensorDataset):
        for t in ds.tensors:
            h.update(t.numpy().tobytes())
    else:
        raise TypeError("Unsupported dataset type for selection digest")
    return h.hexdigest()


def select_fixed_validation(val_data: Dataset, display_per_class: int, seed: int):
    """Fixed balanced subset from holdout for identical visible points every step."""
    by_label = {i: [] for i in range(4)}
    for index in range(len(val_data)):
        _, y = val_data[index]
        by_label[int(y)].append(index)
    rng = np.random.default_rng(seed)
    chosen = []
    for c in range(4):
        if len(by_label[c]) < display_per_class:
            raise ValueError(f"Not enough validation examples for class {c}")
        chosen.extend(rng.permutation(by_label[c])[:display_per_class].tolist())
    images, labels = zip(*(val_data[i] for i in chosen))
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def unique_squared_distances(weights: torch.Tensor) -> np.ndarray:
    """Return six original MES distances for 4 proxies, ordering (0,1) ... (2,3)."""
    left, right = torch.triu_indices(weights.shape[0], weights.shape[0], 1,
                                     device=weights.device)
    return ((weights[left] - weights[right]) ** 2).sum(-1).detach().cpu().numpy()


def accuracy_and_embeddings(model: TinyCNN3D, x: torch.Tensor,
                            y: torch.Tensor, batch_size: int = 256):
    was_training = model.training
    model.eval()
    pieces = []
    with torch.inference_mode():
        for chunk in x.split(batch_size):
            pieces.append(model(chunk.to(next(model.parameters()).device)).cpu())
        z = torch.cat(pieces, dim=0)
        w = model.class_weights.detach().cpu()
        logits = -torch.log1p((z[:, None] - w[None]).square().sum(-1))
        acc = float((logits.argmax(-1) == y).float().mean().item())
        ce = float(apml_loss(z, y, w).item())
        q = unique_squared_distances(w)
        spread = float(np.var(q))  # population variance, not the sample variance
        mean_q = float(np.mean(q))
        cv_q = float(np.std(q) / mean_q) if mean_q > 0 else None
    model.train(was_training)
    return z.numpy(), w.numpy(), q, {"accuracy": acc, "apml": ce,
                                     "mes": spread, "mean_d2": mean_q,
                                     "cv_squared_distances": cv_q}


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_snapshot(path: Path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temp = Path(f.name)
    try:
        np.savez_compressed(temp, **arrays)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_snapshot(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def read_history(run_dir: Path):
    path = run_dir / "history.jsonl"
    if not path.exists():
        return []
    return [json.loads(row) for row in path.read_text(encoding="utf-8").splitlines() if row.strip()]


def collect_snapshot(step: int, epoch: int, left: TinyCNN3D, right: TinyCNN3D,
                     images: torch.Tensor, labels: torch.Tensor):
    lz, lw, lq, lm = accuracy_and_embeddings(left, images, labels)
    rz, rw, rq, rm = accuracy_and_embeddings(right, images, labels)
    metric = {"step": int(step), "epoch": int(epoch),
              **{f"apml_{k}": v for k, v in lm.items()},
              **{f"apml_mes_{k}": v for k, v in rm.items()}}
    values = {"step": np.array(step, dtype=np.int64),
              "epoch": np.array(epoch, dtype=np.int64),
              "labels": labels.cpu().numpy().astype(np.int64),
              "emb_apml": lz, "emb_apml_mes": rz,
              "centers_apml": lw, "centers_apml_mes": rw,
              "d2_apml": lq, "d2_apml_mes": rq}
    return values, metric


def train_run(run_dir: Path, cfg: MovieConfig, live_window: bool = False):
    """Train both models on EXACTLY the same mini-batches, snapshot after updates."""
    cfg.validate()
    run_dir = Path(run_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Output directory not empty: {run_dir}. Choose a NEW run path.")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "snapshots").mkdir(exist_ok=True)
    (run_dir / "live").mkdir(exist_ok=True)
    set_seed(cfg.seed, cfg.num_threads)
    selected_device = "cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device)
    if selected_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but is unavailable")
    device = torch.device(selected_device)

    names = list(map(str, cfg.digits)) if cfg.dataset == "mnist" else list(SHAPE_NAMES)
    metadata = {**vars(cfg), "digits": list(cfg.digits), "class_names": names,
                "experiment_version": "0.2.0",
                "selected_device": selected_device,
                "torch_version": torch.__version__, "split": "holdout from official MNIST TRAIN only" if cfg.dataset == "mnist" else "separately generated procedural training and validation shapes",
                "comparison": "paired same initialization and identical minibatches; alpha=1, beta in right arm only",
                "visualization": "RAW 3D encoder embeddings, no 2D projection, fixed validation examples",
                "status": "running"}
    write_json(run_dir / "config.json", metadata)
    train_data, val_data = build_splits(cfg)
    metadata["train_selection_sha256"] = selection_digest(train_data)
    metadata["holdout_selection_sha256"] = selection_digest(val_data)
    write_json(run_dir / "config.json", metadata)
    val_x, val_y = select_fixed_validation(val_data, cfg.display_per_class, cfg.seed + 3000)
    generator = torch.Generator().manual_seed(cfg.seed + 4000)
    loader = DataLoader(train_data, batch_size=cfg.batch_size, shuffle=True,
                        generator=generator, num_workers=0)

    base = TinyCNN3D(proxy_init_std=cfg.proxy_init_std).to(device)
    model_apml = copy.deepcopy(base)
    model_mes = copy.deepcopy(base)
    opt_apml = torch.optim.Adam(model_apml.parameters(), lr=cfg.lr)
    opt_mes = torch.optim.Adam(model_mes.parameters(), lr=cfg.lr)
    del base

    history = []
    live_fig = None
    step = 0
    pairs = tuple(combinations(range(4), 2))
    initial_history = run_dir / "history.jsonl"
    initial_history.write_text("", encoding="utf-8")

    def capture(epoch: int):
        nonlocal history, live_fig
        data, metric = collect_snapshot(step, epoch, model_apml, model_mes, val_x, val_y)
        write_snapshot(run_dir / "snapshots" / f"step_{step:06d}.npz", **data)
        with initial_history.open("a", encoding="utf-8") as out:
            out.write(json.dumps(metric, ensure_ascii=False) + "\n")
        history.append(metric)
        print(f"[snapshot] step={step:5d} epoch={epoch} | "
              f"acc APML={metric['apml_accuracy']:.3f}, APML+MES={metric['apml_mes_accuracy']:.3f} | "
              f"MES variance={metric['apml_mes']:.4g}, {metric['apml_mes_mes']:.4g}", flush=True)
        if cfg.preview or live_window:
            from mes_apml.cluster_movie_viz import render_frame, make_live_html
            import matplotlib.pyplot as plt
            fig = render_frame(data, metric, history, names,
                               axis_limit=cfg.preview_axis_limit,
                               dataset=cfg.dataset, beta=cfg.beta,
                               fixed_metric_range=None, fig=live_fig if live_window else None)
            if cfg.preview:
                temp = run_dir / "live" / "latest.partial.png"
                fig.savefig(temp, dpi=110, format="png", facecolor=fig.get_facecolor())
                os.replace(temp, run_dir / "live" / "latest.png")
                make_live_html(run_dir / "live" / "index.html")
            if live_window:
                live_fig = fig
                plt.show(block=False)
                plt.pause(0.15)
            else:
                plt.close(fig)
        return metric

    capture(epoch=0)
    try:
        for epoch in range(1, cfg.epochs + 1):
            model_apml.train()
            model_mes.train()
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                for model, optimizer, beta in ((model_apml, opt_apml, 0.0),
                                               (model_mes, opt_mes, cfg.beta)):
                    optimizer.zero_grad(set_to_none=True)
                    z = model(images)
                    loss = apml_loss(z, labels, model.class_weights)
                    if beta > 0:
                        loss = loss + beta * mes_loss(model.class_weights)
                    loss.backward()
                    optimizer.step()
                step += 1
                if step % cfg.snapshot_every == 0:
                    capture(epoch=epoch)
            print(f"[epoch] {epoch}/{cfg.epochs} completed, optimizer steps={step}", flush=True)
        if step % cfg.snapshot_every != 0:
            capture(epoch=cfg.epochs)
        # Final checkpoint is optional for video rendering, but permits a
        # later independent verification of the two trained encoders.
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        for model, name in ((model_apml, "apml"), (model_mes, "apml_mes")):
            torch.save({"state_dict": model.state_dict(), "step": step,
                        "seed": cfg.seed, "class_names": names},
                       run_dir / "checkpoints" / (name + ".pt"))
        metadata["status"] = "complete"
        metadata["total_steps"] = step
        metadata["snapshot_count"] = len(history)
        write_json(run_dir / "config.json", metadata)
    except BaseException:
        # Partial NPZ snapshots can still be rendered and inspected.
        metadata["status"] = "interrupted"
        metadata["total_steps"] = step
        metadata["snapshot_count"] = len(history)
        write_json(run_dir / "config.json", metadata)
        raise
    if live_fig is not None:
        import matplotlib.pyplot as plt
        plt.close(live_fig)
    return run_dir
