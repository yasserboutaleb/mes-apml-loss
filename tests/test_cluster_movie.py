"""Safety/provenance checks for the APML+MES real-snapshot video experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from mes_apml.cluster_movie import (
    MovieConfig, build_splits, make_mnist_splits, make_shapes, read_history, read_snapshot,
    select_fixed_validation, train_run, unique_squared_distances,
)
from mes_apml.cluster_movie_viz import movie_axis_limit, render_movie
from mes_apml.torch_backend import mes_loss


def small_config():
    return MovieConfig(dataset="shapes", train_per_class=12, val_per_class=7,
                       display_per_class=5, batch_size=24, epochs=1,
                       snapshot_every=1, beta=0.5, num_threads=2,
                       preview=False, device="cpu")


def test_original_mes_probes_squared_pairs():
    W = torch.tensor([[0., 0., 0.], [2., 0., 0.], [0., 3., 0.], [0., 0., 4.]], dtype=torch.float64)
    q = unique_squared_distances(W)
    np.testing.assert_array_equal(q, [4., 9., 16., 13., 20., 25.])
    np.testing.assert_allclose(mes_loss(W).item(), np.var(q), rtol=1e-13)


def test_offline_shapes_seed_and_balance():
    a = make_shapes(5, seed=19)
    b = make_shapes(5, seed=19)
    assert torch.equal(a.tensors[0], b.tensors[0])
    assert torch.equal(a.tensors[1], b.tensors[1])
    assert torch.bincount(a.tensors[1]).tolist() == [5, 5, 5, 5]


def test_fixed_holdout_selection():
    cfg = small_config()
    train, valid = build_splits(cfg)
    assert len(train) == 4 * cfg.train_per_class
    images1, labels1 = select_fixed_validation(valid, cfg.display_per_class, cfg.seed)
    images2, labels2 = select_fixed_validation(valid, cfg.display_per_class, cfg.seed)
    assert images1.shape == (4 * cfg.display_per_class, 1, 28, 28)
    assert torch.equal(images1, images2)
    assert torch.equal(labels1, labels2)
    assert torch.bincount(labels1).tolist() == [5, 5, 5, 5]


def test_paired_training_snapshots_are_real_and_identical_at_step0(tmp_path):
    cfg = small_config()
    run_dir = train_run(tmp_path / "experiment", cfg)
    history = read_history(run_dir)
    assert [h["step"] for h in history] == [0, 1, 2]
    config = json.loads((run_dir / "config.json").read_text())
    assert config["status"] == "complete"
    assert len(config["train_selection_sha256"]) == 64
    assert (run_dir / "checkpoints" / "apml.pt").exists()
    checkpoint = torch.load(run_dir / "checkpoints" / "apml_mes.pt", map_location="cpu", weights_only=True)
    assert checkpoint["step"] == 2
    assert checkpoint["state_dict"]["class_weights"].shape == (4, 3)
    initial = read_snapshot(run_dir / "snapshots" / "step_000000.npz")
    final = read_snapshot(run_dir / "snapshots" / "step_000002.npz")
    for key_a, key_b in (("emb_apml", "emb_apml_mes"), ("centers_apml", "centers_apml_mes"),
                         ("d2_apml", "d2_apml_mes")):
        np.testing.assert_array_equal(initial[key_a], initial[key_b])
    assert np.max(np.abs(final["centers_apml"] - initial["centers_apml"])) > 0
    assert np.max(np.abs(final["centers_apml_mes"] - final["centers_apml"])) > 0
    for snap in (initial, final):
        assert snap["emb_apml"].shape == (20, 3)
        assert snap["centers_apml_mes"].shape == (4, 3)
        assert len(snap["d2_apml_mes"]) == 6
        assert np.isfinite(snap["emb_apml_mes"]).all()
        assert np.isfinite(snap["centers_apml_mes"]).all()
    np.testing.assert_allclose(history[-1]["apml_mes_mes"], np.var(final["d2_apml_mes"]), rtol=1e-5)


def test_paired_cpu_training_is_reproducible(tmp_path):
    cfg = small_config()
    one = train_run(tmp_path / "first", cfg)
    two = train_run(tmp_path / "second", cfg)
    for p in sorted((one / "snapshots").glob("*.npz")):
        a = read_snapshot(p)
        b = read_snapshot(two / "snapshots" / p.name)
        for k in a:
            np.testing.assert_array_equal(a[k], b[k])


def test_snapshot_visuals_and_frames_export(tmp_path):
    run = train_run(tmp_path / "experiment", small_config())
    snaps = sorted((run / "snapshots").glob("*.npz"))
    assert movie_axis_limit(snaps) > 0
    directory = render_movie(run, video=False, dpi=60)
    images = sorted(directory.glob("frame_*.png"))
    assert len(images) == len(snaps)
    from PIL import Image
    with Image.open(images[0]) as image:
        assert image.size == (768, 432)


def test_existing_run_is_protected(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "important.txt").write_text("not overwritten")
    with pytest.raises(FileExistsError):
        train_run(d, small_config())
    assert (d / "important.txt").read_text() == "not overwritten"


def test_invalid_parameters():
    with pytest.raises(ValueError):
        MovieConfig(beta=-1).validate()
    with pytest.raises(ValueError):
        MovieConfig(display_per_class=300, val_per_class=100).validate()
    with pytest.raises(ValueError):
        MovieConfig(digits=(1, 2, 3, 3)).validate()


def test_mnist_split_on_in_memory_fixture_without_network(monkeypatch, tmp_path):
    """Exercise actual dataset selection code without claiming an MNIST benchmark."""
    import torchvision.datasets

    class FakeMNIST:
        def __init__(self, root, train, download, transform):
            assert train is True  # Never use the official TEST split.
            assert download is True
            self.targets = torch.arange(10, dtype=torch.long).repeat_interleave(12)

        def __getitem__(self, index):
            label = int(self.targets[index])
            return torch.full((1, 28, 28), float(label) / 10), label

        def __len__(self):
            return len(self.targets)

    monkeypatch.setattr(torchvision.datasets, "MNIST", FakeMNIST)
    train, val = make_mnist_splits(str(tmp_path), (1, 2, 3, 4), 5, 4, seed=42)
    assert len(train) == 20 and len(val) == 16
    assert set(train.indices).isdisjoint(val.indices)
    assert set(y for _, y in train) == {0, 1, 2, 3}
    assert set(y for _, y in val) == {0, 1, 2, 3}
