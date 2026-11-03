#!/usr/bin/env python3
"""Reproducible video: two actual APML / APML+original MES training runs.

Run from repository root after `python -m pip install -e '.[movie]'`:
  python scripts/cluster_movie.py train --dataset mnist --out runs/mnist_01 --video
  python scripts/cluster_movie.py render --run runs/mnist_01 --fps 8 --dpi 150

Offline smoke demonstration (genuine training on synthetic SHAPES, not MNIST):
  python scripts/cluster_movie.py train --dataset shapes --epochs 2 \\
      --train-per-class 64 --val-per-class 16 --display-per-class 12 \\
      --snapshot-every 4 --out runs/offline_demo --video
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Allow running directly from source checkout without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def cli(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    tr = sub.add_parser("train", help="train matched APML/APML+MES models and record snapshots")
    tr.add_argument("--dataset", choices=["mnist", "shapes"], default="mnist")
    tr.add_argument("--digits", default="1,2,3,4", help="four distinct MNIST digits")
    tr.add_argument("--out", type=Path, required=True, help="NEW empty output directory")
    tr.add_argument("--data-dir", type=Path, default=Path("data"))
    tr.add_argument("--epochs", type=int, default=6)
    tr.add_argument("--train-per-class", type=int, default=900)
    tr.add_argument("--val-per-class", type=int, default=120)
    tr.add_argument("--display-per-class", type=int, default=90)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--snapshot-every", type=int, default=20)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--lr", type=float, default=0.001)
    tr.add_argument("--beta", type=float, default=0.2, help="original MES weight, right model only")
    tr.add_argument("--proxy-init-std", type=float, default=0.6)
    tr.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    tr.add_argument("--num-threads", type=int, default=4)
    tr.add_argument("--preview-axis-limit", type=float, default=4.5)
    tr.add_argument("--no-preview", action="store_true", help="do not update live/latest.png")
    tr.add_argument("--live-window", action="store_true", help="open an interactive matplotlib window (desktop only)")
    tr.add_argument("--video", action="store_true", help="render MP4 automatically after training")
    tr.add_argument("--fps", type=int, default=8)
    tr.add_argument("--frames-per-snapshot", type=int, default=3, help="repeat each exact snapshot for a slower video (no interpolation)")
    tr.add_argument("--dpi", type=int, choices=[100, 150], default=100, help="100=720p, 150=1080p")

    re = sub.add_parser("render", help="make an MP4 from stored snapshots (no training)")
    re.add_argument("--run", type=Path, required=True)
    re.add_argument("--output", type=Path, help="MP4 output path; default: RUN/movie.mp4")
    re.add_argument("--fps", type=int, default=8)
    re.add_argument("--frames-per-snapshot", type=int, default=3)
    re.add_argument("--dpi", type=int, choices=[100, 150], default=150)
    re.add_argument("--axis-limit", type=float, help="override automatically calculated fixed 3D scale")
    re.add_argument("--save-frames", action="store_true", help="also output numbered PNG frames")
    re.add_argument("--frames-only", action="store_true", help="do not encode MP4; export PNG frames")

    args = p.parse_args(argv)
    # PNG/movie rendering must work in a headless terminal, including Jupyter servers.
    # --live-window keeps the configured interactive backend instead.
    import matplotlib
    if getattr(args, "live_window", False):
        if "agg" in matplotlib.get_backend().lower() or (os.name != "nt" and not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY")):
            p.error("--live-window needs a local graphical desktop. On a Jupyter server, monitor live/latest.png in the notebook.")
    else:
        matplotlib.use("Agg", force=True)

    from mes_apml.cluster_movie import MovieConfig, train_run
    from mes_apml.cluster_movie_viz import render_movie
    try:
        if args.command == "train":
            digits = tuple(int(s.strip()) for s in args.digits.split(","))
            cfg = MovieConfig(dataset=args.dataset, digits=digits, epochs=args.epochs,
                              train_per_class=args.train_per_class,
                              val_per_class=args.val_per_class,
                              display_per_class=args.display_per_class,
                              batch_size=args.batch_size,
                              snapshot_every=args.snapshot_every,
                              seed=args.seed, lr=args.lr, beta=args.beta,
                              proxy_init_std=args.proxy_init_std, device=args.device,
                              num_threads=args.num_threads,
                              preview=not args.no_preview,
                              preview_axis_limit=args.preview_axis_limit,
                              data_dir=str(args.data_dir))
            train_run(args.out, cfg, live_window=args.live_window)
            if args.video:
                render_movie(args.out, fps=args.fps, dpi=args.dpi, frames_per_snapshot=args.frames_per_snapshot)
        else:
            if args.frames_only and args.output:
                p.error("--output is not applicable with --frames-only")
            render_movie(args.run, video=False if args.frames_only else args.output,
                         fps=args.fps, dpi=args.dpi, frames_per_snapshot=args.frames_per_snapshot, save_frames=args.save_frames,
                         axis_limit=args.axis_limit)
    except (ValueError, FileExistsError, RuntimeError) as exc:
        p.exit(status=2, message=f"ERROR: {exc}\n")


if __name__ == "__main__":
    cli()
