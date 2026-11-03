"""Actual-snapshot visualization for APML vs APML+original MES.

All positions displayed here are read from model-created .npz snapshots.
The 3D limits, viewpoints, and class colors stay constant within a movie.
"""
from __future__ import annotations

import json
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np

from mes_apml.cluster_movie import COLORS, read_history, read_snapshot

BG = "#0C1425"
PANEL = "#111E32"
GRID = "#31425B"
WHITE = "#F2F5FA"
MUTED = "#AEC0D9"
LEFT = "#49C2E6"
RIGHT = "#FAA767"


def make_live_html(path: Path):
    # Use only local files. This is an auto-refresh viewer; it NEVER edits runs.
    html = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>APML + MES — live snapshots</title>
<style>body{margin:0;background:#0c1425;color:#f2f5fa;font:16px system-ui,sans-serif;text-align:center}
img{width:min(100vw,1440px);height:auto}p{color:#aec0d9;margin:12px}</style></head>
<body><p>Live monitor: real training snapshots (refreshes every 2 seconds)</p>
<img id="frame" src="latest.png" alt="Live paired APML/MES optimization">
<script>setInterval(()=>{document.getElementById('frame').src='latest.png?t='+Date.now()},2000)</script>
</body></html>'''
    path.write_text(html, encoding="utf-8")


def movie_axis_limit(paths: list[Path]) -> float:
    maximum = 0.0
    for path in paths:
        snap = read_snapshot(path)
        for key in ("emb_apml", "emb_apml_mes", "centers_apml", "centers_apml_mes"):
            maximum = max(maximum, float(np.max(np.abs(snap[key]))))
    return float(max(0.8, maximum * 1.12))


def _style_3d(ax, lim):
    ax.set_facecolor(PANEL)
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.set_box_aspect((1, 1, 1), zoom=1.03)
    ax.view_init(elev=19, azim=-59)
    ticks = [-round(lim / 2, 1), 0, round(lim / 2, 1)]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    ax.set_xlabel("z₁", color=MUTED, labelpad=-1, fontsize=9)
    ax.set_ylabel("z₂", color=MUTED, labelpad=-1, fontsize=9)
    ax.set_zlabel("z₃", color=MUTED, labelpad=-1, fontsize=9)
    ax.tick_params(labelsize=7, colors=MUTED, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.line.set_color(GRID)
        axis._axinfo["grid"].update({"color": GRID, "linewidth": 0.4})


def _draw_cluster(ax, z: np.ndarray, w: np.ndarray, labels: np.ndarray,
                  class_names: list[str], lim: float):
    _style_3d(ax, lim)
    for ci, name in enumerate(class_names):
        points = z[labels == ci]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=COLORS[ci], s=11, alpha=0.57, linewidths=0,
                   depthshade=False, rasterized=True)
    # Display identical edge definitions for APML-only and APML+MES.
    for i, j in combinations(range(4), 2):
        ax.plot([w[i, 0], w[j, 0]], [w[i, 1], w[j, 1]],
                [w[i, 2], w[j, 2]], color="#D3DEE9", alpha=0.42, linewidth=1.2)
    for ci, name in enumerate(class_names):
        ax.scatter([w[ci, 0]], [w[ci, 1]], [w[ci, 2]],
                   s=120, marker="D", color=COLORS[ci], edgecolors=WHITE,
                   linewidths=1.25, depthshade=False)


def _style_2d(ax):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(MUTED)
    ax.xaxis.label.set_color(MUTED)
    ax.title.set_color(WHITE)
    ax.grid(axis="y", color=GRID, alpha=0.55, linewidth=0.6)
    ax.set_axisbelow(True)


def render_frame(snapshot: dict, metric: dict, history: list[dict],
                 class_names: list[str], *, axis_limit: float,
                 dataset: str, beta: float, fixed_metric_range: tuple[float, float] | None,
                 fig=None):
    """Draw one frame without changing coordinates or synthesizing model data."""
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    else:
        fig.clear()
        fig.set_facecolor(BG)
    gs = fig.add_gridspec(2, 2, height_ratios=(3.0, 1.15),
                          left=0.065, right=0.958, top=0.81, bottom=0.13,
                          hspace=0.38, wspace=0.18)
    ax_left = fig.add_subplot(gs[0, 0], projection="3d")
    ax_right = fig.add_subplot(gs[0, 1], projection="3d")
    ax_bars = fig.add_subplot(gs[1, 0])
    ax_trend = fig.add_subplot(gs[1, 1])
    step = int(np.asarray(snapshot["step"]).item())
    epoch = int(np.asarray(snapshot["epoch"]).item())
    fig.text(0.5, 0.970, "WATCH THE CLASS GEOMETRY EMERGE", ha="center", va="top",
             color=WHITE, fontsize=21, fontweight="bold")
    source = "MNIST train holdout" if dataset == "mnist" else "Procedural shapes (offline demonstration)"
    fig.text(0.5, 0.912,
             f"{source}    •    real 3D embeddings    •    optimizer step {step:,}    •    epoch {epoch}",
             ha="center", va="center", fontsize=10, color=MUTED)
    legend_names = "   •   ".join(f"{tone}: {name}" for tone, name in
                                  zip(("Cyan", "Orange", "Purple", "Green"), class_names))
    fig.text(0.5, 0.877, legend_names + "   |   Diamonds: trainable proxies",
             ha="center", fontsize=9, color=MUTED)
    for ax, prefix, weights, embeddings, headline in [
        (ax_left, "apml", snapshot["centers_apml"], snapshot["emb_apml"], "APML ONLY  •  β = 0"),
        (ax_right, "apml_mes", snapshot["centers_apml_mes"], snapshot["emb_apml_mes"],
         f"APML + MES  •  β = {beta:g}")
    ]:
        _draw_cluster(ax, embeddings, weights, snapshot["labels"], class_names, axis_limit)
        ax.set_title(f"{headline}\n"
                     f"shown holdout acc {metric[prefix + '_accuracy']:.1%}   |   "
                     f"MES {metric[prefix + '_mes']:.3g}",
                     color=WHITE, fontsize=11.0, fontweight="medium", pad=3)
    _style_2d(ax_bars)
    pairs = list(combinations(range(4), 2))
    names = [f"{class_names[i]}–{class_names[j]}" for i, j in pairs]
    if max(len(s) for s in names) > 4:
        names = [f"{i+1}–{j+1}" for i, j in pairs]
    x = np.arange(len(pairs))
    ax_bars.bar(x - .18, snapshot["d2_apml"], .35, color=LEFT, label="APML", alpha=0.85)
    ax_bars.bar(x + .18, snapshot["d2_apml_mes"], .35, color=RIGHT, label="APML + MES", alpha=0.85)
    ax_bars.set_xticks(x, names)
    ax_bars.set_ylabel("Squared distance", fontsize=9)
    ax_bars.set_title("Six squared proxy-pair distances", fontsize=10.7, loc="left", pad=9, color=WHITE)
    ax_bars.legend(frameon=False, ncol=2, fontsize=8, labelcolor=WHITE, loc="upper right")
    ax_bars.set_ylim(bottom=0)

    _style_2d(ax_trend)
    steps = [int(m["step"]) for m in history if int(m["step"]) <= step]
    a = [max(1e-12, float(m["apml_mes"])) for m in history if int(m["step"]) <= step]
    b = [max(1e-12, float(m["apml_mes_mes"])) for m in history if int(m["step"]) <= step]
    ax_trend.plot(steps, a, color=LEFT, linewidth=2, label="APML")
    ax_trend.plot(steps, b, color=RIGHT, linewidth=2, label="APML + MES")
    ax_trend.scatter([steps[-1]], [a[-1]], color=LEFT, s=17)
    ax_trend.scatter([steps[-1]], [b[-1]], color=RIGHT, s=17)
    ax_trend.set_yscale("log")
    if fixed_metric_range is not None:
        ax_trend.set_ylim(*fixed_metric_range)
        ax_trend.set_xlim(0, max(1, int(history[-1]["step"])))
    else:
        ax_trend.set_xlim(0, max(1, step))
    ax_trend.set_xlabel("Optimizer step", fontsize=9)
    ax_trend.set_ylabel("MES variance (log scale)", fontsize=9)
    ax_trend.set_title("Original MES value during training", fontsize=10.7, loc="left", pad=9, color=WHITE)
    ax_trend.legend(frameon=False, ncol=2, fontsize=8, labelcolor=WHITE, loc="upper right")

    fig.text(0.5, 0.065,
             "Same model initialization + same minibatches  |  Diamonds: trainable class proxies  |  "
             "Dots: fixed held-out examples  |  Shared fixed 3D axes",
             color=MUTED, ha="center", fontsize=9)
    fig.text(0.5, 0.033,
             "Educational experiment, not a benchmark. MES equalizes squared proxy distances; it does not guarantee separation.",
             color="#91A4BE", ha="center", fontsize=8.1)
    return fig


def render_movie(run_dir: Path, *, video: Path | None = None, fps: int = 8,
                 dpi: int = 100, frames_per_snapshot: int = 3, save_frames: bool = False,
                 axis_limit: float | None = None):
    """Encode actual snapshots into H.264 MP4 (no interpolated model positions)."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    paths = sorted((run_dir / "snapshots").glob("step_*.npz"))
    history = read_history(run_dir)
    if not paths or not history:
        raise ValueError("No saved training snapshots to render")
    if fps < 1 or dpi < 50 or frames_per_snapshot < 1:
        raise ValueError("fps and frames_per_snapshot must be positive; dpi >= 50")
    if axis_limit is None:
        axis_limit = movie_axis_limit(paths)
    if axis_limit <= 0:
        raise ValueError("axis_limit must be positive")
    values = [max(1e-12, float(m[k])) for m in history for k in ("apml_mes", "apml_mes_mes")]
    lo, hi = min(values), max(values)
    metric_range = (max(1e-13, lo / 1.8), max(lo * 2, hi * 1.8))
    # History is one record for each snapshot. Each slide corresponds to a
    # REAL saved optimizer step; no tweening of embeddings or proxy trajectories.
    by_step = {m["step"]: m for m in history}
    frames_only = video is False
    video = None if frames_only else (Path(video) if video is not None else run_dir / "movie.mp4")
    if video is not None:
        video.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = run_dir / "frames"
    if save_frames or frames_only:
        frame_dir.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=BG)
    writer = None
    if video:
        if not shutil.which("ffmpeg") or not FFMpegWriter.isAvailable():
            plt.close(fig)
            raise RuntimeError("ffmpeg executable is required for MP4 export; install it or use --frames-only")
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=3500 if dpi <= 100 else 6500,
                              extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])

    def draw_one(i, path):
        snap = read_snapshot(path)
        step = int(snap["step"].item())
        m = by_step[step]
        return render_frame(snap, m, history, metadata["class_names"],
                            axis_limit=axis_limit, dataset=metadata["dataset"],
                            beta=metadata["beta"], fixed_metric_range=metric_range, fig=fig)

    try:
        if writer:
            with writer.saving(fig, str(video), dpi=dpi):
                for i, path in enumerate(paths):
                    draw_one(i, path)
                    if save_frames:
                        fig.savefig(frame_dir / f"frame_{i:05d}.png", dpi=dpi, facecolor=BG)
                    repeats = max(8, frames_per_snapshot) if i == 0 else max(14, frames_per_snapshot) if i == len(paths) - 1 else frames_per_snapshot
                    for _ in range(repeats):
                        writer.grab_frame(facecolor=BG)
                    if i % 10 == 0 or i == len(paths)-1:
                        print(f"[render] {i+1}/{len(paths)} snapshots", flush=True)
        else:
            for i, path in enumerate(paths):
                draw_one(i, path)
                fig.savefig(frame_dir / f"frame_{i:05d}.png", dpi=dpi, facecolor=BG)
    finally:
        plt.close(fig)
    print(f"Video: {video}" if writer else f"Frames: {frame_dir}")
    return video if writer else frame_dir
