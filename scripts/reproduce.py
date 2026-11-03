#!/usr/bin/env python3
"""One entry point for independent MES-loss educational reproduction.

Always run commands from the repository root; never overwrites an experiment.
Uses the current Python environment, never changes PyTorch installations.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def invoke(*args: str) -> None:
    print('[MES-loss]', sys.executable, *args, flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('check', help='Run offline mathematical/geometry/unit tests')
    sub.add_parser('quickstart', help='Train a tiny synthetic encoder on CPU')
    sub.add_parser('figures', help='Regenerate analytic and synthetic figures')
    smoke = sub.add_parser('smoke', help='Offline shapes test; real learning, no MNIST download')
    smoke.add_argument('--out', type=Path, required=True)
    smoke.add_argument('--device', choices=('cpu','cuda','auto'), default='cpu')
    mnist = sub.add_parser('mnist', help='Four digit MNIST educational demonstration')
    mnist.add_argument('--out', type=Path, required=True)
    mnist.add_argument('--device', choices=('cpu','cuda','auto'), default='auto')
    mnist.add_argument('--seed', type=int, default=42)
    mnist.add_argument('--video', action='store_true', help='Encode H264; needs FFmpeg in PATH')
    args = parser.parse_args(argv)
    if args.action == 'check':
        invoke('-m', 'pytest', '-q')
    elif args.action == 'quickstart':
        invoke('examples/quickstart_torch.py')
    elif args.action == 'figures':
        invoke('scripts/make_figures.py', '--out', 'figures')
    else:
        output = args.out.expanduser().resolve()
        if output == ROOT or ROOT.is_relative_to(output):
            parser.error('Refusing to use the repository root or an ancestor as output')
        if output.exists():
            parser.error(f'Output already exists; choose a new directory: {output}')
        base = ['scripts/cluster_movie.py', 'train', '--dataset',
                'shapes' if args.action == 'smoke' else 'mnist',
                '--out', str(output), '--device', args.device,
                '--seed', str(args.seed if args.action == 'mnist' else 42)]
        if args.action == 'smoke':
            base += ['--epochs', '1', '--train-per-class', '16', '--val-per-class', '8',
                     '--display-per-class', '4', '--batch-size', '32', '--snapshot-every',
                     '2', '--no-preview']
        else:
            base += ['--digits', '1,2,3,4', '--epochs', '8', '--train-per-class', '1200',
                     '--val-per-class', '200', '--display-per-class', '120', '--batch-size',
                     '128', '--snapshot-every', '5', '--beta', '0.2', '--lr', '0.001']
            if args.video:
                base += ['--video', '--dpi', '150', '--fps', '8']
        invoke(*base)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
