#!/usr/bin/env python
"""Thin root-level wrapper for the original PFLlib-style training entrypoint."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DATASET_MAP = {
    "mnist": "MNIST",
    "MNIST": "MNIST",
    "fashionmnist": "FashionMNIST",
    "FashionMNIST": "FashionMNIST",
    "cifar10": "Cifar10",
    "Cifar10": "Cifar10",
}

ALGORITHM_MAP = {
    "fedavg": "FedAvg",
    "FedAvg": "FedAvg",
    "fedprox": "FedProx",
    "FedProx": "FedProx",
    "moon": "MOON",
    "MOON": "MOON",
    "feddyn": "FedDyn",
    "FedDyn": "FedDyn",
    "feddyn_fpi": "FedDyn",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run experiments through system/main.py without changing core training code."
    )
    parser.add_argument("--dataset", default="mnist", choices=sorted(DATASET_MAP))
    parser.add_argument("--algorithm", default="feddyn_fpi", choices=sorted(ALGORITHM_MAP))
    parser.add_argument("--model", default="CNN")
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--local-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=0.001)
    parser.add_argument("--num-clients", type=int, default=20)
    parser.add_argument("--join-ratio", type=float, default=1.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--device-id", default="0")
    parser.add_argument("--goal", default="paper")
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--eval-gap", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args, extra = build_parser().parse_known_args()
    root = Path(__file__).resolve().parents[1]
    system_dir = root / "system"
    main_py = system_dir / "main.py"

    cmd = [
        sys.executable,
        str(main_py.name),
        "-data",
        DATASET_MAP[args.dataset],
        "-algo",
        ALGORITHM_MAP[args.algorithm],
        "-m",
        args.model,
        "-gr",
        str(args.rounds),
        "-ls",
        str(args.local_epochs),
        "-lbs",
        str(args.batch_size),
        "-lr",
        str(args.learning_rate),
        "-al",
        str(args.alpha),
        "-nc",
        str(args.num_clients),
        "-jr",
        str(args.join_ratio),
        "-dev",
        args.device,
        "-did",
        args.device_id,
        "-go",
        args.goal,
        "-t",
        str(args.times),
        "-eg",
        str(args.eval_gap),
        "--rho",
        str(args.rho),
        *extra,
    ]

    if args.dry_run:
        print(" ".join(cmd))
        return 0

    return subprocess.call(cmd, cwd=system_dir)


if __name__ == "__main__":
    raise SystemExit(main())
