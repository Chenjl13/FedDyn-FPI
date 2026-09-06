#!/usr/bin/env python
"""Summarize provided FedDyn-FPI compression result curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


THRESHOLDS = {
    "MNIST": 0.90,
    "FashionMNIST": 0.65,
    "Cifar10": 0.50,
}

RHOS = ("1.0", "0.5", "0.2", "0.1")


def summarize_file(path: Path, dataset: str, rho: float, threshold: float) -> tuple[str, float, int, float, float, int | None, float | None]:
    with h5py.File(path, "r") as handle:
        acc = np.asarray(handle["rs_test_acc"])

    reached = np.flatnonzero(acc >= threshold)
    round_at_threshold = int(reached[0]) if reached.size else None
    normalized_cost = round_at_threshold * rho if round_at_threshold is not None else None
    return (
        dataset,
        rho,
        len(acc),
        float(np.max(acc)),
        float(acc[-1]),
        round_at_threshold,
        normalized_cost,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute best/final accuracy and normalized communication cost from HDF5 curves.")
    parser.add_argument("--results-dir", default="results/h5")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    rows = []
    for dataset, threshold in THRESHOLDS.items():
        for rho_text in RHOS:
            path = results_dir / f"{dataset}_{rho_text}.h5"
            if path.exists():
                rows.append(summarize_file(path, dataset, float(rho_text), threshold))

    print("Dataset       rho   Points  Best Acc  Final Acc  R@threshold  C=R*rho")
    print("------------  ----  ------  --------  ---------  -----------  -------")
    for dataset, rho, points, best_acc, final_acc, round_at_threshold, normalized_cost in rows:
        r_text = "NA" if round_at_threshold is None else str(round_at_threshold)
        c_text = "NA" if normalized_cost is None else f"{normalized_cost:.2f}"
        print(f"{dataset:<12}  {rho:<4.1f}  {points:>6}  {best_acc:>8.4f}  {final_acc:>9.4f}  {r_text:>11}  {c_text:>7}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
