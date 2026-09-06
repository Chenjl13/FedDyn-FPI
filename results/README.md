# Results

The HDF5 files in `results/h5/` are provided result curves.

## Provenance

- These files do not contain complete command, seed, commit, environment, or rho metadata.
- Some file names do not match the current `system/flcore/servers/serverbase.py::save_results()` naming pattern.
- Treat them as provided result curves, not as fully provenance-tracked artifacts.
- Metrics derived from these curves by `scripts/summarize_results.py` may not exactly match all values in the paper tables, including parts of Table III-V.
- These files should not be described as exact reproduction artifacts for the paper tables unless their original commands and seeds are confirmed.

## Compression Curves

Compression result files follow this pattern:

```text
<Dataset>_<rho>.h5
```

Examples:

```text
MNIST_1.0.h5
MNIST_0.5.h5
MNIST_0.2.h5
MNIST_0.1.h5
```

Each file stores:

```text
rs_test_acc
rs_test_auc
rs_train_loss
```

## Baseline Curves

Baseline files follow this pattern:

```text
<Dataset>_<Algorithm>.h5
```

Examples:

```text
MNIST_FedAvg.h5
MNIST_FedProx.h5
MNIST_MOON.h5
MNIST_FedDyn.h5
```

## Normalized Communication Cost

Use:

```bash
python scripts/summarize_results.py
```

The script reports:

```text
Best Acc
Final Acc
R@threshold
C = R(rho) * rho
```

Thresholds:

```text
MNIST: 0.90
FashionMNIST: 0.65
Cifar10: 0.50
```

This is a normalized communication metric. It is not byte-level traffic and does not include index overhead.
