# Refactor Changelog

This file records repository-cleanup changes made for GitHub release preparation.

## Unreleased

### Changed

- Expanded `.gitignore` to exclude Python cache files, local environments, logs, generated datasets, raw dataset caches, and model checkpoints.
  - Why: prevent large/generated/private machine artifacts from entering GitHub.
  - Algorithm behavior affected: No.

- Replaced `prepare.sh` with a generic environment creation helper.
  - Why: remove machine/user-specific shell aliases and keep installation portable.
  - Algorithm behavior affected: No.

- Added `environment.yml` and `requirements.txt` with versions compatible with the checked `testdyn` CUDA 11.8 environment.
  - Why: provide reproducible installation instructions.
  - Algorithm behavior affected: No.

- Added `docs/method.md`.
  - Why: document the mapping between the paper method and the current implementation.
  - Algorithm behavior affected: No.

- Added `scripts/train.py` as a thin wrapper around `system/main.py`.
  - Why: provide a root-level training command without moving or rewriting core code.
  - Algorithm behavior affected: No.

- Rewrote `README.md` as a paper-code repository draft.
  - Why: replace generic PFLlib documentation with method, setup, and reproduction guidance for this project.
  - Algorithm behavior affected: No.

- Fixed the `rho=1.0` control-flow path in `clientdyn.py`.
  - Why: full-upload mode should keep the current local model and avoid an undefined `uploaded_model_vector`.
  - Algorithm behavior affected: No for `rho < 1`; `rho = 1.0` now runs as intended instead of hitting a branch error after the first round.

- Added HDF5 metadata attributes to future `save_results()` outputs.
  - Why: record dataset, algorithm, rho, rounds, local epochs, batch size, learning rate, and FedDyn alpha for reproducibility.
  - Algorithm behavior affected: No.

- Added `scripts/summarize_results.py`.
  - Why: compute best accuracy, final accuracy, threshold round, and normalized communication cost from provided compression curves.
  - Algorithm behavior affected: No.

- Added `results/README.md`.
  - Why: document the limited provenance of existing HDF5 files and define the normalized communication metric.
  - Algorithm behavior affected: No.

- Updated README and `results/README.md` with final release notes about provided result curves, incomplete provenance, and possible mismatch with paper Table III-V values.
  - Why: avoid overstating reproducibility of existing HDF5 files.
  - Algorithm behavior affected: No.

- Added PFLlib acknowledgement and citation guidance to README.
  - Why: preserve upstream attribution for the adapted codebase.
  - Algorithm behavior affected: No.

### Removed

- Removed `CNAME`.
  - Why: it pointed to the upstream project website configuration and is unrelated to this paper repository.
  - Algorithm behavior affected: No.
