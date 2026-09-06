#!/usr/bin/env bash
set -euo pipefail

# Optional helper for Linux/macOS users. Windows users should follow README.md.
conda env create -f environment.yml

echo "Environment created. Activate it with:"
echo "conda activate testdyn"
