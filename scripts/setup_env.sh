#!/usr/bin/env bash
# One-time setup on the Mac (Miniconda). On Kaggle nothing is needed: the libraries are preinstalled.
set -euo pipefail
cd "$(dirname "$0")/.."
conda env create -f environment.yml || conda env update -f environment.yml --prune
echo "Now run: conda activate mamba-cert"
