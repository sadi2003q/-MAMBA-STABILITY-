#!/usr/bin/env bash
# check -> train (seed 0) -> summary, for one experiment.
#   bash scripts/run_experiment.sh exp01_certificate_ablation
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="${1:?usage: run_experiment.sh <experiment name>}"
CONFIG="$ROOT/configs/experiments/$NAME.yaml"
python "$ROOT/tools/check_model.py" --config "$CONFIG"
python "$ROOT/train.py" --config "$CONFIG" --seed 0 --yes
python "$ROOT/summarize.py" --experiment "$ROOT/outputs/$NAME"
