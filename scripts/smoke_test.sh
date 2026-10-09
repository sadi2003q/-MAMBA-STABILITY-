#!/usr/bin/env bash
# Tiny run of everything (about a minute or two). Code check only — never interpret the numbers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${1:-$ROOT/configs/experiments/exp01_certificate_ablation.yaml}"
python "$ROOT/tools/check_model.py" --config "$CONFIG"
python "$ROOT/train.py" --config "$CONFIG" --quick --fresh --yes
