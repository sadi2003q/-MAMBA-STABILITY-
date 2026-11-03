#!/usr/bin/env bash
# Runs the whole exp03b fair read-out check in one go:
#   1. the mathematical check (a06, a few seconds)
#   2. the architecture checks (seconds)
#   3. training: 3 arms x seeds 0 and 1 (about 1.5 hours on a Kaggle T4)
#   4. the plain-language summary
#
# On Kaggle, in one cell:
#   !bash /kaggle/working/-MAMBA-STABILITY-/scripts/run_exp03b.sh
#
# Training resumes by default, so if the Kaggle session stops, run the same command again.

# Stop at the first command that fails
set -euo pipefail

# Step 1: find the repository folder (the folder above this script)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$ROOT/configs/experiments/exp03b_fair_readout.yaml"
OUTPUT="$ROOT/outputs/exp03b_fair_readout"

echo "=== Step 1 of 4: mathematical check (a06) ==="
python "$ROOT/analysis/a06_anchor_linear_share.py"

echo "=== Step 2 of 4: architecture checks ==="
python "$ROOT/tools/check_model.py" \
    --config "$CONFIG"

echo "=== Step 3 of 4: training (3 arms, seeds 0 and 1) ==="
# To also run the slow full-Mamba control (about 1 extra hour per seed),
# add  full_mamba_anchor_recipe  to the --arms line.
python "$ROOT/train.py" \
    --config "$CONFIG" \
    --arms lstm_anchor_recipe chain_mamba_recipe lstm_recipe \
    --seeds 0 1 \
    --ask-first no

echo "=== Step 4 of 4: summary ==="
python "$ROOT/summarize.py" \
    --experiment "$OUTPUT"

echo "Done. Full summary: $OUTPUT/summary.md"
