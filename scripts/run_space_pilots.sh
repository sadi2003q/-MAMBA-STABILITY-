#!/usr/bin/env bash
# Runs both space-engineering pilots in one go (seed 0 only; direction only):
#   1. the two mathematical checks (seconds, NumPy only)
#   2. pilot A: satellite pointing without the steady pointing error   (about 10-20 minutes on a processor)
#   3. pilot B: staying on an orbit near an irregular asteroid           (about 20-40 minutes on a processor,
#                                                                        faster on a Kaggle graphics card)
# Each pilot ends with  Decision: GO  or  Decision: STOP  (also saved in its outputs/<name>/summary.md).
#
# On Kaggle, in one cell:
#   !bash /kaggle/working/-MAMBA-STABILITY-/scripts/run_space_pilots.sh
#
# Both pilots resume by default: if a session stops, run the same command again.
# To run only one pilot:  bash scripts/run_space_pilots.sh A    or    bash scripts/run_space_pilots.sh B

# Stop at the first command that fails
set -euo pipefail

# Step 1: find the repository folder (the folder above this script) and which pilot(s) to run
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHICH="${1:-AB}"

echo "=== Step 1 of 3: mathematical checks ==="
python "$ROOT/analysis/p01_attitude_offset_fixed_point.py"
python "$ROOT/analysis/p02_asteroid_model_mismatch.py"

if [[ "$WHICH" == *A* ]]; then
    echo "=== Step 2 of 3: pilot A (satellite pointing) ==="
    # the predictor is tiny and planned step by step: the processor is usually faster than a graphics card
    python "$ROOT/tools/run_pilot_attitude.py" \
        --config "$ROOT/configs/pilots/pilot_a_attitude_offset.yaml" \
        --seed 0 \
        --device cpu \
        --quick no
fi

if [[ "$WHICH" == *B* ]]; then
    echo "=== Step 3 of 3: pilot B (asteroid proximity) ==="
    # the real-asteroid gravity (642 mass points) is the slow part: a graphics card helps if there is one
    python "$ROOT/tools/run_pilot_asteroid.py" \
        --config "$ROOT/configs/pilots/pilot_b_asteroid_proximity.yaml" \
        --seed 0 \
        --device auto \
        --quick no
fi

echo ""
echo "=== Verdicts ==="
for NAME in pilot_a_attitude_offset pilot_b_asteroid_proximity; do
    SUMMARY="$ROOT/outputs/$NAME/summary.md"
    if [[ -f "$SUMMARY" ]]; then
        echo "$NAME: $(grep '^Decision:' "$SUMMARY" || echo 'no decision line')"
    fi
done
