#!/usr/bin/env bash
# Phase 0 of the Chain-Mamba plan: reproduce the seed-0 results on your own machine (Kaggle graphics card).
#
# Runs in three parts so each fits in one Kaggle session. Training resumes by default, so if a session
# stops, run the same part again and it continues where it stopped.
#
#   part 1  Van der Pol: Chain-Mamba, Long Short-Term Memory network, and the Long Short-Term Memory
#           network with the same anchor (exp03b, the fair test)          about 45-60 minutes on a T4
#   part 2  Van der Pol: memory removed, multilayer perceptron (exp04)    about 20-30 minutes on a T4
#   part 3  Four-tank: Chain-Mamba, memory removed, Long Short-Term
#           Memory network (exp05)                                        about 60-90 minutes on a T4
#   check   compare everything with the reported numbers (no training, seconds)
#   all     tests + parts 1, 2, 3 + check
#
# On Kaggle, in one cell:
#   !bash /kaggle/working/-MAMBA-STABILITY-/scripts/reproduce_seed0.sh all
# or one part at a time:
#   !bash /kaggle/working/-MAMBA-STABILITY-/scripts/reproduce_seed0.sh 1

# Stop at the first command that fails
set -euo pipefail

# Step 1: find the repository folder (the folder above this script) and read which part to run
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PART="${1:-all}"

EXP03B="$ROOT/configs/experiments/exp03b_fair_readout.yaml"
EXP04="$ROOT/configs/experiments/exp04_mlp_control.yaml"
EXP05="$ROOT/configs/experiments/exp05_four_tank.yaml"

run_tests() {
    echo "=== Tests and architecture checks (seconds) ==="
    cd "$ROOT"
    python -m pytest -q
    python "$ROOT/tools/check_model.py" \
        --config "$EXP03B"
    python "$ROOT/tools/check_model.py" \
        --config "$EXP04"
    python "$ROOT/tools/check_model.py" \
        --config "$EXP05"
}

run_part1() {
    echo "=== Part 1 of 3: Van der Pol, exp03b fair read-out check, seed 0 ==="
    python "$ROOT/train.py" \
        --config "$EXP03B" \
        --arms chain_mamba_recipe lstm_recipe lstm_anchor_recipe \
        --seed 0 \
        --ask-first no
    python "$ROOT/summarize.py" \
        --experiment "$ROOT/outputs/exp03b_fair_readout"
}

run_part2() {
    echo "=== Part 2 of 3: Van der Pol, memory-free controls (exp04), seed 0 ==="
    python "$ROOT/train.py" \
        --config "$EXP04" \
        --arms chain_mamba_no_memory mlp_recipe \
        --seed 0 \
        --ask-first no
}

run_part3() {
    echo "=== Part 3 of 3: four-tank system (exp05), seed 0 ==="
    python "$ROOT/train.py" \
        --config "$EXP05" \
        --arms chain_mamba_recipe chain_mamba_no_memory lstm_recipe \
        --seed 0 \
        --ask-first no
}

run_check() {
    echo "=== Reproduction check ==="
    python "$ROOT/tools/check_reproduction.py" \
        --outputs "$ROOT/outputs"
}

# Step 2: run what was asked for
if [ "$PART" = "1" ]; then
    run_part1
    run_check
elif [ "$PART" = "2" ]; then
    run_part2
    run_check
elif [ "$PART" = "3" ]; then
    run_part3
    run_check
elif [ "$PART" = "check" ]; then
    run_check
elif [ "$PART" = "all" ]; then
    run_tests
    run_part1
    run_part2
    run_part3
    run_check
else
    echo "Unknown part '$PART'. Use one of: 1, 2, 3, check, all"
    exit 1
fi
