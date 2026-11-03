"""Check whether the seed-0 Chain-Mamba results reproduce on your own machine (Phase 0 of the Chain-Mamba plan).

What it does:
    1. Reads outputs/<experiment>/results.csv for exp03b, exp04 and exp05.
    2. Compares each seed-0 run with the number the earlier processor run reported (docs/chain_mamba_plan.md).
    3. Prints PASS when the 50-step chained error is within 10% of the reported number, FAIL otherwise.
    4. Prints the three comparisons that decide the paper's direction.

Pass rule (written before the reproduction runs, from the Chain-Mamba paper plan, Phase 0):
    each 50-step chained error must be within 10% of the reported value.
    If any run fails, do not continue to eight seeds: compare train_log.csv epoch by epoch with
    docs/evidence/<experiment>/runs/ to find where the two runs part.

This program does no training. It only reads results files.

Usage:
    python tools/check_reproduction.py
    python tools/check_reproduction.py --outputs /kaggle/working/-MAMBA-STABILITY-/outputs
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


# ---------------------------------------------------------------------------------------------
# The numbers reported by the earlier processor run (seed 0), copied from docs/chain_mamba_plan.md.
# Each entry: (experiment folder, arm name, readable name, reported 50-step error, reported one-step error)
# lstm_anchor_recipe is new (the fair baseline), so it has no reported number: it is shown, not checked.
# ---------------------------------------------------------------------------------------------
EXPECTED = [
    ("exp03b_fair_readout", "chain_mamba_recipe", "Van der Pol: Chain-Mamba + recipe", 0.0664, 0.0135),
    ("exp03b_fair_readout", "lstm_recipe", "Van der Pol: Long Short-Term Memory network + recipe", 0.0980, 0.0346),
    ("exp03b_fair_readout", "lstm_anchor_recipe", "Van der Pol: Long Short-Term Memory network + anchor + recipe", None, None),
    ("exp04_mlp_control", "chain_mamba_no_memory", "Van der Pol: Chain-Mamba, memory removed", 0.0484, 0.0116),
    ("exp04_mlp_control", "mlp_recipe", "Van der Pol: multilayer perceptron + recipe", 0.1021, 0.0206),
    ("exp05_four_tank", "chain_mamba_recipe", "Four-tank: Chain-Mamba + recipe", 0.002423, 0.000229),
    ("exp05_four_tank", "chain_mamba_no_memory", "Four-tank: Chain-Mamba, memory removed", 0.002289, 0.000207),
    ("exp05_four_tank", "lstm_recipe", "Four-tank: Long Short-Term Memory network + recipe", 0.003373, 0.000598),
]

TOLERANCE_PERCENT = 10.0
SEED = 0


def read_seed0_row(results_csv: Path, arm: str):
    """Return the results row for (arm, seed 0), or None if that run has not finished yet."""
    if not results_csv.exists():
        return None

    with open(results_csv) as file:
        reader = csv.DictReader(file)
        for row in reader:
            same_arm = row["arm"] == arm
            same_seed = int(row["seed"]) == SEED
            if same_arm and same_seed:
                return row

    return None


def percent_difference(new_value: float, old_value: float) -> float:
    """How far new_value is from old_value, in percent of old_value (positive = new is larger)."""
    difference = new_value - old_value
    return 100.0 * difference / old_value


def word_for(percent: float) -> str:
    """Plain-language label for a difference, as in CLAUDE.md Section 10 (smaller error is better)."""
    if abs(percent) <= 10:
        return "about the same"
    if percent < -20:
        return "clearly lower (better, on this seed)"
    if percent < 0:
        return "slightly lower (better)"
    if percent > 20:
        return "clearly higher (worse, on this seed)"
    return "slightly higher (worse)"


def main():
    # Step 1: read the command-line option (where the outputs folder is)
    parser = argparse.ArgumentParser(description="Check the seed-0 Chain-Mamba reproduction.")
    repository_root = Path(__file__).resolve().parent.parent
    parser.add_argument("--outputs", default=str(repository_root / "outputs"),
                        help="the outputs folder written by train.py")
    args = parser.parse_args()
    outputs = Path(args.outputs)

    # Step 2: go through every expected run and compare
    print()
    print("REPRODUCTION CHECK — seed 0, 50-step chained error (smaller is better)")
    print(f"Pass rule: within {TOLERANCE_PERCENT:.0f}% of the number reported by the earlier processor run.")
    print()
    print(f"  {'Run':<62} {'Reported':>10} {'Now':>10} {'Change':>8}  Verdict")

    found_errors = {}        # (experiment, arm) -> 50-step error, used for the comparisons below
    count_pass = 0
    count_fail = 0
    count_missing = 0

    for experiment, arm, readable_name, reported_error, reported_one_step in EXPECTED:
        results_csv = outputs / experiment / "results.csv"
        row = read_seed0_row(results_csv, arm)

        # Run not finished yet
        if row is None:
            print(f"  {readable_name:<62} {'':>10} {'—':>10} {'':>8}  not run yet")
            count_missing += 1
            continue

        # A run that blew up is reported as a count, never as an error number
        if row["diverged"].strip().lower() == "true":
            print(f"  {readable_name:<62} {'':>10} {'—':>10} {'':>8}  BLEW UP (FAIL)")
            count_fail += 1
            continue

        error_now = float(row["rollout_rmse"])
        found_errors[(experiment, arm)] = error_now

        # New arm with nothing to compare against: show only
        if reported_error is None:
            print(f"  {readable_name:<62} {'new':>10} {error_now:>10.4g} {'':>8}  (new arm, shown only)")
            continue

        change = percent_difference(error_now, reported_error)
        if abs(change) <= TOLERANCE_PERCENT:
            verdict = "PASS"
            count_pass += 1
        else:
            verdict = "FAIL"
            count_fail += 1
        print(f"  {readable_name:<62} {reported_error:>10.4g} {error_now:>10.4g} {change:>+7.1f}%  {verdict}")

    # Step 3: overall verdict
    print()
    print(f"Passed {count_pass}, failed {count_fail}, not run yet {count_missing}.")
    if count_fail > 0:
        print("Decision: NOT reproduced. Do not start the eight-seed runs. Compare train_log.csv epoch by epoch")
        print("          with docs/evidence/<experiment>/runs/ to find where the runs part.")
    elif count_missing > 0:
        print("Decision: not complete yet. Run the remaining parts of scripts/reproduce_seed0.sh.")
    else:
        print("Decision: REPRODUCED. Go on to the eight-seed runs (Phase 1 of the Chain-Mamba plan).")

    # Step 4: the three comparisons that decide the paper's direction (direction only, one seed)
    print()
    print("The three comparisons that matter (one seed: direction only)")

    comparisons = [
        ("Chain-Mamba vs Long Short-Term Memory network with the SAME anchor (exp03b, the fair test; "
         "rule: at least 20% lower)",
         ("exp03b_fair_readout", "chain_mamba_recipe"), ("exp03b_fair_readout", "lstm_anchor_recipe")),
        ("Van der Pol: Chain-Mamba vs Chain-Mamba with memory removed",
         ("exp03b_fair_readout", "chain_mamba_recipe"), ("exp04_mlp_control", "chain_mamba_no_memory")),
        ("Four-tank: Chain-Mamba vs Chain-Mamba with memory removed",
         ("exp05_four_tank", "chain_mamba_recipe"), ("exp05_four_tank", "chain_mamba_no_memory")),
    ]

    for title, first_key, second_key in comparisons:
        if first_key not in found_errors or second_key not in found_errors:
            print(f"  • {title}: not run yet")
            continue
        first_error = found_errors[first_key]
        second_error = found_errors[second_key]
        change = percent_difference(first_error, second_error)
        print(f"  • {title}:")
        print(f"      {first_error:.4g} against {second_error:.4g}  → Chain-Mamba is {change:+.1f}%, {word_for(change)}")

    print()


if __name__ == "__main__":
    main()
