"""Apply the learning-rate rule that was written down BEFORE the check (exp01c), and fill in the final run.

What this program does, in plain words:
  1. Reads the finished runs of the learning-rate check (outputs/exp01c_learning_rate_check).
  2. For each network (full Mamba, Long Short-Term Memory network) it looks at the validation loss of
     every learning rate. The validation loss is measured on data the network never trained on, and
     it is NOT the test error that the paper reports, so the choice does not peek at the final answer.
  3. It applies the rule from the experiment file:
       - lowest validation loss wins;
       - scores within 3% of the best count as a tie (one seed is noisy);
       - in a tie, the learning rate already used (0.001) wins, so nothing changes without evidence.
  4. It writes the chosen number into configs/models/mamba_full_final.yaml and lstm_final.yaml, which
     replaces the word PLACEHOLDER. It refuses to overwrite a number that is already there, so the
     choice cannot be quietly changed after the final run has started.
  5. It saves everything it used to outputs/exp01c_learning_rate_check/chosen_learning_rates.json.

Run it once, after the learning-rate check has finished:

python tools/pick_learning_rate.py \
    --config configs/experiments/exp01c_learning_rate_check.yaml

Options:
    --apply no     only show the choice, do not write it into the final-run files (default yes)
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

from src.utils.cli import add_yes_no  # noqa: E402
from src.utils.config import REPO_ROOT, load_experiment, output_dir  # noqa: E402
from src.utils.io import load_json, save_json  # noqa: E402

PLACEHOLDER_LINE = "lr: PLACEHOLDER"


def to_number(text):
    """Turn text from the training log into a number, or None if it is empty, not a number, or infinite."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def read_training_record(run_dir: Path):
    """Best validation loss of one finished run, and the training loss at that same epoch.

    Returns None if the run has not finished yet.
    """
    # Step 1: the run is finished only when metrics.json exists
    metrics_file = run_dir / "metrics.json"
    log_file = run_dir / "train_log.csv"
    if not metrics_file.exists() or not log_file.exists():
        return None
    metrics = load_json(metrics_file)

    # Step 2: go through every epoch and remember the lowest validation loss
    best_validation = None
    training_at_best = None
    best_epoch = None
    with open(log_file) as handle:
        for row in csv.DictReader(handle):
            validation = to_number(row["val_loss"])
            training = to_number(row["train_loss"])
            if validation is None:
                continue
            if best_validation is None or validation < best_validation:
                best_validation = validation
                training_at_best = training
                best_epoch = int(row["epoch"])

    # Step 3: collect the facts
    record = {
        "blew_up": bool(metrics.get("diverged", False)),
        "best_validation_loss": best_validation,
        "training_loss_at_best": training_at_best,
        "best_epoch": best_epoch,
    }
    return record


def choose_learning_rate(candidates: list, tie_percent: float, preferred: float):
    """The rule from the experiment file. Returns (chosen candidate, list of tied candidates)."""
    # Step 1: the best (lowest) score
    best_score = min(c["score"] for c in candidates)

    # Step 2: everything within tie_percent of the best counts as tied
    limit = best_score * (1.0 + tie_percent / 100.0)
    tied = []
    for candidate in candidates:
        if candidate["score"] <= limit:
            tied.append(candidate)

    # Step 3: if the already-used learning rate is among the tied ones, keep it
    for candidate in tied:
        if math.isclose(candidate["lr"], preferred, rel_tol=1e-9):
            return candidate, tied

    # Step 4: otherwise take the tied one closest to the already-used value (on a logarithmic scale)
    def distance_to_preferred(candidate):
        return abs(math.log10(candidate["lr"]) - math.log10(preferred))

    tied_sorted = sorted(tied, key=distance_to_preferred)
    return tied_sorted[0], tied


def write_into_final_file(final_name: str, chosen_lr: float) -> str:
    """Replace PLACEHOLDER in configs/models/<final_name>.yaml. Refuses if it was already filled in."""
    path = REPO_ROOT / "configs" / "models" / f"{final_name}.yaml"
    text = path.read_text()
    if PLACEHOLDER_LINE not in text:
        sys.exit(f"{path.name} has no PLACEHOLDER left: the learning rate was already chosen.\n"
                 f"To choose again on purpose, restore the file from git first "
                 f"(git checkout -- configs/models/{path.name}).")
    new_text = text.replace(PLACEHOLDER_LINE, f"lr: {chosen_lr:.1e}")
    path.write_text(new_text)
    return path.name


def main():
    parser = argparse.ArgumentParser(description="Apply the written-down learning-rate rule.")
    parser.add_argument("--config", required=True)
    add_yes_no(parser, "--apply", True, "write the chosen values into the final-run files")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()

    # Step 1: read the experiment file and its rule
    exp = load_experiment(args.config)
    rule = exp["decision_rule"]["selection"]
    tie_percent = float(rule["tie_within_percent"])
    preferred = float(rule["tie_prefer"])
    out = output_dir(exp, quick=False, override_root=args.output_root)

    # Step 2: group the arms by the final-run file they will fill in
    groups = {}
    for arm in exp["arms"]:
        groups.setdefault(arm["sweep_for"], []).append(arm)

    # Step 3: read every run; stop clearly if something is missing
    scores = {}
    missing = []
    for final_name, arms in groups.items():
        scores[final_name] = []
        for arm in arms:
            record = read_training_record(out / "runs" / f"{arm['name']}_seed0")
            if record is None:
                missing.append(arm["name"])
                continue
            record["lr"] = float(arm["sweep_lr"])
            record["arm"] = arm["name"]
            scores[final_name].append(record)
    if missing:
        sys.exit("These runs have not finished yet: " + ", ".join(missing) +
                 "\nFinish the learning-rate check first (train.py with --config for exp01c).")

    # Step 4: apply the rule to each network
    print(f"Learning-rate check — {exp['name']} (seed 0)")
    print("Score = lowest validation loss over 40 epochs (smaller is better).")
    summary = {"rule": rule, "networks": {}}
    for final_name, records in scores.items():
        usable = []
        for record in records:
            if record["blew_up"] or record["best_validation_loss"] is None:
                continue
            record["score"] = record["best_validation_loss"]
            usable.append(record)
        print("")
        print(final_name)
        if not usable:
            sys.exit(f"Every run of {final_name} blew up or has no validation loss. Look at the log file.")

        chosen, tied = choose_learning_rate(usable, tie_percent, preferred)

        # Step 5: one line per learning rate
        for record in sorted(records, key=lambda r: r["lr"]):
            if record["blew_up"] or record["best_validation_loss"] is None:
                print(f"  learning rate {record['lr']:.4f}   blew up")
                continue
            if record["training_loss_at_best"]:
                gap = record["best_validation_loss"] / record["training_loss_at_best"]
                gap_text = f"validation/training {gap:.2f}"
            else:
                gap_text = "validation/training —"
            marker = ""
            if record is chosen:
                marker = "   <- chosen"
            print(f"  learning rate {record['lr']:.4f}   score {record['best_validation_loss']:.4f}   "
                  f"{gap_text}{marker}")

        # Step 6: say how the choice was made
        if len(tied) > 1:
            tied_text = ", ".join(f"{t['lr']:.4f}" for t in sorted(tied, key=lambda t: t["lr"]))
            print(f"  tie within {tie_percent:.0f}% between: {tied_text} → kept the one closest to {preferred:g}")
        grid = sorted(r["lr"] for r in usable)
        edge_note = ""
        if len(grid) > 1 and chosen["lr"] == grid[0]:
            edge_note = "lowest value tested; a lower one was not tried"
        if len(grid) > 1 and chosen["lr"] == grid[-1]:
            edge_note = "highest value tested; a higher one was not tried"
        if edge_note:
            print(f"  note: {edge_note}")

        summary["networks"][final_name] = {
            "chosen_learning_rate": chosen["lr"],
            "edge_of_grid_note": edge_note,
            "tied": [t["lr"] for t in tied],
            "all_runs": records,
        }

    # Step 7: write the choice into the final-run files (once) and save the evidence
    print("")
    if args.apply:
        # First make sure every file still has its PLACEHOLDER, so we never write only half of the choice
        for final_name in summary["networks"]:
            path = REPO_ROOT / "configs" / "models" / f"{final_name}.yaml"
            if PLACEHOLDER_LINE not in path.read_text():
                sys.exit(f"{path.name} has no PLACEHOLDER left: the learning rate was already chosen.\n"
                         f"To choose again on purpose, restore the files from git first "
                         f"(git checkout -- configs/models).")
        for final_name, info in summary["networks"].items():
            file_name = write_into_final_file(final_name, info["chosen_learning_rate"])
            print(f"Wrote {info['chosen_learning_rate']:.1e} into configs/models/{file_name}")
        save_json(summary, out / "chosen_learning_rates.json")
        print(f"Evidence saved to {out / 'chosen_learning_rates.json'}")
        print("Next: run the architecture checks and the final 8-seed run (exp01d).")
    else:
        print("Preview only (--apply no): nothing was written.")


if __name__ == "__main__":
    main()
