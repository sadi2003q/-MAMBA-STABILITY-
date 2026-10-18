"""Write the plain-language summary for an experiment folder.

The full summary is saved to summary.md; the screen shows the short table only.

python summarize.py --experiment outputs/exp01_certificate_ablation
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402

import yaml  # noqa: E402

from src.utils.io import save_json  # noqa: E402
from src.utils.summary import build_summary, read_results, short_summary  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Plain-language summary of one experiment folder.")
    p.add_argument("--experiment", required=True, help="outputs/<experiment name>")
    args = p.parse_args()

    # Step 1: find the results table
    folder = Path(args.experiment)
    results_csv = folder / "results.csv"
    if not results_csv.exists():
        sys.exit(f"No results.csv in {folder}. Run train.py first.")

    # Step 2: read the experiment settings and the results
    exp = yaml.safe_load((folder / "experiment_config.yaml").read_text())
    rows = read_results(results_csv)
    is_quick = folder.name.endswith("_quick")

    # Step 3: build the full summary and save it
    full_text, machine = build_summary(exp, rows, quick=is_quick)
    summary_path = folder / "summary.md"
    summary_path.write_text(full_text)
    save_json(machine, folder / "summary.json")

    # Step 4: show the short version on screen
    print(short_summary(machine, full_summary_path=summary_path))


if __name__ == "__main__":
    main()
