"""Write the plain-language summary for an experiment folder (also printed to the terminal).

python summarize.py --experiment outputs/exp01_certificate_ablation
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse  # noqa: E402

import yaml  # noqa: E402

from src.utils.io import save_json  # noqa: E402
from src.utils.summary import build_summary, read_results  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Plain-language summary of one experiment folder.")
    p.add_argument("--experiment", required=True, help="outputs/<experiment name>")
    args = p.parse_args()
    folder = Path(args.experiment)
    if not (folder / "results.csv").exists():
        sys.exit(f"No results.csv in {folder}. Run train.py first.")
    exp = yaml.safe_load((folder / "experiment_config.yaml").read_text())
    text, machine = build_summary(exp, read_results(folder / "results.csv"), quick=folder.name.endswith("_quick"))
    (folder / "summary.md").write_text(text)
    save_json(machine, folder / "summary.json")
    print(text)
    print(f"Summary written to {folder / 'summary.md'}")


if __name__ == "__main__":
    main()
