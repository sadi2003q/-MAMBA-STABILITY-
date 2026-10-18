"""Short screen output, full details in a log file.

The rule for every program in this project:
  - the screen shows only progress bars and the most important numbers;
  - everything else (tables, gradient norms, warnings, settings) goes into a log file,
    which can be opened later if something needs checking.

How to use it:

    log = ConsoleLog("outputs/exp01_certificate_ablation/log.txt")
    log.say("Training started")          # screen AND log file
    log.detail("learning rate 0.001")    # log file only
    log.close()
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

from tqdm import tqdm


class ConsoleLog:
    """Writes short messages to the screen and full details to a log file."""

    def __init__(self, log_path, fresh: bool = False):
        # Step 1: make sure the folder for the log file exists
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Step 2: open the file
        #   fresh=True  -> start a new, empty file ("w" = write)
        #   fresh=False -> keep old content and add to the end ("a" = append)
        if fresh:
            mode = "w"
        else:
            mode = "a"
        self.file = open(self.path, mode)

        # Step 3: count warnings instead of showing them on screen
        self.warning_count = 0

        # Step 4: write a header line with the date and time
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.file.write("\n")
        self.file.write("=" * 80 + "\n")
        self.file.write(f"Started {now}\n")
        self.file.write("=" * 80 + "\n")
        self.file.flush()

    def say(self, text: str = "") -> None:
        """Show on screen and also save in the log file."""
        # tqdm.write prints without breaking any progress bar on screen
        tqdm.write(text)
        self.detail(text)

    def detail(self, text: str = "") -> None:
        """Save in the log file only (not shown on screen)."""
        self.file.write(text + "\n")
        self.file.flush()

    def capture_warnings(self) -> None:
        """Send every Python and PyTorch warning to the log file instead of the screen."""

        def write_warning_to_log(message, category, filename, lineno, file=None, line=None):
            # Step 1: count it
            self.warning_count = self.warning_count + 1
            # Step 2: save the full warning in the log file
            short_name = Path(filename).name
            self.detail(f"[warning] {category.__name__} ({short_name}, line {lineno}): {message}")

        # From now on, Python calls our function instead of printing the warning
        warnings.showwarning = write_warning_to_log

    def close(self) -> None:
        """Close the log file. Mention how many warnings were hidden, if any."""
        if self.warning_count > 0:
            self.say(f"  ({self.warning_count} warnings were saved to the log file, not shown here)")
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"Finished {now}\n")
        self.file.close()


def progress_bar(total: int, description: str, unit: str, already_done: int = 0):
    """One tidy progress bar that works in a terminal and in a Kaggle notebook cell.

    leave=False   -> the bar disappears when finished (a short result line replaces it)
    ncols=100     -> fixed width, so it does not wrap onto several lines
    mininterval=1 -> redraw at most once per second (fewer lines in saved Kaggle logs)
    """
    bar = tqdm(
        total=total,
        initial=already_done,
        desc=description,
        unit=unit,
        leave=False,
        ncols=100,
        mininterval=1.0,
    )
    return bar
