"""
Training loop for one (arm, seed) run.

What this file does, in order:
    1. Builds the model and the optimizer.
    2. Trains the model epoch by epoch, with a tqdm progress bar.
    3. Saves a checkpoint every few epochs, so an interrupted run can continue.
    4. Stops early if training "blows up" (too many non-finite losses or gradients).
    5. Keeps the model weights that gave the best validation loss.
    6. Runs the full evaluation on the test set.

Everything for one run is written into its own folder (run_dir).

Beginner note: "non-finite" means the number is NaN (not a number) or infinity.
That happens when training becomes unstable.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.evaluation.diagnostics import certificate_status, error_amplification, limit_cycle
from src.evaluation.metrics import rollout_errors, teacher_forced_one_step
from src.losses.regimes import regime_loss
from src.models.registry import build_model, count_parameters
from src.utils.config import save_yaml
from src.utils.io import save_json, write_train_log
from src.utils.seed import set_seed


# ----------------------------------------------------------------------------------------------
# Small helper: remove private settings (keys starting with "_") before saving a config
# ----------------------------------------------------------------------------------------------
def _public_config(cfg: dict) -> dict:
    clean_cfg = {}
    for key, value in cfg.items():
        if key.startswith("_"):
            continue          # skip private / internal keys
        clean_cfg[key] = value
    return clean_cfg


# ----------------------------------------------------------------------------------------------
# Optimizer and learning-rate schedule
# ----------------------------------------------------------------------------------------------
def make_optimizer(model, cfg: dict):
    """
    Build the Adam optimizer and the learning-rate scheduler.

    Weight decay (a small pull of the weights towards zero) is applied only to
    weight matrices. It is NOT applied to:
        - parameters with ndim < 2 (biases, single vectors, scalars), and
        - parameters marked with the attribute _no_weight_decay = True
          (for example the decay parameter A of the state-space model).
    """
    training_cfg = cfg["training"]

    # Step 1: sort every trainable parameter into one of two lists
    params_with_decay = []
    params_without_decay = []

    for param in model.parameters():
        # Skip frozen parameters
        if not param.requires_grad:
            continue

        marked_no_decay = getattr(param, "_no_weight_decay", False)
        is_vector_or_scalar = param.ndim < 2

        if marked_no_decay or is_vector_or_scalar:
            params_without_decay.append(param)
        else:
            params_with_decay.append(param)

    # Step 2: one parameter group per list
    group_with_decay = {
        "params": params_with_decay,
        "weight_decay": training_cfg["weight_decay"],
    }
    group_without_decay = {
        "params": params_without_decay,
        "weight_decay": 0.0,
    }

    # Step 3: the Adam optimizer
    optimizer = torch.optim.Adam(
        [group_with_decay, group_without_decay],
        lr=training_cfg["lr"],
    )

    # Step 4: the learning rate is multiplied by gamma after every epoch
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=training_cfg["lr_decay_gamma"],
    )

    return optimizer, scheduler


# ----------------------------------------------------------------------------------------------
# Mini-batches
# ----------------------------------------------------------------------------------------------
def _batches(n: int, batch_size: int, generator: torch.Generator):
    """
    Shuffle the numbers 0 .. n-1 and hand them out in chunks of batch_size.
    Each chunk is a list of window indices that make one mini-batch.

    The shuffle uses our own random generator, so the order is reproducible
    for a given seed (and is saved in the checkpoint for resuming).
    """
    # Step 1: a random order of all window indices
    shuffled_order = torch.randperm(n, generator=generator)

    # Step 2: hand out one chunk at a time
    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_indices = shuffled_order[start:end]
        yield batch_indices


# ----------------------------------------------------------------------------------------------
# Validation loss
# ----------------------------------------------------------------------------------------------
@torch.no_grad()
def validation_loss(model, data: dict, regime: str, device, batch_size: int = 512) -> float:
    """
    Average training-style loss on the validation split.
    Returns infinity if any batch gives a non-finite loss.
    """
    model.eval()

    val_split = data["splits"]["val"]
    context_len = data["context_len"]
    horizon = data["horizon"]
    number_of_windows = val_split.xw.shape[0]

    total_loss = 0.0
    total_windows = 0

    for start in range(0, number_of_windows, batch_size):
        end = start + batch_size

        # Step 1: take one batch of windows and move it to the device
        x = val_split.xw[start:end].to(device)     # physical states
        u = val_split.uw[start:end].to(device)     # control inputs

        # Step 2: compute the loss for this batch
        loss, _ = regime_loss(model, x, u, regime, context_len, horizon)
        loss_value = float(loss)

        # Step 3: if the loss blew up, the whole validation loss is infinite
        if not math.isfinite(loss_value):
            return float("inf")

        # Step 4: add this batch, weighted by how many windows it has
        windows_in_batch = x.shape[0]
        total_loss = total_loss + loss_value * windows_in_batch
        total_windows = total_windows + windows_in_batch

    average_loss = total_loss / total_windows
    return average_loss


# ----------------------------------------------------------------------------------------------
# Timing (used only for the run-time estimate before a long job)
# ----------------------------------------------------------------------------------------------
def time_batches(cfg: dict, data: dict, device, n_batches: int = 3) -> float:
    """
    Measure how many seconds one training batch takes.
    The first batch is a warm-up and is not counted (it is always slower).
    """
    # Step 1: a fresh model and optimizer, just for timing
    set_seed(0)
    model = build_model(cfg).to(device)
    optimizer, _ = make_optimizer(model, cfg)

    train_split = data["splits"]["train"]
    batch_size = cfg["training"]["batch_size"]
    regime = cfg["training"]["regime"]
    number_of_windows = train_split.xw.shape[0]
    on_gpu = device.type == "cuda"

    measured_times = []

    # Step 2: run (warm-up + n_batches) batches
    for batch_number in range(n_batches + 1):
        # Pick consecutive windows; "% number_of_windows" wraps around at the end
        indices = torch.arange(batch_number * batch_size, (batch_number + 1) * batch_size)
        indices = indices % number_of_windows

        x = train_split.xw[indices].to(device)
        u = train_split.uw[indices].to(device)

        # On a graphics card, wait for earlier work to finish before starting the clock
        if on_gpu:
            torch.cuda.synchronize()
        start_time = time.time()

        # One normal training step
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, _ = regime_loss(model, x, u, regime, data["context_len"], data["horizon"])
        if torch.isfinite(loss):
            loss.backward()
            optimizer.step()

        # Wait again so we measure the real finishing time
        if on_gpu:
            torch.cuda.synchronize()
        elapsed = time.time() - start_time

        # Skip the warm-up batch (batch_number 0)
        if batch_number > 0:
            measured_times.append(elapsed)

    # Step 3: average seconds per batch
    average_seconds = sum(measured_times) / len(measured_times)
    return average_seconds


# ----------------------------------------------------------------------------------------------
# One full training run
# ----------------------------------------------------------------------------------------------
def train_run(cfg: dict, data: dict, seed: int, run_dir: Path, device, fresh: bool = False,
              position_label: str = "") -> dict:
    """
    Train one (arm, seed), then evaluate it. Returns the metrics dictionary.

    fresh = True  -> ignore any old checkpoint and start from epoch 0.
    fresh = False -> if checkpoint_last.pt exists, continue from it.
    """
    # ------------------------------------------------------------------
    # Step 1: prepare the run folder and save the settings used
    # ------------------------------------------------------------------
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(_public_config(cfg), run_dir / "config.yaml")

    training_cfg = cfg["training"]
    regime = training_cfg["regime"]
    context_len = data["context_len"]
    horizon = data["horizon"]

    # ------------------------------------------------------------------
    # Step 2: build the model, optimizer and the batch-shuffling generator
    # (set_seed must come first so the starting weights are reproducible)
    # ------------------------------------------------------------------
    set_seed(seed)
    model = build_model(cfg).to(device)
    optimizer, scheduler = make_optimizer(model, cfg)
    shuffle_generator = torch.Generator().manual_seed(seed)

    # ------------------------------------------------------------------
    # Step 3: starting values (used when there is no checkpoint)
    # ------------------------------------------------------------------
    checkpoint_path = run_dir / "checkpoint_last.pt"

    start_epoch = 0
    log_rows = []                    # one dictionary per finished epoch
    best_val = float("inf")          # best validation loss so far
    best_state = None                # model weights at the best validation loss
    nonfinite_total = 0              # bad steps over the whole run
    diverged = False                 # becomes True if training blows up
    train_seconds = 0.0              # total training time (without evaluation)

    # ------------------------------------------------------------------
    # Step 4: resume from the last checkpoint if there is one
    # ------------------------------------------------------------------
    if checkpoint_path.exists() and not fresh:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        shuffle_generator.set_state(checkpoint["generator"])

        start_epoch = checkpoint["epoch"]
        log_rows = checkpoint["log_rows"]
        best_val = checkpoint["best_val"]
        best_state = checkpoint["best_state"]
        nonfinite_total = checkpoint["nonfinite_total"]
        diverged = checkpoint["diverged"]
        train_seconds = checkpoint.get("train_seconds", 0.0)   # older checkpoints may not have it

        tqdm.write(f"  resuming {position_label} from epoch {start_epoch}")

    # ------------------------------------------------------------------
    # Step 5: the epoch loop
    # ------------------------------------------------------------------
    train_split = data["splits"]["train"]
    number_of_windows = train_split.xw.shape[0]
    total_epochs = int(training_cfg["epochs"])
    batch_size = training_cfg["batch_size"]
    grad_clip = training_cfg["grad_clip"]
    max_bad_fraction = training_cfg["max_nonfinite_fraction"]
    checkpoint_every = int(training_cfg["checkpoint_every"])

    progress_bar = tqdm(
        range(start_epoch, total_epochs),
        desc=f"  {position_label}",
        unit="epoch",
        leave=False,
        initial=start_epoch,
        total=total_epochs,
    )

    for epoch in progress_bar:
        # A run that has already blown up is not trained any further
        if diverged:
            break

        epoch_start_time = time.time()
        model.train()

        loss_sum = 0.0       # sum of the good (finite) losses in this epoch
        steps = 0            # all batches tried in this epoch
        nonfinite = 0        # batches skipped because of NaN / infinity

        # ---- 5a: go through the training data one mini-batch at a time ----
        for batch_indices in _batches(number_of_windows, batch_size, shuffle_generator):
            x = train_split.xw[batch_indices].to(device)
            u = train_split.uw[batch_indices].to(device)

            # Forward pass: compute the loss
            optimizer.zero_grad(set_to_none=True)
            loss, _ = regime_loss(model, x, u, regime, context_len, horizon)
            steps = steps + 1

            # Bad loss? Skip this batch.
            if not torch.isfinite(loss):
                nonfinite = nonfinite + 1
                continue

            # Backward pass: compute the gradients
            loss.backward()

            # Bad gradient anywhere? Throw the gradients away and skip this batch.
            grads_ok = True
            for param in model.parameters():
                if param.grad is None:
                    continue
                if not torch.isfinite(param.grad).all():
                    grads_ok = False
                    break

            if not grads_ok:
                nonfinite = nonfinite + 1
                optimizer.zero_grad(set_to_none=True)
                continue

            # Limit the gradient size, then update the weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            loss_sum = loss_sum + float(loss)

        # ---- 5b: end of the epoch: lower the learning rate ----
        scheduler.step()
        nonfinite_total = nonfinite_total + nonfinite

        # ---- 5c: did training blow up in this epoch? ----
        if steps == 0:
            diverged = True
        elif nonfinite / steps > max_bad_fraction:
            diverged = True

        # ---- 5d: validation loss, and remember the best weights ----
        if diverged:
            val = float("inf")
        else:
            val = validation_loss(model, data, regime, device)

        if math.isfinite(val) and val < best_val:
            best_val = val
            best_state = copy.deepcopy(model.state_dict())

        # ---- 5e: average training loss over the good batches only ----
        good_steps = steps - nonfinite
        if good_steps < 1:
            good_steps = 1                     # avoid dividing by zero
        train_loss = loss_sum / good_steps

        # ---- 5f: timing and the log row for this epoch ----
        epoch_seconds = time.time() - epoch_start_time
        train_seconds = train_seconds + epoch_seconds

        current_lr = optimizer.param_groups[0]["lr"]
        log_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val,
            "lr": current_lr,
            "nonfinite_steps": nonfinite,
            "seconds": round(epoch_seconds, 2),
        }
        log_rows.append(log_row)

        # Show the latest numbers next to the progress bar
        progress_bar.set_postfix(loss=f"{train_loss:.3e}", val=f"{val:.3e}", bad=nonfinite)

        # ---- 5g: save a checkpoint every few epochs, at the last epoch, or on divergence ----
        epochs_done = epoch + 1
        time_to_save = (epochs_done % checkpoint_every == 0)
        is_last_epoch = (epochs_done == total_epochs)

        if time_to_save or is_last_epoch or diverged:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "generator": shuffle_generator.get_state(),
                "epoch": epochs_done,
                "log_rows": log_rows,
                "best_val": best_val,
                "best_state": best_state,
                "nonfinite_total": nonfinite_total,
                "diverged": diverged,
                "train_seconds": train_seconds,
            }
            torch.save(checkpoint, checkpoint_path)
            write_train_log(run_dir / "train_log.csv", log_rows)

    progress_bar.close()
    write_train_log(run_dir / "train_log.csv", log_rows)

    # ------------------------------------------------------------------
    # Step 6: go back to the best weights (if the config asks for it)
    # ------------------------------------------------------------------
    use_best = training_cfg.get("select_best_on_validation", True)
    if use_best and best_state is not None:
        model.load_state_dict(best_state)

    final_checkpoint = {
        "model": model.state_dict(),
        "config": _public_config(cfg),
    }
    torch.save(final_checkpoint, run_dir / "checkpoint.pt")

    # ------------------------------------------------------------------
    # Step 7: full evaluation, then add training information
    # ------------------------------------------------------------------
    metrics = evaluate_model(model, cfg, data, device)

    metrics["training"] = {
        "diverged_during_training": diverged,
        "nonfinite_steps": nonfinite_total,
        "epochs_run": len(log_rows),
        "best_val_loss": best_val,
        "train_minutes": train_seconds / 60.0,
    }
    metrics["parameters"] = count_parameters(model)

    # The run counts as diverged if it blew up in training OR in the test rollout
    diverged_in_test = metrics["primary"]["diverged"]
    metrics["diverged"] = bool(diverged or diverged_in_test)

    save_json(metrics, run_dir / "metrics.json")
    return metrics


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------
def evaluate_model(model, cfg: dict, data: dict, device) -> dict:
    """
    Evaluate a trained model on the test split.

    "Primary" regime = the way the model was trained to predict:
        - trained "direct"  -> primary is direct multi-step prediction
        - anything else     -> primary is chained prediction (feeding its own outputs back)
    The other regime is also measured, for comparison.
    """
    # Step 1: decide which prediction regime is primary
    regime = cfg["training"]["regime"]
    if regime == "direct":
        primary = "direct"
        other = "chained"
    else:
        primary = "chained"
        other = "direct"

    eval_cfg = cfg["evaluation"]
    context_len = data["context_len"]
    horizon = data["horizon"]
    test_split = data["splits"]["test"]
    norm = data["norm"]

    # Step 2: the main error numbers
    primary_errors = rollout_errors(model, primary, test_split, norm, context_len, horizon,
                                    device, eval_cfg["report_steps"])
    other_errors = rollout_errors(model, other, test_split, norm, context_len, horizon,
                                  device, eval_cfg["report_steps"])
    one_step_rmse = teacher_forced_one_step(model, test_split, norm, device)
    certificate = certificate_status(model)

    out = {
        "primary_regime": primary,
        "primary": primary_errors,
        "other_regime": other_errors,
        "teacher_forced_one_step_rmse": one_step_rmse,
        "certificate": certificate,
    }

    # Step 3: extra diagnostics.
    # If a diagnostic crashes, we store the error message instead,
    # so a finished training run is never lost because of a diagnostic.
    try:
        out["amplification"] = error_amplification(
            model, primary, test_split, norm, data["system"], context_len, horizon,
            eval_cfg["jacobian_windows"], device,
        )
    except Exception as exc:
        out["amplification"] = {"error": repr(exc)}

    try:
        out["limit_cycle"] = limit_cycle(
            model, primary, norm, data["system"], context_len,
            eval_cfg["limit_cycle_steps"], eval_cfg["limit_cycle_start"], device,
        )
    except Exception as exc:
        out["limit_cycle"] = {"status": "error", "error": repr(exc)}

    return out


# ----------------------------------------------------------------------------------------------
# One row of results.csv
# ----------------------------------------------------------------------------------------------
def result_row(arm_cfg: dict, seed: int, metrics: dict) -> dict:
    """Turn one run's metrics into a flat row for results.csv."""
    # Step 1: pull out the pieces we need (empty dictionary if a diagnostic is missing)
    arm = arm_cfg["arm"]
    model_cfg = arm_cfg["model"]
    primary = metrics["primary"]
    amplification = metrics.get("amplification", {})
    limit_cycle_result = metrics.get("limit_cycle", {})
    certificate = metrics.get("certificate", {})
    training_info = metrics["training"]

    # Step 2: build the row
    row = {
        # which run this is
        "arm": arm["name"],
        "seed": seed,
        "group": arm.get("group", ""),
        "role": arm.get("role", ""),
        "training_regime": arm_cfg["training"]["regime"],
        "model": model_cfg["name"],
        "decay_mode": model_cfg.get("decay_mode", ""),
        "parameters": metrics["parameters"],

        # main errors
        "diverged": metrics["diverged"],
        "one_step_rmse": primary["one_step_rmse"],
        "rollout_rmse": primary["rollout_rmse"],
        "error_at_10": primary.get("error_at_10"),
        "error_at_50": primary.get("error_at_50"),
        "growth_1_to_50": primary.get("growth_1_to_end"),
        "other_regime_rollout_rmse": metrics["other_regime"]["rollout_rmse"],

        # latent certificate
        "certificate_holds_at_end": certificate.get("certificate_holds_at_end"),
        "fraction_decay_positive": certificate.get("fraction_decay_positive"),

        # error amplification (model versus true system)
        "model_amplification_median": amplification.get("model_amplification_median"),
        "true_amplification_median": amplification.get("true_amplification_median"),
        "amplification_ratio_median": amplification.get("amplification_ratio_median"),

        # limit cycle
        "limit_cycle_amplitude_error": limit_cycle_result.get("amplitude_error"),
        "limit_cycle_period_error": limit_cycle_result.get("period_error"),
        "limit_cycle_status": limit_cycle_result.get("status"),

        # training information
        "nonfinite_steps": training_info["nonfinite_steps"],
        "epochs_run": training_info["epochs_run"],
        "train_minutes": training_info["train_minutes"],
    }

    return row