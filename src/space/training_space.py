"""
Training one spacecraft predictor with YOUR recipe, reusing the certificate paper's code unchanged:
    model:      src/models/registry.build_model  (Chain-Mamba + anchor + increment scale)
    loss:       src/losses/regimes.regime_loss   (chained prediction + 0.5 x one-step loss)
    optimizer:  src/training/trainer.make_optimizer (Adam, cosine schedule with warm-up)
    curriculum: src/training/trainer.rollout_options (chain length grows 5 -> full horizon)

Three versions are trained per seed:
    classical            normal Chain-Mamba
    quantum              x_proj replaced by the quantum selection block, trained on perfect hardware
    quantum_noise_aware  the same, but every training batch gets random hardware defects ("practising with noise")

The weights with the lowest validation loss (perfect hardware) are kept, as in the main trainer.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import torch

from src.losses.regimes import regime_loss
from src.models.registry import build_model, count_parameters
from src.space.quantum import add_quantum_selection, set_hardware, set_random_training_defects
from src.training.trainer import make_optimizer, rollout_options
from src.utils.console import progress_bar
from src.utils.seed import set_seed


def build_space_model(cfg: dict, version: str, increment_std) -> torch.nn.Module:
    """Chain-Mamba for 6 states and 3 pushes; the quantum versions get the quantum selection block."""
    model = build_model(cfg, n_x=6, n_u=3)
    if version in ("quantum", "quantum_noise_aware"):
        quantum_cfg = cfg["quantum"]
        add_quantum_selection(model, int(quantum_cfg["qubits"]), int(quantum_cfg["circuit_layers"]))
    model.set_increment_scale(increment_std)
    return model


@torch.no_grad()
def validation_loss(model, data: dict, cfg: dict, device, batch_size: int = 512) -> float:
    """Chained loss over the full horizon on the validation windows (perfect hardware)."""
    model.eval()
    set_hardware(model)
    context_len = int(cfg["data"]["context_len"])
    horizon = int(cfg["data"]["horizon"])
    total = 0.0
    count = 0
    number = data["val_x"].shape[0]
    for start in range(0, number, batch_size):
        x = data["val_x"][start:start + batch_size].to(device)
        u = data["val_u"][start:start + batch_size].to(device)
        loss, _ = regime_loss(model, x, u, "chained", context_len, horizon)
        total = total + float(loss) * x.shape[0]
        count = count + x.shape[0]
    model.train()
    return total / max(1, count)


def train_space_model(cfg: dict, data: dict, version: str, seed: int, device, save_path: Path, log, run_label: str):
    """Train (or load, if already trained) one predictor. Returns the model in evaluation mode."""
    # Step 1: build the model with the seed fixed, so every run is repeatable
    set_seed(seed)
    model = build_space_model(cfg, version, data["increment_std"]).to(device)

    # Step 2: already trained earlier? load it and stop here
    if save_path.exists():
        saved = torch.load(save_path, map_location=device)
        model.load_state_dict(saved["model"])
        model.eval()
        log.say(f"  {run_label}: loaded from {save_path.name} ({count_parameters(model):,} parameters)")
        return model

    # Step 3: optimizer and settings (your recipe from the config)
    optimizer, scheduler = make_optimizer(model, cfg)
    training_cfg = cfg["training"]
    epochs = int(training_cfg["epochs"])
    batch_size = int(training_cfg["batch_size"])
    context_len = int(cfg["data"]["context_len"])
    horizon = int(cfg["data"]["horizon"])
    noise_aware = version == "quantum_noise_aware"
    python_random = random.Random(seed)
    batch_generator = torch.Generator().manual_seed(seed)
    number_of_windows = data["train_x"].shape[0]

    best_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    log.detail(f"{run_label}: {count_parameters(model):,} parameters, {epochs} epochs, {number_of_windows} windows")

    # Step 4: the epochs
    bar = progress_bar(epochs, run_label, "epoch")
    for epoch in range(epochs):
        options = rollout_options(training_cfg, epoch, horizon)
        order = torch.randperm(number_of_windows, generator=batch_generator)
        epoch_losses = []
        skipped = 0

        for start in range(0, number_of_windows, batch_size):
            batch_index = order[start:start + batch_size]
            x = data["train_x"][batch_index].to(device)
            u = data["train_u"][batch_index].to(device)

            # 4a: practising with noise: random hardware defects for this batch (quantum_noise_aware only)
            if noise_aware:
                set_random_training_defects(model, python_random, cfg["noise_aware_training"])
            else:
                set_hardware(model)

            # 4b: your recipe's loss: chained prediction (curriculum length) + 0.5 x one-step loss
            loss, _ = regime_loss(model, x, u, "chained", context_len, horizon, **options)

            # 4c: skip a step whose loss is not a finite number (counted in the log)
            if not torch.isfinite(loss):
                skipped = skipped + 1
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["grad_clip"]))
            optimizer.step()
            epoch_losses.append(float(loss))

        scheduler.step()

        # Step 5: validation on perfect hardware; keep the best weights
        set_hardware(model)
        current = validation_loss(model, data, cfg, device)
        if current < best_loss:
            best_loss = current
            best_state = copy.deepcopy(model.state_dict())
        average = sum(epoch_losses) / max(1, len(epoch_losses))
        log.detail(f"{run_label} epoch {epoch + 1}: train {average:.5f} validation {current:.5f} "
                   f"chain length {options.get('train_horizon', horizon)} skipped {skipped}")
        bar.set_postfix(validation=f"{current:.4f}")
        bar.update(1)
    bar.close()

    # Step 6: keep the best weights, save them
    model.load_state_dict(best_state)
    set_hardware(model)
    model.eval()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "best_validation_loss": best_loss, "version": version, "seed": seed}, save_path)
    log.say(f"  {run_label}: trained, best validation loss {best_loss:.5f} ({count_parameters(model):,} parameters)")
    return model


@torch.no_grad()
def chained_error(model, data: dict, cfg: dict, device, steps: int, max_windows: int = 1000, **hardware) -> float:
    """Root-mean-square chained-prediction error after `steps` steps on test windows (normalised units)."""
    model.eval()
    set_hardware(model, **hardware)
    context_len = int(cfg["data"]["context_len"])
    x = data["test_x"][:max_windows].to(device)
    u = data["test_u"][:max_windows].to(device)
    predictions, _ = model.chained(x[:, :context_len + 1], u, steps)
    truth = x[:, context_len + 1:context_len + 1 + steps]
    error = torch.sqrt(((predictions - truth) ** 2).mean())
    set_hardware(model)
    return float(error)
