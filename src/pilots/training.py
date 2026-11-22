"""
Training one predictor for a space pilot with YOUR recipe, reusing the certificate paper's code unchanged:
    model:      src/models/registry.build_model  (Chain-Mamba with memory removed + anchor + increment scale)
    loss:       src/losses/regimes.regime_loss   (chained prediction + 0.5 x one-step loss)
    optimizer:  src/training/trainer.make_optimizer (Adam, cosine schedule with warm-up)
    curriculum: src/training/trainer.rollout_options (chain length grows from 5 to the full horizon)

The same steps as src/space/training_space.py on the space-pilot branch, with the quantum parts removed and
the number of states / pushes taken from the system (9 and 3 for pilot A, 6 and 3 for pilot B).
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from src.data.datasets import Normaliser
from src.losses.regimes import regime_loss
from src.models.registry import build_model, count_parameters
from src.training.trainer import make_optimizer, rollout_options
from src.utils.console import progress_bar
from src.utils.seed import set_seed


# ==========================================================================================================
# Data: normalisation numbers and windows
# ==========================================================================================================
def make_normaliser(states: torch.Tensor, pushes: torch.Tensor) -> Normaliser:
    """Mean and standard deviation of every state and push, from TRAINING trajectories only."""
    n_x = states.shape[2]
    n_u = pushes.shape[2]
    all_states = states.reshape(-1, n_x).numpy()
    all_pushes = pushes.reshape(-1, n_u).numpy()
    return Normaliser(
        x_mean=np.mean(all_states, axis=0),
        x_std=np.std(all_states, axis=0) + 1e-12,
        u_mean=np.mean(all_pushes, axis=0),
        u_std=np.std(all_pushes, axis=0) + 1e-12,
    )


def cut_windows(states: torch.Tensor, pushes: torch.Tensor, norm: Normaliser, length: int, stride: int):
    """
    Cut normalised windows of `length` states (and length - 1 pushes) from every trajectory.
        states [N, T + 1, n_x], pushes [N, T, n_u]  ->  windows [W, length, n_x], [W, length - 1, n_u]
    """
    x_mean = torch.tensor(norm.x_mean, dtype=torch.float32)
    x_std = torch.tensor(norm.x_std, dtype=torch.float32)
    u_mean = torch.tensor(norm.u_mean, dtype=torch.float32)
    u_std = torch.tensor(norm.u_std, dtype=torch.float32)

    state_windows = []
    push_windows = []
    total_states = states.shape[1]
    start = 0
    while start + length <= total_states:
        state_windows.append((states[:, start:start + length] - x_mean) / x_std)
        push_windows.append((pushes[:, start:start + length - 1] - u_mean) / u_std)
        start = start + stride
    return torch.cat(state_windows, dim=0), torch.cat(push_windows, dim=0)


def increment_scale(states: torch.Tensor, norm: Normaliser) -> np.ndarray:
    """Typical size of one true step-to-step change (normalised units), for the anchor's increment scale."""
    n_x = states.shape[2]
    x_mean = torch.tensor(norm.x_mean, dtype=torch.float32)
    x_std = torch.tensor(norm.x_std, dtype=torch.float32)
    normalised = (states - x_mean) / x_std
    changes = normalised[:, 1:] - normalised[:, :-1]
    return changes.reshape(-1, n_x).std(dim=0).numpy() + 1e-12


def build_windows(cfg: dict, train, val, test) -> dict:
    """
    train / val / test = (states, pushes) in physical units. Returns the windows, the normalisation numbers
    and the increment scale, in the format the trainer below expects.
    """
    data_cfg = cfg["data"]
    norm = make_normaliser(train[0], train[1])
    length = int(data_cfg["context_len"]) + int(data_cfg["horizon"]) + 1
    stride = int(data_cfg["window_stride"])
    data = {"norm": norm, "n_x": int(train[0].shape[2]), "n_u": int(train[1].shape[2])}
    data["train_x"], data["train_u"] = cut_windows(train[0], train[1], norm, length, stride)
    data["val_x"], data["val_u"] = cut_windows(val[0], val[1], norm, length, stride)
    data["test_x"], data["test_u"] = cut_windows(test[0], test[1], norm, length, stride)
    data["increment_std"] = increment_scale(train[0], norm)
    return data


# ==========================================================================================================
# Training
# ==========================================================================================================
def build_predictor_model(cfg: dict, data: dict) -> torch.nn.Module:
    """The model from the config (memory removed: model.mixer = false), with the increment scale attached."""
    model = build_model(cfg, n_x=data["n_x"], n_u=data["n_u"])
    model.set_increment_scale(data["increment_std"])
    return model


@torch.no_grad()
def validation_loss(model, data: dict, cfg: dict, device, batch_size: int = 512) -> float:
    """Chained loss over the full horizon on the validation windows."""
    model.eval()
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


def train_predictor(cfg: dict, data: dict, seed: int, device, save_path: Path, log, run_label: str):
    """Train (or load, if already trained) one predictor. Returns the model in evaluation mode."""
    # Step 1: build the model with the seed fixed, so every run is repeatable
    set_seed(seed)
    model = build_predictor_model(cfg, data).to(device)

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

            # 4a: your recipe's loss: chained prediction (curriculum length) + 0.5 x one-step loss
            loss, _ = regime_loss(model, x, u, "chained", context_len, horizon, **options)

            # 4b: skip a step whose loss is not a finite number (counted in the log)
            if not torch.isfinite(loss):
                skipped = skipped + 1
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training_cfg["grad_clip"]))
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        scheduler.step()

        # Step 5: validation; keep the best weights
        current = validation_loss(model, data, cfg, device)
        if current < best_loss:
            best_loss = current
            best_state = copy.deepcopy(model.state_dict())
        average = sum(epoch_losses) / max(1, len(epoch_losses))
        log.detail(f"{run_label} epoch {epoch + 1}: train {average:.6f} validation {current:.6f} "
                   f"chain length {options.get('train_horizon', horizon)} skipped {skipped}")
        bar.set_postfix(validation=f"{current:.5f}")
        bar.update(1)
    bar.close()

    # Step 6: keep the best weights, save them
    model.load_state_dict(best_state)
    model.eval()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "best_validation_loss": best_loss, "seed": seed}, save_path)
    log.say(f"  {run_label}: trained, best validation loss {best_loss:.6f} ({count_parameters(model):,} parameters)")
    return model
