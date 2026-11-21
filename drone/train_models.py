"""
Train the three learned drone models on the normal-mass flights (drone/data/flights.npz).

    A  chained   : the user's memory-free chained predictor, trained with the chained-prediction recipe
                   (uses src/losses/regimes.py and src/training/trainer.py from this repository)
    B  one_step  : the same model, trained one step at a time (teacher forcing)
    C  plain     : plain one-step network, trained one step at a time

Recipe (the same numbers as exp04_memory_removed.yaml):
    80 epochs, batch 128, Adam, learning rate 2e-3, cosine schedule with 3 warm-up epochs down to 1%,
    weight decay 1e-5, gradient clip 1.0, chain length 5 -> 50 over the first 25 epochs,
    extra one-step loss with weight 0.5.

Saved: drone/models/<name>_seed<seed>.pt  and  drone/models/normalisation.pt

Run:  python drone/train_models.py --model chained --seed 0
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drone import drone_models as dm
from src.losses.regimes import regime_loss
from src.training.trainer import make_optimizer, rollout_options

HERE = os.path.dirname(os.path.abspath(__file__))
HORIZON = 50
WINDOW_STRIDE = 5
N_TEST_FLIGHTS = 30          # the last 30 flights are never used for training (prediction check)
N_VALIDATION_FLIGHTS = 20    # the 20 flights before them pick the best epoch

TRAINING_SETTINGS = {
    "training": {
        "epochs": 80,
        "batch_size": 128,
        "lr": 2.0e-3,
        "lr_schedule": "cosine",
        "warmup_epochs": 3,
        "lr_final_fraction": 0.01,
        "weight_decay": 1.0e-5,
        "rollout_curriculum": {"start": 5, "epochs": 25},
        "teacher_aux_weight": 0.5,
        "state_noise_std": 0.0,
    }
}
GRADIENT_CLIP = 1.0


# ======================================================================================================
# Data
# ======================================================================================================
def load_flights():
    data = np.load(os.path.join(HERE, "data", "flights.npz"))
    measured = data["measured"]                                    # [F, T + 1, 12]
    commands = np.concatenate([data["applied"], data["previous"]], axis=2)    # [F, T, 8]
    return measured, commands


def split_flights(n_flights):
    test_start = n_flights - N_TEST_FLIGHTS
    validation_start = test_start - N_VALIDATION_FLIGHTS
    train = list(range(0, validation_start))
    validation = list(range(validation_start, test_start))
    test = list(range(test_start, n_flights))
    return train, validation, test


def normalisation_numbers(measured, commands, train):
    """Average and spread of every number, from the training flights only."""
    states = measured[train].reshape(-1, 12)
    all_commands = commands[train].reshape(-1, 8)
    numbers = {
        "state_mean": states.mean(axis=0),
        "state_std": states.std(axis=0) + 1e-6,
        "command_mean": all_commands.mean(axis=0),
        "command_std": all_commands.std(axis=0) + 1e-6,
    }

    # the typical size of one step's change, in normalised units (for the increment scaling)
    normalised = (measured[train] - numbers["state_mean"]) / numbers["state_std"]
    changes = normalised[:, 1:] - normalised[:, :-1]
    numbers["increment_std"] = changes.reshape(-1, 12).std(axis=0) + 1e-6
    return numbers


def cut_windows(measured, commands, flights, numbers):
    """Cut every flight into overlapping windows of 51 states and 50 commands (normalised)."""
    state_windows = []
    command_windows = []
    for f in flights:
        x = (measured[f] - numbers["state_mean"]) / numbers["state_std"]
        u = (commands[f] - numbers["command_mean"]) / numbers["command_std"]
        n_steps = u.shape[0]
        for start in range(0, n_steps - HORIZON + 1, WINDOW_STRIDE):
            state_windows.append(x[start:start + HORIZON + 1])
            command_windows.append(u[start:start + HORIZON])
    x_all = torch.tensor(np.array(state_windows), dtype=torch.float32)
    u_all = torch.tensor(np.array(command_windows), dtype=torch.float32)
    return x_all, u_all


# ======================================================================================================
# Losses
# ======================================================================================================
def plain_network_loss(model, x, u):
    """One-step loss for the plain network: predict every next state from the TRUE current state."""
    current = x[:, :-1]
    target = x[:, 1:]
    prediction = model.next_state(current, u)
    return torch.mean((prediction - target) ** 2)


@torch.no_grad()
def chained_error(model, x, u, horizon=HORIZON):
    """Average squared error of a chained prediction over `horizon` steps (normalised units)."""
    predictions = dm.roll_out(model, x[:, 0], u[:, :horizon])
    target = x[:, 1:horizon + 1]
    return float(torch.mean((predictions - target) ** 2))


# ======================================================================================================
# Training
# ======================================================================================================
def train(model_name, seed, epochs=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(2)

    # Step 1: data
    measured, commands = load_flights()
    train_flights, validation_flights, _ = split_flights(measured.shape[0])
    numbers = normalisation_numbers(measured, commands, train_flights)
    x_train, u_train = cut_windows(measured, commands, train_flights, numbers)
    x_val, u_val = cut_windows(measured, commands, validation_flights, numbers)
    print(f"training windows: {x_train.shape[0]}, validation windows: {x_val.shape[0]}")

    # Step 2: the model
    if model_name == "plain":
        model = dm.build_plain_network()
    else:
        model = dm.build_chained_predictor()
    model.set_increment_scale(numbers["increment_std"])
    print(f"model {model_name}: {dm.count_parameters(model)} parameters")

    # Step 3: optimiser and learning-rate schedule (the repository's own function)
    settings = {"training": dict(TRAINING_SETTINGS["training"])}
    if epochs is not None:
        settings["training"]["epochs"] = epochs
    total_epochs = settings["training"]["epochs"]
    optimizer, scheduler = make_optimizer(model, settings)

    # Step 4: the epochs
    generator = torch.Generator().manual_seed(seed)
    batch_size = settings["training"]["batch_size"]
    best_validation = math.inf
    best_state = None
    log_lines = []
    start_time = time.time()

    progress = tqdm(range(total_epochs), desc=model_name)
    for epoch in progress:
        model.train()
        order = torch.randperm(x_train.shape[0], generator=generator)
        options = rollout_options(settings["training"], epoch, HORIZON)
        epoch_losses = []

        for first in range(0, x_train.shape[0], batch_size):
            rows = order[first:first + batch_size]
            x = x_train[rows]
            u = u_train[rows]

            # 4a: the loss for this model
            if model_name == "chained":
                loss, _ = regime_loss(model, x, u, "chained", 0, HORIZON, **options)
            elif model_name == "one_step":
                loss, _ = regime_loss(model, x, u, "teacher_forcing", 0, HORIZON)
            else:
                loss = plain_network_loss(model, x, u)

            # 4b: one update of the weights
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        scheduler.step()

        # 4c: validation: 50-step chained error on the validation flights (what the planner needs)
        model.eval()
        validation = chained_error(model, x_val, u_val)
        if validation < best_validation:
            best_validation = validation
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        chain_length = options.get("train_horizon", 1)
        line = (f"epoch {epoch + 1} chain {chain_length} train {np.mean(epoch_losses):.5f}"
                f" validation_50_step {validation:.5f}")
        log_lines.append(line)
        progress.set_postfix(val=f"{validation:.4f}")

    minutes = (time.time() - start_time) / 60.0

    # Step 5: save the best epoch
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    model.load_state_dict(best_state)
    out_path = os.path.join(HERE, "models", f"{model_name}_seed{seed}.pt")
    torch.save({"model_name": model_name, "state_dict": best_state, "minutes": minutes,
                "parameters": dm.count_parameters(model), "best_validation": best_validation}, out_path)
    torch.save(numbers, os.path.join(HERE, "models", "normalisation.pt"))
    with open(os.path.join(HERE, "models", f"{model_name}_seed{seed}_log.txt"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"saved {out_path}  best 50-step validation error {best_validation:.5f}  ({minutes:.1f} min)")


def load_model(model_name, seed):
    saved = torch.load(os.path.join(HERE, "models", f"{model_name}_seed{seed}.pt"), weights_only=False)
    if model_name == "plain":
        model = dm.build_plain_network()
    else:
        model = dm.build_chained_predictor()
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["chained", "one_step", "plain"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="only for a quick code check")
    args = parser.parse_args()
    train(args.model, args.seed, args.epochs)
