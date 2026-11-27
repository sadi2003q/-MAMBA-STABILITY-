import os

import torch

from config_loader import load_config
from data import CONTEXT_LENGTH, HORIZON, build_data, typical_step_change
from evaluation import chained_error, one_step_error
from plain_predictor import count_parameter
from readout_predictor import ReadoutPredictor
from seed_helper import set_seed
from training import teacher_forcing_loss, train

# Step 1: settings and data (built once, shared by all versions)
config = load_config()
data = build_data()
normaliser = data["normaliser"]
test_states, test_pushes = data["windows"]["test"]
typical_change = typical_step_change(data)
print("typical change (output scale):", typical_change)

hidden_size = config["plain_predictor"]["hidden_size"]
settings = config["teacher_forcing_training"]
os.makedirs("outputs", exist_ok=True)

all_results = []

for version in config["stage4_versions"]:
    name = version["name"]
    print(f"\n===== Version {name} =====")

    # Step 2: same seed -> every version starts from the same body weights
    set_seed(config["model_seed"])
    model = ReadoutPredictor(
        hidden_size,
        predict_change=version["predict_change"],
        use_scaling=version["use_scaling"],
        change_scale=typical_change,
        use_anchor=version["use_anchor"]
    )
    print("parameters:", count_parameter(model))

    # Step 3: seed again -> every version sees batches in the same shuffled order
    set_seed(config["model_seed"])

    # every argument passed BY NAME, so the order can never go wrong
    train(
        model=model,
        batch_size=int(settings["batch_size"]),
        data=data,
        epochs=settings["epochs"],
        learning_rate=settings["learning_rate"],
        loss_function=teacher_forcing_loss
    )

    torch.save(model.state_dict(), f"outputs/stage4_{name}.pt")

    # Step 4: measure on the test windows (arguments by name again)
    one_step = one_step_error(
        model=model,
        state_window=test_states,
        push_window=test_pushes,
        normaliser=normaliser
    )

    chained = chained_error(
        model=model,
        state_window=test_states,
        push_window=test_pushes,
        context_length=CONTEXT_LENGTH,
        horizon=HORIZON,
        normaliser=normaliser
    )

    result = {
        "name": name,
        "one_step": one_step,
        "step_1": chained["per_step"][0],
        "step_50": chained["per_step"][HORIZON - 1],
        "overall": chained["overall"],
        "blown_up": chained["blown_up"],
    }
    all_results.append(result)

    # Step 5: for the anchor version, look at what the anchor learned
    if version["use_anchor"]:
        print("learned anchor weights")
        print("  rows: position change, velocity change")
        print("  columns: position, velocity, push")
        print(model.anchor.weight.detach())

# Step 6: one table, every 50-step error compared with version A
baseline = all_results[0]["overall"]
print("\nRESULT - Stage 4, teacher forcing, seed 0, real units (smaller is better)")
print(f"{'version':26s} {'one-step':>9s} {'step 1':>8s} {'step 50':>8s} {'50-step':>8s} {'vs A':>8s} {'blown up':>9s}")
for result in all_results:
    change_percent = (result["overall"] - baseline) / baseline * 100
    print(f"{result['name']:26s} {result['one_step']:9.4f} {result['step_1']:8.4f} "
          f"{result['step_50']:8.4f} {result['overall']:8.4f} {change_percent:+7.0f}% "
          f"{result['blown_up']:9d}")