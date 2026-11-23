"""
Checks for the rotating-memory switch of Chain-Mamba (run BEFORE training; fails loudly).

    1. same number of trainable weights with rotation on and off
    2. rotation on draws the same random starting weights as rotation off (fair comparison, same seed)
    3. whole-sequence (parallel), whole-sequence (one step at a time) and step-by-step passes agree (< 1e-4)
    4. the stability certificate still holds: every |decay factor| <= rho_max = 0.951
    5. gradient reaches the new rotation speeds and the decay rates (not zero)

python tools/check_rotation.py \
    --config configs/pilots/pilot_c_slosh_rotation.yaml
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse  # noqa: E402

import torch  # noqa: E402

from src.models.dynamics import DynamicsModel  # noqa: E402
from src.models.registry import build_core  # noqa: E402
from src.utils.config import REPO_ROOT, load_yaml_with_inherit  # noqa: E402


def build(model_cfg: dict, seed: int):
    torch.manual_seed(seed)
    in_dim = DynamicsModel.in_dim(2, 1)          # [angle, turn rate, push, mask] = 4 numbers
    return build_core(model_cfg, in_dim)


def count(module) -> int:
    total = 0
    for parameter in module.parameters():
        if parameter.requires_grad:
            total = total + parameter.numel()
    return total


def main():
    parser = argparse.ArgumentParser(description="Check the rotating-memory switch")
    parser.add_argument("--config", default="configs/pilots/pilot_c_slosh_rotation.yaml")
    args = parser.parse_args()
    cfg = load_yaml_with_inherit(REPO_ROOT / args.config)

    rotating_cfg = dict(cfg["arms"]["chain_mamba_rotating"]["model"])
    plain_cfg = dict(cfg["arms"]["chain_mamba_memory"]["model"])
    results = []

    # ---------------------------------------------------------------- check 1: same size
    rotating = build(rotating_cfg, 0)
    plain = build(plain_cfg, 0)
    n_rot = count(rotating)
    n_plain = count(plain)
    results.append(("same number of trainable weights", n_rot == n_plain, f"rotating {n_rot:,} vs plain {n_plain:,}"))

    # ---------------------------------------------------------------- check 2: same random starting weights
    plain_weights = dict(plain.named_parameters())
    largest_difference = 0.0
    shared = 0
    for name, parameter in rotating.named_parameters():
        if name in plain_weights and plain_weights[name].shape == parameter.shape and "decay" not in name:
            difference = float((plain_weights[name] - parameter).abs().max())
            largest_difference = max(largest_difference, difference)
            shared = shared + 1
    results.append(("same random starting weights (seed 0)", largest_difference == 0.0,
                    f"{shared} shared weight tensors, largest difference {largest_difference:.1e}"))

    # ---------------------------------------------------------------- check 3: three passes agree
    torch.manual_seed(1)
    inp = torch.randn(3, 30, DynamicsModel.in_dim(2, 1))
    with torch.no_grad():
        y_parallel, state_parallel = rotating.sequence(inp, None, parallel=True)
        y_sequential, _ = rotating.sequence(inp, None, parallel=False)
        step_outputs = []
        state = None
        for t in range(inp.shape[1]):
            y_t, state = rotating.step(inp[:, t], state)
            step_outputs.append(y_t)
        y_steps = torch.stack(step_outputs, dim=1)
        # continuing from a saved memory must also agree
        first, mid_state = rotating.sequence(inp[:, :12], None, parallel=True)
        second, _ = rotating.sequence(inp[:, 12:], mid_state, parallel=True)
        y_split = torch.cat([first, second], dim=1)
    gap_1 = float((y_parallel - y_sequential).abs().max())
    gap_2 = float((y_parallel - y_steps).abs().max())
    gap_3 = float((y_parallel - y_split).abs().max())
    gap = max(gap_1, gap_2, gap_3)
    results.append(("parallel = one-at-a-time = step-by-step = split", gap < 1e-4,
                    f"largest gap {gap:.1e} (limit 1e-4)"))

    # ---------------------------------------------------------------- check 4: certificate
    largest_size = 0.0
    with torch.no_grad():
        for layer in rotating.layers:
            xc = torch.randn(500, layer.E) * 3.0
            a, _, _ = layer._selective(xc)
            largest_size = max(largest_size, float(a.abs().max()))
    bound = rotating.rho_max()
    results.append(("|decay factor| <= rho_max (certificate)", largest_size <= bound + 1e-7,
                    f"largest {largest_size:.4f} vs bound {bound:.4f}"))

    # ---------------------------------------------------------------- check 5: gradient reaches new parts
    rotating.zero_grad()
    y, _ = rotating.sequence(inp, None, parallel=True)
    loss = (y ** 2).mean()
    loss.backward()
    speed_gradients = []
    decay_gradients = []
    for layer in rotating.layers:
        speed_gradients.append(float(layer.rotation_speed.w_log.grad.norm()))
        decay_gradients.append(float(layer.decay.A_log.grad.norm()))
    gradient_ok = min(speed_gradients) > 0 and min(decay_gradients) > 0
    speed_text = ", ".join(f"{g:.2e}" for g in speed_gradients)
    decay_text = ", ".join(f"{g:.2e}" for g in decay_gradients)
    results.append(("gradient reaches rotation speeds and decay rates", gradient_ok,
                    f"speeds [{speed_text}]  decays [{decay_text}]"))

    # ---------------------------------------------------------------- report
    print("+" + "-" * 78 + "+")
    print("| " + "CHECK — rotating memory switch (Chain-Mamba)".ljust(77) + "|")
    print("+" + "-" * 78 + "+")
    all_passed = True
    for name, passed, detail in results:
        mark = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  {mark}  {name:<48} {detail}")
    print("+" + "-" * 78 + "+")
    if not all_passed:
        print("  Some checks FAILED: do not train. Send the output to Claude.")
        sys.exit(1)
    print("  All checks passed: safe to train.")


if __name__ == "__main__":
    main()
