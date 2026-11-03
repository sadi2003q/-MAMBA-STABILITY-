"""How slowly does a trained model's memory fade? Decay factors A_bar = exp(Delta * A) on test windows.

python tools/diagnose_memory_margin.py \
    --runs outputs/exp03_chain_mamba/runs/chain_mamba_recipe_seed0 \
           models/seed0/exp01d_certificate_ablation_confirm/certified_chained_seed0

For every layer: median and largest decay factor, share of factors above 0.95 and 0.99, the step size Delta,
and the memory's steady-state gain 1 / (1 - largest factor). The gain is how strongly a mistake that enters
the memory is fed back under chained prediction (docs/chain_mamba_plan.md, Section 3). Works for the full
Mamba (mamba_full) and Chain-Mamba (chain_mamba); other models are skipped. No training, processor only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.data.datasets import build_data  # noqa: E402
from src.models.mamba_block import MambaLayer  # noqa: E402
from src.models.mamba_chain import ChainMambaLayer  # noqa: E402
from src.models.registry import build_model  # noqa: E402
from src.training.trainer import attach_data_statistics  # noqa: E402
from src.utils.config import load_yaml_with_inherit  # noqa: E402


@torch.no_grad()
def layer_decays(model, inp):
    """Decay factors and step sizes of every layer for a batch of teacher-forced inputs."""
    out = []
    h = model.core.embed(inp)
    for layer in model.core.layers:
        if isinstance(layer, ChainMambaLayer):
            xi, _ = layer.in_proj(layer.norm(h)).chunk(2, dim=-1)
            xc, _ = layer._conv(xi, h.new_zeros(h.shape[0], layer.E, layer.d_conv - 1))
            xc = F.silu(xc)
            dt_in, _, _ = torch.split(layer.x_proj(xc), [layer.dt_rank, layer.d_state, layer.d_state], dim=-1)
            dt = layer.delta_floor + F.softplus(layer.dt_proj(dt_in))
        elif isinstance(layer, MambaLayer):
            xi, _ = layer.in_proj(layer.norm(h)).chunk(2, dim=-1)
            buf = h.new_zeros(h.shape[0], layer.E, layer.d_conv - 1)
            xc = F.silu(layer.conv(torch.cat([buf, xi.transpose(1, 2)], 2))).transpose(1, 2)
            dt_in, _, _ = torch.split(layer.x_proj(xc), [layer.dt_rank, layer.d_state, layer.d_state], dim=-1)
            dt = F.softplus(layer.dt_proj(dt_in))
        else:
            return None
        a = torch.exp(dt.unsqueeze(-1) * layer.decay.A())
        out.append((a.flatten().numpy(), dt.flatten().numpy()))
        h, _ = layer.sequence(h)
    return out


def main():
    p = argparse.ArgumentParser(description="Decay factors of trained state-space models.")
    p.add_argument("--runs", nargs="+", required=True, help="run folders holding config.yaml and checkpoint.pt")
    p.add_argument("--windows", type=int, default=256)
    args = p.parse_args()
    data = None
    for run in args.runs:
        run = Path(run)
        cfg = load_yaml_with_inherit(run / "config.yaml")
        if data is None:
            data = build_data(cfg)
        model = build_model(cfg)
        attach_data_statistics(model, data)
        ck = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        split = data["splits"]["test"]
        x, u = split.xw[:args.windows], split.uw[:args.windows]
        res = layer_decays(model, model._inp(x[:, :-1], u, 0.0))
        print(f"\n{run.name}  ({cfg['model']['name']})")
        if res is None:
            print("  no selective state-space layers: skipped")
            continue
        for i, (a, dt) in enumerate(res):
            print(f"  layer {i}: decay median {np.median(a):.3f}, largest {a.max():.4f}, "
                  f"share > 0.95 {np.mean(a > 0.95):.2f}, share > 0.99 {np.mean(a > 0.99):.2f}, "
                  f"step size median {np.median(dt):.3f}, memory gain 1/(1 - largest) {1.0 / (1.0 - a.max()):.0f}")
        if hasattr(model.core, "rho_max"):
            r = model.core.rho_max()
            if r < 1.0:
                print(f"  guaranteed bound on every decay factor: {r:.4f}  (memory gain at most {1.0 / (1.0 - r):.1f})")
            else:
                print("  no margin: decay factors are only guaranteed to be below 1 (the plain certificate)")


if __name__ == "__main__":
    main()
