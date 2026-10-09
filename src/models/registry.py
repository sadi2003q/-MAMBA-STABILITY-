"""Model name -> class. Builds the full dynamics model from a resolved arm config."""

from __future__ import annotations

from src.models.dynamics import DynamicsModel
from src.models.lstm_dynamics import LSTMCore
from src.models.mamba_block import MambaCore
from src.models.ssm_minimal import MinimalSelectiveSSM

# Published reference sizes for the Van der Pol example (Cevaal, de Jong and Lazar, 2026, Table 2).
REFERENCE_PARAMETERS = {"mamba_full": 3418, "lstm": 3052, "ssm_minimal": None}


def build_core(model_cfg: dict, in_dim: int):
    m = dict(model_cfg)
    name = m.pop("name")
    if name == "ssm_minimal":
        return MinimalSelectiveSSM(in_dim, **m)
    if name == "mamba_full":
        return MambaCore(in_dim, **m)
    if name == "lstm":
        return LSTMCore(in_dim, **m)
    raise ValueError(f"Unknown model '{name}'. Known: ssm_minimal, mamba_full, lstm.")


def build_model(cfg: dict, n_x: int = 2, n_u: int = 1) -> DynamicsModel:
    core = build_core(cfg["model"], DynamicsModel.in_dim(n_x, n_u))
    return DynamicsModel(core, n_x, n_u, clamp=cfg.get("training", {}).get("clamp_state", 50.0))


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
