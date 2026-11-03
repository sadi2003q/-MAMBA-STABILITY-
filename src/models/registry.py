"""Model name -> class. Builds the full dynamics model from a resolved arm config."""

from __future__ import annotations

from src.models.dynamics import DynamicsModel
from src.models.lstm_dynamics import LSTMCore
from src.models.mamba_block import MambaCore
from src.models.mamba_chain import ChainMambaCore
from src.models.mlp_dynamics import MLPCore
from src.models.ssm_minimal import MinimalSelectiveSSM

# Published reference sizes for the Van der Pol example (Cevaal, de Jong and Lazar, 2026, Table 2).
REFERENCE_PARAMETERS = {"mamba_full": 3418, "lstm": 3052, "ssm_minimal": None, "chain_mamba": 3418, "mlp": None}


def build_core(model_cfg: dict, in_dim: int):
    m = dict(model_cfg)
    name = m.pop("name")
    if name == "ssm_minimal":
        return MinimalSelectiveSSM(in_dim, **m)
    if name == "mamba_full":
        return MambaCore(in_dim, **m)
    if name == "lstm":
        return LSTMCore(in_dim, **m)
    if name == "chain_mamba":
        return ChainMambaCore(in_dim, **m)
    if name == "mlp":
        return MLPCore(in_dim, **m)
    raise ValueError(f"Unknown model '{name}'. Known: ssm_minimal, mamba_full, lstm, chain_mamba, mlp.")


def build_model(cfg: dict, n_x: int | None = None, n_u: int | None = None) -> DynamicsModel:
    from src.systems import SYSTEMS
    sys_cls = SYSTEMS.get(cfg.get("system", {}).get("name", "vanderpol"), SYSTEMS["vanderpol"])
    n_x = sys_cls.n_x if n_x is None else n_x
    n_u = sys_cls.n_u if n_u is None else n_u
    core = build_core(cfg["model"], DynamicsModel.in_dim(n_x, n_u))
    readout = cfg.get("readout", {}) or {}
    return DynamicsModel(core, n_x, n_u, clamp=cfg.get("training", {}).get("clamp_state", 50.0),
                         linear_skip=readout.get("linear_skip", False),
                         increment_normalisation=readout.get("increment_normalisation", False))


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
