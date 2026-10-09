"""Configuration loading: inheritance, deep merge, arm resolution, placeholder refusal."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
PLACEHOLDER = "PLACEHOLDER"


def deep_merge(base: dict, override: dict) -> dict:
    """Return a new dict: `override` merged into `base` (nested dicts merged, everything else replaced)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml_with_inherit(path: str | Path) -> dict:
    """Load a YAML file; files listed under `inherit` (relative to this file) are merged first, in order."""
    path = Path(path).resolve()
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    merged: dict = {}
    for parent in raw.pop("inherit", []) or []:
        merged = deep_merge(merged, load_yaml_with_inherit(path.parent / parent))
    return deep_merge(merged, raw)


def load_experiment(path: str | Path) -> dict:
    """Load an experiment config. Keeps the config's own folder so arm model paths can be resolved."""
    path = Path(path).resolve()
    cfg = load_yaml_with_inherit(path)
    cfg["_config_path"] = str(path)
    for key in ("name", "hypothesis", "math_signal", "arms", "decision_rule"):
        if key not in cfg:
            raise ValueError(f"Experiment config {path.name} is missing the required field '{key}'.")
    names = [a["name"] for a in cfg["arms"]]
    if len(names) != len(set(names)):
        raise ValueError("Arm names must be unique.")
    return cfg


def resolve_arm(exp_cfg: dict, arm: dict, quick: bool = False) -> dict:
    """Full configuration for one arm: experiment settings + model file + arm overrides (+ quick block)."""
    base = {k: v for k, v in exp_cfg.items() if k not in ("arms", "quick")}
    config_dir = Path(exp_cfg["_config_path"]).parent
    model_cfg = load_yaml_with_inherit(config_dir / arm["model"])
    cfg = deep_merge(base, model_cfg)
    cfg = deep_merge(cfg, arm.get("overrides", {}))
    cfg["training"] = dict(cfg.get("training", {}))
    cfg["training"]["regime"] = arm["training_regime"]
    cfg["arm"] = {k: v for k, v in arm.items() if k != "overrides"}
    if quick:
        cfg = deep_merge(cfg, exp_cfg.get("quick", {}))
        cfg["quick_mode"] = True
    else:
        cfg["quick_mode"] = False
    check_placeholders(cfg)
    if cfg["training"]["regime"] not in ("teacher_forcing", "chained", "direct"):
        raise ValueError(f"Unknown training regime '{cfg['training']['regime']}' in arm {arm['name']}.")
    return cfg


def check_placeholders(cfg: Any, where: str = "config") -> None:
    """Refuse to run if any value is still the literal PLACEHOLDER (set by a pending pre-sweep)."""
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            check_placeholders(v, f"{where}.{k}")
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            check_placeholders(v, f"{where}[{i}]")
    elif isinstance(cfg, str) and cfg.strip() == PLACEHOLDER:
        raise ValueError(f"{where} is still '{PLACEHOLDER}'. Fill it from the pre-sweep before running.")


def output_dir(exp_cfg: dict, quick: bool, override_root: str | None = None) -> Path:
    root = Path(override_root or exp_cfg.get("output_root", "outputs"))
    if not root.is_absolute():
        root = REPO_ROOT / root
    name = exp_cfg["name"] + ("_quick" if quick else "")
    return root / name


def save_yaml(obj: dict, path: str | Path) -> None:
    with open(path, "w") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False)
