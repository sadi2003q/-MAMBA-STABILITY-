"""Device selection: auto picks an NVIDIA graphics card, then Apple graphics, then the processor."""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402


def pick_device(name: str = "auto") -> torch.device:
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but no NVIDIA graphics card is available.")
    if name == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("--device mps requested but Apple graphics is not available.")
    return torch.device(name)


def describe(device: torch.device) -> str:
    if device.type == "cuda":
        return f"NVIDIA graphics card ({torch.cuda.get_device_name(device)})"
    if device.type == "mps":
        return "Apple graphics (may be slightly non-deterministic)"
    return "processor"
