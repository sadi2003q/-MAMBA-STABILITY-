"""True systems. `make_system(cfg)` picks one by cfg["system"]["name"]."""

from src.systems.four_tank import FourTank
from src.systems.vanderpol import VanDerPol

SYSTEMS = {"vanderpol": VanDerPol, "four_tank": FourTank}


def make_system(cfg: dict):
    name = cfg["system"].get("name", "vanderpol")
    if name not in SYSTEMS:
        raise ValueError(f"Unknown system '{name}'. Known: {', '.join(SYSTEMS)}.")
    return SYSTEMS[name].from_config(cfg)
