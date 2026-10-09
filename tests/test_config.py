"""Configuration: inheritance and placeholder refusal."""
from pathlib import Path

import pytest

from src.utils.config import check_placeholders, deep_merge, load_experiment, resolve_arm

ROOT = Path(__file__).resolve().parents[1]


def test_deep_merge():
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"c": 3}}) == {"a": {"b": 1, "c": 3}}


def test_placeholder_refused():
    with pytest.raises(ValueError):
        check_placeholders({"training": {"lr": "PLACEHOLDER"}})


@pytest.mark.parametrize("name", ["exp01_certificate_ablation", "exp01b_certificate_ablation_full_mamba"])
def test_experiment_resolves(name):
    exp = load_experiment(ROOT / "configs" / "experiments" / f"{name}.yaml")
    for arm in exp["arms"]:
        cfg = resolve_arm(exp, arm)
        assert cfg["system"]["dt"] == 0.1
        assert cfg["training"]["regime"] == arm["training_regime"]
        quick = resolve_arm(exp, arm, quick=True)
        assert quick["training"]["epochs"] == 2
