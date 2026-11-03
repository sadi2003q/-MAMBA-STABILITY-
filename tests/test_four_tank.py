"""Four-tank system: analytic steady state, mass balance, non-negative levels, and model sizes."""
import numpy as np
import pytest

from src.systems.four_tank import FourTank


def steady_state(s: FourTank, u1: float, u2: float) -> np.ndarray:
    k = 1.0 / 3600.0
    level = lambda flow, a: (flow / a) ** 2 / (2 * s.g)          # a sqrt(2 g x) = flow
    x3 = level((1 - s.gamma_b) * k * u2, s.a[2])
    x4 = level((1 - s.gamma_a) * k * u1, s.a[3])
    x1 = level(s.gamma_a * k * u1 + s.a[2] * np.sqrt(2 * s.g * x3), s.a[0])
    x2 = level(s.gamma_b * k * u2 + s.a[3] * np.sqrt(2 * s.g * x4), s.a[1])
    return np.array([x1, x2, x3, x4])


def test_analytic_steady_state_is_a_fixed_point():
    s = FourTank()
    xs = steady_state(s, 2.0, 3.0)
    assert np.allclose(s.f(xs, np.array([2.0, 3.0])), 0.0, atol=1e-12)
    assert np.allclose(s.step(xs, [2.0, 3.0]), xs, atol=1e-10)


def test_simulation_converges_to_steady_state_and_stays_non_negative():
    s = FourTank()
    x = s.simulate(np.zeros(4), np.tile([2.0, 3.0], (3000, 1)))       # 4 hours
    assert np.all(x >= 0)
    assert np.allclose(x[-1], steady_state(s, 2.0, 3.0), rtol=1e-3)


def test_sample_map_is_contracting_at_mid_range():
    J = FourTank().jacobian(np.array([0.8, 0.9, 0.7, 1.0]))
    assert np.max(np.abs(np.linalg.eigvals(J))) < 1.0


def test_models_size_themselves_from_the_system():
    torch = pytest.importorskip("torch")
    from src.models.registry import build_model
    m = build_model({"system": {"name": "four_tank"}, "model": {"name": "mlp", "hidden": 52, "layers": 2},
                     "readout": {"linear_skip": True, "increment_normalisation": True}})
    x, u = torch.randn(3, 61, 4), torch.randn(3, 60, 2)
    preds, _ = m.chained(x[:, :11], u, 50)
    assert preds.shape == (3, 50, 4)
