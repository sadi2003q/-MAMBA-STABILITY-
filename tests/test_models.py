"""Equivalence of forward passes, gradient flow, the certificate, and the three regimes."""
import pytest

torch = pytest.importorskip("torch")

from src.models.common import parallel_scan, sequential_scan  # noqa: E402
from src.models.registry import build_model, count_parameters  # noqa: E402

CONFIGS = {
    "ssm_minimal": {"name": "ssm_minimal", "d_model": 32, "d_state": 16, "dt_min": 0.001, "dt_max": 0.1,
                    "decay_mode": "certified", "unstable_init_scale": 1.0, "use_parallel_scan": True},
    "mamba_full": {"name": "mamba_full", "d_model": 8, "d_state": 8, "d_conv": 10, "n_layers": 6, "expand": 1,
                   "dt_rank": 1, "dt_min": 0.001, "dt_max": 0.1, "bias": True, "decay_mode": "certified",
                   "unstable_init_scale": 1.0, "use_parallel_scan": True},
    "lstm": {"name": "lstm", "hidden_size": 25},
}


def make(name, **over):
    torch.manual_seed(0)
    return build_model({"model": {**CONFIGS[name], **over}, "training": {"clamp_state": 50.0}}).double()


def test_scan_matches_loop():
    g = torch.Generator().manual_seed(0)
    a = torch.rand(3, 37, 5, 4, generator=g, dtype=torch.float64)
    b = torch.randn(3, 37, 5, 4, generator=g, dtype=torch.float64)
    h0 = torch.randn(3, 5, 4, generator=g, dtype=torch.float64)
    assert torch.allclose(parallel_scan(a, b, h0), sequential_scan(a, b, h0), atol=1e-10)


@pytest.mark.parametrize("name", list(CONFIGS))
def test_step_matches_parallel(name):
    m = make(name)
    inp = torch.randn(2, 60, 4, dtype=torch.float64)
    with torch.no_grad():
        y_par, _ = m.core.sequence(inp, parallel=True)
        state, ys = None, []
        for t in range(60):
            y, state = m.core.step(inp[:, t], state)
            ys.append(y)
    assert torch.allclose(y_par, torch.stack(ys, 1), atol=1e-4)


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("regime", ["teacher_forcing", "chained", "direct"])
def test_gradient_reaches_every_parameter(name, regime):
    from src.losses.regimes import regime_loss
    m = make(name).float()
    x = torch.randn(4, 61, 2)
    u = torch.randn(4, 60, 1)
    loss, _ = regime_loss(m, x, u, regime, context_len=10, horizon=50)
    loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and float(p.grad.norm()) > 0, n


def test_chained_matches_teacher_forcing_when_fed_truth_first_step():
    m = make("ssm_minimal")
    x = torch.randn(2, 61, 2, dtype=torch.float64)
    u = torch.randn(2, 60, 1, dtype=torch.float64)
    with torch.no_grad():
        tf = m.teacher_forced(x[:, :-1], u)
        ch, _ = m.chained(x[:, :11], u, 1)
    assert torch.allclose(tf[:, 10], ch[:, 0], atol=1e-8)


def test_certificate_by_construction():
    for name in ("ssm_minimal", "mamba_full"):
        m = make(name)
        assert all(float(d.A().max()) < 0 for d in m.decay_modules())
        u = make(name, decay_mode="unstable_init")
        assert all(float(d.A().min()) > 0 for d in u.decay_modules())


def test_parameter_counts_near_reference():
    assert abs(count_parameters(make("mamba_full").float()) - 3418) / 3418 < 0.05
    assert abs(count_parameters(make("lstm").float()) - 3052) / 3052 < 0.05
