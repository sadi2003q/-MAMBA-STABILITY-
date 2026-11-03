"""Chain-Mamba: forward-pass equivalence, gradient flow, the decay margin, read-out options, and the
chained-training options. Run with: pytest -q tests/test_chain_mamba.py"""
import math

import pytest

torch = pytest.importorskip("torch")

from src.losses.regimes import regime_loss  # noqa: E402
from src.models.registry import build_model, count_parameters  # noqa: E402

CHAIN = {"name": "chain_mamba", "d_model": 12, "d_state": 8, "d_conv": 4, "n_layers": 2, "expand": 1,
         "dt_rank": 1, "dt_min": 0.06, "dt_max": 0.5, "delta_floor": 0.05, "a_min": 1.0, "ffn_hidden": 20,
         "bias": True, "final_norm": False, "use_parallel_scan": True}
READOUT = {"linear_skip": True, "increment_normalisation": True}


def make(readout=READOUT, **over):
    torch.manual_seed(0)
    return build_model({"model": {**CHAIN, **over}, "readout": readout,
                        "training": {"clamp_state": 50.0}}).double()


def test_parameter_budget_matches_full_mamba():
    # full Mamba 3,282; Cevaal et al. 3,418; stay within 5% of the published size
    assert abs(count_parameters(make()) - 3418) / 3418 < 0.05


def test_step_matches_parallel_and_state_carries_over():
    m = make()
    inp = torch.randn(2, 60, 4, dtype=torch.float64)
    with torch.no_grad():
        y_par, _ = m.core.sequence(inp, parallel=True)
        y_seq, _ = m.core.sequence(inp, parallel=False)
        state, ys = None, []
        for t in range(60):
            y, state = m.core.step(inp[:, t], state)
            ys.append(y)
        _, st = m.core.sequence(inp[:, :25])
        y_cont, _ = m.core.sequence(inp[:, 25:], st)
    assert torch.allclose(y_par, y_seq, atol=1e-10)
    assert torch.allclose(y_par, torch.stack(ys, 1), atol=1e-10)
    assert torch.allclose(y_par[:, 25:], y_cont, atol=1e-10)


def test_decay_margin_holds_for_any_weights():
    m = make()
    rho = math.exp(-CHAIN["delta_floor"] * CHAIN["a_min"])
    assert abs(m.core.rho_max() - rho) < 1e-12
    with torch.no_grad():                       # push every parameter far away and check the bound again
        for p in m.parameters():
            p.add_(5.0 * torch.randn_like(p))
        for layer in m.core.layers:
            xc = torch.randn(64, 7, layer.E, dtype=torch.float64) * 10
            a, _, _ = layer._selective(xc)
            assert float(a.max()) <= rho + 1e-12
            assert float(layer.decay.A().max()) < -CHAIN["a_min"] + 1e-12


@pytest.mark.parametrize("regime", ["teacher_forcing", "chained", "direct"])
def test_gradient_reaches_every_parameter(regime):
    m = make().float()
    m.set_increment_scale([0.09, 0.39])
    x, u = torch.randn(4, 61, 2), torch.randn(4, 60, 1)
    loss, _ = regime_loss(m, x, u, regime, context_len=10, horizon=50)
    loss.backward()
    for n, p in m.named_parameters():
        assert p.grad is not None and float(p.grad.norm()) > 0, n


def test_new_training_options_change_only_what_they_should():
    m = make().float()
    x, u = torch.randn(4, 61, 2), torch.randn(4, 60, 1)
    with torch.no_grad():
        full, _ = regime_loss(m, x, u, "chained", 10, 50)
        explicit, _ = regime_loss(m, x, u, "chained", 10, 50, train_horizon=50)
        short, _ = regime_loss(m, x, u, "chained", 10, 50, train_horizon=5)
        with_tf, _ = regime_loss(m, x, u, "chained", 10, 50, teacher_weight=0.5)
        preds, _ = m.chained(x[:, :11], u, 5)
    assert torch.allclose(full, explicit)
    assert torch.allclose(short, torch.mean((preds - x[:, 11:16]) ** 2))
    assert float(with_tf) > float(full)


def test_readout_options_default_off_keep_old_models_unchanged():
    old = build_model({"model": {"name": "lstm", "hidden_size": 25}, "training": {"clamp_state": 50.0}})
    assert not hasattr(old, "skip") and "increment_scale" not in old.state_dict()


def test_anchor_alone_can_represent_a_linear_step():
    """With the core's head at zero, the anchor reproduces x_{t+1} = x_t + W [x_t, u_t] exactly."""
    m = make()
    with torch.no_grad():
        m.head.weight.zero_()
        m.head.bias.zero_()
        W = torch.randn(2, 4, dtype=torch.float64)
        m.skip.weight.copy_(W)
    x, u = torch.randn(3, 20, 2, dtype=torch.float64), torch.randn(3, 20, 1, dtype=torch.float64)
    pred = m.teacher_forced(x, u)
    inp = torch.cat([x, u, torch.zeros_like(x[..., :1])], -1)
    assert torch.allclose(pred, x + inp @ W.T, atol=1e-12)
