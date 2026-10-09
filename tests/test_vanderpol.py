"""True-system sanity: Jacobian matches finite differences; the limit cycle has the known period."""
import numpy as np

from src.systems.vanderpol import VanDerPol


def test_jacobian_matches_finite_differences():
    sys_ = VanDerPol(mu=1.0, dt=0.1)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(-3, 3, 2)
        J = sys_.jacobian(x)
        eps = 1e-6
        num = np.stack([(sys_.step(x + eps * e, 0.3) - sys_.step(x - eps * e, 0.3)) / (2 * eps)
                        for e in np.eye(2)], axis=1)
        assert np.allclose(J, num, atol=1e-6)


def test_limit_cycle_period_small_step():
    sys_ = VanDerPol(mu=1.0, dt=0.001)
    x = sys_.simulate(np.array([2.0, 0.0]), np.zeros(40000))[20000:, 0]
    up = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    period = np.mean(np.diff(up)) * 0.001
    assert abs(period - 6.6633) < 0.02          # continuous-time period for mu = 1


def test_without_minus_x1_has_no_oscillation():
    sys_ = VanDerPol(mu=1.0, dt=0.01, include_minus_x1=False)
    x = sys_.simulate(np.array([0.5, 0.1]), np.zeros(5000))
    assert np.all(x[:, 1] > 0)                 # velocity never changes sign: no oscillation at all
    assert abs(x[-1, 1]) < abs(x[2500, 1])     # and it is dying out towards an equilibrium
