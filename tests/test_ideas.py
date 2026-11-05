"""Unit tests for the sampler-ideas math helpers (run where torch exists)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import nodes  # noqa: E402


def test_overshoot_sigma_math():
    # step 1.0 -> 0.8 with overshoot 0.5 lands at 0.7, clamped at 0
    assert abs(nodes.overshoot_sigma(1.0, 0.8, 0.5) - 0.7) < 1e-9
    assert nodes.overshoot_sigma(0.1, 0.0, 1.0) == 0.0
    assert abs(nodes.overshoot_sigma(1.0, 0.8, 0.0) - 0.8) < 1e-9


def test_warp_endpoints_and_monotonic():
    base = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.0])
    for curve in ("detail", "smooth"):
        out = nodes.warp_sigmas(base, curve, 1.0)
        assert abs(float(out[0]) - 10.0) < 1e-5
        assert float(out[-1]) == 0.0
        assert bool((out[:-1] > out[1:]).all()), curve


def test_warp_detail_densifies_low_sigma():
    base = nodes.flow_linear_shift_sigmas(30, 3.0, 1.0)
    out = nodes.warp_sigmas(base, "detail", 1.0)
    assert int((out < 0.5).sum()) > int((base < 0.5).sum())


def test_warp_linear_is_identity():
    base = nodes.flow_linear_shift_sigmas(30, 3.0, 1.0)
    assert torch.equal(nodes.warp_sigmas(base, "linear", 1.0), base)


def _const_velocity_model(c):
    def model(x, sigma, **kwargs):
        return x - sigma * c
    return model


def test_taylor_exact_on_constant_velocity():
    c = torch.full((1, 4), 0.3)
    x0 = torch.zeros(1, 4)
    sigmas = nodes.flow_linear_shift_sigmas(10, 3.0, 1.0)
    out = nodes._make_taylor_function()(_const_velocity_model(c),
                                        x0.clone(), sigmas)
    assert torch.allclose(out, x0 - c, atol=1e-5)


def test_overshoot_zero_is_euler_terminal_exact():
    c = torch.full((1, 4), 0.3)
    x0 = torch.zeros(1, 4)
    sigmas = nodes.flow_linear_shift_sigmas(10, 3.0, 1.0)
    out = nodes._make_overshoot_function(0.0, 0)(_const_velocity_model(c),
                                                 x0.clone(), sigmas)
    assert torch.allclose(out, x0 - c, atol=1e-5)
