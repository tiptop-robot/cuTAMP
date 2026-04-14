"""Test Warp sphere-to-sphere overlap against the PyTorch reference.

Run with: pytest tests/test_sphere_overlap.py -v
"""

import pytest
import torch

from cutamp.costs import sphere_to_sphere_overlap_pytorch
from cutamp.costs_warp import sphere_to_sphere_overlap_warp

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_spheres(shape, device="cuda", spread=1.0, radius_range=(0.01, 0.05)):
    """Create random sphere tensors. shape = (*batch, n_spheres, 4)."""
    centers = torch.randn(*shape[:-1], 3, device=device) * spread
    radii = torch.rand(*shape[:-1], 1, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    return torch.cat([centers, radii], dim=-1)


@pytest.mark.parametrize(
    "label, batch_shape, n1, n2",
    [
        ("small", (2,), 3, 4),
        ("medium", (32, 2), 10, 50),
        ("realistic", (512, 8), 38, 200),
        ("large", (512, 8), 38, 800),
    ],
)
def test_forward_and_backward(label, batch_shape, n1, n2):
    """Compare Warp vs PyTorch forward output and gradients."""
    act_dist = 0.01
    shape_1 = (*batch_shape, n1, 4)
    shape_2 = (*batch_shape, n2, 4)

    s1 = make_spheres(shape_1)
    s2 = make_spheres(shape_2)

    # Forward — PyTorch reference
    s1_ref = s1.clone().requires_grad_(True)
    s2_ref = s2.clone().requires_grad_(True)
    out_ref = sphere_to_sphere_overlap_pytorch(s1_ref, s2_ref, act_dist)

    # Forward — Warp
    s1_warp = s1.clone().requires_grad_(True)
    s2_warp = s2.clone().requires_grad_(True)
    out_warp = sphere_to_sphere_overlap_warp(s1_warp, s2_warp, act_dist)

    torch.testing.assert_close(out_warp, out_ref, atol=1e-4, rtol=1e-5)

    # Backward — use same upstream gradient
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_warp.backward(grad_out)

    torch.testing.assert_close(s1_warp.grad, s1_ref.grad, atol=1e-4, rtol=1e-5)
    torch.testing.assert_close(s2_warp.grad, s2_ref.grad, atol=1e-4, rtol=1e-5)


def test_no_overlap():
    """Spheres far apart should produce zero cost and zero gradients."""
    s1 = torch.tensor([[[0.0, 0.0, 0.0, 0.01], [1.0, 0.0, 0.0, 0.01]]], device="cuda")
    s2 = torch.tensor([[[10.0, 10.0, 10.0, 0.01], [20.0, 20.0, 20.0, 0.01]]], device="cuda")

    s1_w = s1.clone().requires_grad_(True)
    s2_w = s2.clone().requires_grad_(True)
    out = sphere_to_sphere_overlap_warp(s1_w, s2_w, 0.0)
    out.sum().backward()

    torch.testing.assert_close(out, torch.zeros_like(out))
    torch.testing.assert_close(s1_w.grad, torch.zeros_like(s1_w.grad))
    torch.testing.assert_close(s2_w.grad, torch.zeros_like(s2_w.grad))


def test_mixed_overlap():
    """Mix of overlapping, non-overlapping, and coincident sphere pairs should match PyTorch reference."""
    s1 = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.05],  # overlaps with s2[0] (coincident) and s2[1] (partial)
                [1.0, 0.0, 0.0, 0.05],  # overlaps with s2[1] (partial)
                [10.0, 10.0, 10.0, 0.01],  # no overlap with anything
            ]
        ],
        device="cuda",
    )
    s2 = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.05],  # coincident with s1[0]
                [0.95, 0.0, 0.0, 0.05],  # close to s1[0] and s1[1]
                [20.0, 20.0, 20.0, 0.01],  # far from everything
            ]
        ],
        device="cuda",
    )
    act_dist = 0.0

    s1_ref = s1.clone().requires_grad_(True)
    s2_ref = s2.clone().requires_grad_(True)
    out_ref = sphere_to_sphere_overlap_pytorch(s1_ref, s2_ref, act_dist)

    s1_w = s1.clone().requires_grad_(True)
    s2_w = s2.clone().requires_grad_(True)
    out_warp = sphere_to_sphere_overlap_warp(s1_w, s2_w, act_dist)

    assert out_warp.item() > 0, "Expected some overlap"
    torch.testing.assert_close(out_warp, out_ref, atol=1e-5, rtol=1e-5)

    out_ref.sum().backward()
    out_warp.sum().backward()
    torch.testing.assert_close(s1_w.grad, s1_ref.grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s2_w.grad, s2_ref.grad, atol=1e-5, rtol=1e-5)


def test_finite_diff_gradients():
    """Verify Warp gradients against finite differences."""
    s1 = make_spheres((2, 3, 4)).requires_grad_(True)
    s2 = make_spheres((2, 4, 4)).requires_grad_(True)
    act_dist = 0.01

    out = sphere_to_sphere_overlap_warp(s1, s2, act_dist)
    out.sum().backward()

    eps = 1e-3

    # Finite difference for s1
    grad_fd = torch.zeros_like(s1)
    for idx in range(s1.numel()):
        s1_plus = s1.detach().clone()
        s1_plus.view(-1)[idx] += eps
        out_plus = sphere_to_sphere_overlap_pytorch(s1_plus, s2.detach(), act_dist).sum()
        s1_minus = s1.detach().clone()
        s1_minus.view(-1)[idx] -= eps
        out_minus = sphere_to_sphere_overlap_pytorch(s1_minus, s2.detach(), act_dist).sum()
        grad_fd.view(-1)[idx] = (out_plus - out_minus) / (2 * eps)

    torch.testing.assert_close(s1.grad, grad_fd, atol=1e-2, rtol=1e-2)

    # Finite difference for s2
    grad_fd2 = torch.zeros_like(s2)
    for idx in range(s2.numel()):
        s2_plus = s2.detach().clone()
        s2_plus.view(-1)[idx] += eps
        out_plus = sphere_to_sphere_overlap_pytorch(s1.detach(), s2_plus, act_dist).sum()
        s2_minus = s2.detach().clone()
        s2_minus.view(-1)[idx] -= eps
        out_minus = sphere_to_sphere_overlap_pytorch(s1.detach(), s2_minus, act_dist).sum()
        grad_fd2.view(-1)[idx] = (out_plus - out_minus) / (2 * eps)

    torch.testing.assert_close(s2.grad, grad_fd2, atol=1e-2, rtol=1e-2)
