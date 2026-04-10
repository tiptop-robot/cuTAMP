#!/usr/bin/env python3
"""Test and benchmark Warp sphere-to-sphere overlap against the PyTorch reference.

Usage:
    pixi shell
    python -m cutamp.scripts.test_sphere_overlap
"""

import torch
import time


def pytorch_reference(spheres_1, spheres_2, activation_distance):
    """Reference implementation from costs.py (_sphere_to_sphere_overlap)."""
    centers_1, radii_1 = spheres_1[..., :3], spheres_1[..., 3]
    centers_2, radii_2 = spheres_2[..., :3], spheres_2[..., 3]

    diff = centers_1.unsqueeze(-2) - centers_2.unsqueeze(-3)
    dist_sq = (diff * diff).sum(dim=-1)
    dist = torch.sqrt(dist_sq + 1e-8)

    radii_sum = radii_1.unsqueeze(-1) + radii_2.unsqueeze(-2)
    penetration = radii_sum - dist + activation_distance

    return torch.relu(penetration).sum((-2, -1))


def make_spheres(shape, device="cuda", spread=1.0, radius_range=(0.01, 0.05)):
    """Create random sphere tensors. shape = (*batch, n_spheres, 4)."""
    centers = torch.randn(*shape[:-1], 3, device=device) * spread
    radii = torch.rand(*shape[:-1], 1, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    return torch.cat([centers, radii], dim=-1)


def check_close(name, a, b, atol=1e-5, rtol=1e-5):
    max_err = (a - b).abs().max().item()
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}: max_abs_err={max_err:.2e}")
    return ok


def test_correctness(label, batch_shape, n1, n2, act_dist=0.01):
    """Compare Warp vs PyTorch output and gradients."""
    from cutamp.costs_warp import sphere_to_sphere_overlap_warp

    print(f"\n{'='*60}")
    print(f"Test: {label}")
    print(f"  batch_shape={batch_shape}, n1={n1}, n2={n2}, act_dist={act_dist}")
    print(f"{'='*60}")

    shape_1 = (*batch_shape, n1, 4)
    shape_2 = (*batch_shape, n2, 4)

    s1 = make_spheres(shape_1)
    s2 = make_spheres(shape_2)

    # Forward — PyTorch reference
    s1_ref = s1.clone().requires_grad_(True)
    s2_ref = s2.clone().requires_grad_(True)
    out_ref = pytorch_reference(s1_ref, s2_ref, act_dist)

    # Forward — Warp
    s1_warp = s1.clone().requires_grad_(True)
    s2_warp = s2.clone().requires_grad_(True)
    out_warp = sphere_to_sphere_overlap_warp(s1_warp, s2_warp, act_dist)

    all_ok = True
    all_ok &= check_close("forward output", out_ref, out_warp, atol=1e-4)

    # Backward — use same upstream gradient
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_warp.backward(grad_out)

    all_ok &= check_close("grad_spheres_1", s1_ref.grad, s1_warp.grad, atol=1e-4)
    all_ok &= check_close("grad_spheres_2", s2_ref.grad, s2_warp.grad, atol=1e-4)

    return all_ok


def test_gradcheck():
    """Run torch.autograd.gradcheck on the Warp implementation with small tensors."""
    from cutamp.costs_warp import sphere_to_sphere_overlap_warp

    print(f"\n{'='*60}")
    print("Test: torch.autograd.gradcheck (double precision)")
    print(f"{'='*60}")

    # gradcheck requires float64
    s1 = make_spheres((2, 3, 4)).double().requires_grad_(True)
    s2 = make_spheres((2, 4, 4)).double().requires_grad_(True)

    # Our warp kernel is fp32 only, so gradcheck with fp64 inputs won't work directly.
    # Instead we do a manual finite-difference check in fp32.
    print("  [SKIP] gradcheck requires fp64 but Warp kernel is fp32. Using manual finite-diff instead.")

    s1 = make_spheres((2, 3, 4)).requires_grad_(True)
    s2 = make_spheres((2, 4, 4)).requires_grad_(True)
    act_dist = 0.01

    out = sphere_to_sphere_overlap_warp(s1, s2, act_dist)
    grad_out = torch.ones_like(out)
    out.backward(grad_out)

    # Finite difference for s1
    eps = 1e-3
    grad_fd = torch.zeros_like(s1)
    for idx in range(s1.numel()):
        s1_plus = s1.detach().clone()
        s1_plus.view(-1)[idx] += eps
        out_plus = pytorch_reference(s1_plus, s2.detach(), act_dist).sum()
        s1_minus = s1.detach().clone()
        s1_minus.view(-1)[idx] -= eps
        out_minus = pytorch_reference(s1_minus, s2.detach(), act_dist).sum()
        grad_fd.view(-1)[idx] = (out_plus - out_minus) / (2 * eps)

    ok = check_close("finite-diff grad_spheres_1", s1.grad, grad_fd, atol=1e-2)

    # Finite difference for s2
    grad_fd2 = torch.zeros_like(s2)
    for idx in range(s2.numel()):
        s2_plus = s2.detach().clone()
        s2_plus.view(-1)[idx] += eps
        out_plus = pytorch_reference(s1.detach(), s2_plus, act_dist).sum()
        s2_minus = s2.detach().clone()
        s2_minus.view(-1)[idx] -= eps
        out_minus = pytorch_reference(s1.detach(), s2_minus, act_dist).sum()
        grad_fd2.view(-1)[idx] = (out_plus - out_minus) / (2 * eps)

    ok &= check_close("finite-diff grad_spheres_2", s2.grad, grad_fd2, atol=1e-2)
    return ok


def benchmark(label, batch_shape, n1, n2, act_dist=0.01, n_iters=100, n_warmup=10):
    """Benchmark forward+backward for both implementations."""
    from cutamp.costs_warp import sphere_to_sphere_overlap_warp

    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"  batch_shape={batch_shape}, n1={n1}, n2={n2}, act_dist={act_dist}")
    print(f"  n_iters={n_iters}, n_warmup={n_warmup}")
    print(f"{'='*60}")

    shape_1 = (*batch_shape, n1, 4)
    shape_2 = (*batch_shape, n2, 4)

    def time_fn(fn, label):
        # Warmup
        for _ in range(n_warmup):
            s1 = make_spheres(shape_1).requires_grad_(True)
            s2 = make_spheres(shape_2).requires_grad_(True)
            out = fn(s1, s2, act_dist)
            out.sum().backward()

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(n_iters):
            s1 = make_spheres(shape_1).requires_grad_(True)
            s2 = make_spheres(shape_2).requires_grad_(True)
            out = fn(s1, s2, act_dist)
            out.sum().backward()
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        per_iter_ms = elapsed_ms / n_iters
        print(f"  {label}: {per_iter_ms:.3f} ms/iter ({elapsed_ms:.1f} ms total)")
        return per_iter_ms

    t_pytorch = time_fn(pytorch_reference, "PyTorch")
    t_warp = time_fn(sphere_to_sphere_overlap_warp, "Warp   ")

    speedup = t_pytorch / t_warp
    print(f"  Speedup: {speedup:.2f}x")
    return speedup


def test_no_overlap():
    """Test case where spheres are far apart (no overlaps at all)."""
    from cutamp.costs_warp import sphere_to_sphere_overlap_warp

    print(f"\n{'='*60}")
    print("Test: no overlap (all spheres far apart)")
    print(f"{'='*60}")

    s1 = torch.tensor([[[0.0, 0.0, 0.0, 0.01], [1.0, 0.0, 0.0, 0.01]]], device="cuda")
    s2 = torch.tensor([[[10.0, 10.0, 10.0, 0.01], [20.0, 20.0, 20.0, 0.01]]], device="cuda")
    act_dist = 0.0

    s1_w = s1.clone().requires_grad_(True)
    s2_w = s2.clone().requires_grad_(True)
    out = sphere_to_sphere_overlap_warp(s1_w, s2_w, act_dist)
    out.sum().backward()

    all_ok = True
    all_ok &= check_close("output should be zero", out, torch.zeros_like(out))
    all_ok &= check_close("grad_s1 should be zero", s1_w.grad, torch.zeros_like(s1_w.grad))
    all_ok &= check_close("grad_s2 should be zero", s2_w.grad, torch.zeros_like(s2_w.grad))
    return all_ok


def test_full_overlap():
    """Test case where spheres are at the same position (maximum overlap)."""
    from cutamp.costs_warp import sphere_to_sphere_overlap_warp

    print(f"\n{'='*60}")
    print("Test: full overlap (coincident spheres)")
    print(f"{'='*60}")

    s1 = torch.tensor([[[0.0, 0.0, 0.0, 0.05]]], device="cuda")
    s2 = torch.tensor([[[0.0, 0.0, 0.0, 0.05]]], device="cuda")
    act_dist = 0.0

    s1_ref = s1.clone().requires_grad_(True)
    s2_ref = s2.clone().requires_grad_(True)
    out_ref = pytorch_reference(s1_ref, s2_ref, act_dist)

    s1_w = s1.clone().requires_grad_(True)
    s2_w = s2.clone().requires_grad_(True)
    out_warp = sphere_to_sphere_overlap_warp(s1_w, s2_w, act_dist)

    all_ok = check_close("full overlap output", out_ref, out_warp)

    out_ref.sum().backward()
    out_warp.sum().backward()
    all_ok &= check_close("full overlap grad_s1", s1_ref.grad, s1_w.grad)
    all_ok &= check_close("full overlap grad_s2", s2_ref.grad, s2_w.grad)
    return all_ok


def test_grouped_correctness(label, batch_shape, n1, n2, n_groups, act_dist=0.01):
    """Compare grouped Warp kernel vs ungrouped Warp kernel output and gradients."""
    from cutamp.costs_warp import sphere_to_sphere_overlap_grouped_warp, sphere_to_sphere_overlap_warp

    print(f"\n{'='*60}")
    print(f"Test grouped: {label}")
    print(f"  batch_shape={batch_shape}, n1={n1}, n2={n2}, n_groups={n_groups}, act_dist={act_dist}")
    print(f"{'='*60}")

    s1 = make_spheres((*batch_shape, n1, 4))
    s2 = make_spheres((*batch_shape, n2, 4))

    # Create group assignments — assign spheres round-robin to groups
    group_idx = torch.arange(n1, device="cuda", dtype=torch.int32) % n_groups

    # Compute per-group AABB from spheres_1. Since spheres_1 varies per batch element,
    # we need the AABB to be conservative (union over all batch elements).
    # group_aabb_1: (n_groups, 6) = [min_x, min_y, min_z, max_x, max_y, max_z]
    s1_flat = s1.reshape(-1, n1, 4)  # (flat_batch, n1, 4)
    group_aabb = torch.zeros(n_groups, 6, device="cuda")
    for g in range(n_groups):
        mask = group_idx == g
        g_spheres = s1_flat[:, mask]  # (flat_batch, n_in_group, 4)
        centers = g_spheres[..., :3]
        radii = g_spheres[..., 3]
        max_r = radii.max().item()
        mins = centers.min(dim=0).values.min(dim=0).values  # (3,)
        maxs = centers.max(dim=0).values.max(dim=0).values  # (3,)
        group_aabb[g, :3] = mins - max_r - act_dist
        group_aabb[g, 3:] = maxs + max_r + act_dist

    # Compute per-batch AABB for spheres_2
    s2_flat = s2.reshape(-1, n2, 4)
    centers_2 = s2_flat[..., :3]
    radii_2 = s2_flat[..., 3]
    max_r2 = radii_2.max(dim=-1).values  # (flat_batch,)
    mins_2 = centers_2.min(dim=1).values  # (flat_batch, 3)
    maxs_2 = centers_2.max(dim=1).values  # (flat_batch, 3)
    batch_aabb_2 = torch.cat([mins_2 - max_r2.unsqueeze(-1), maxs_2 + max_r2.unsqueeze(-1)], dim=-1)
    batch_aabb_2 = batch_aabb_2.reshape(*batch_shape, 6)

    # Ungrouped reference
    s1_ref = s1.clone().requires_grad_(True)
    s2_ref = s2.clone().requires_grad_(True)
    out_ref = sphere_to_sphere_overlap_warp(s1_ref, s2_ref, act_dist)

    # Grouped
    s1_grp = s1.clone().requires_grad_(True)
    s2_grp = s2.clone().requires_grad_(True)
    out_grp = sphere_to_sphere_overlap_grouped_warp(s1_grp, s2_grp, act_dist, group_aabb, group_idx, batch_aabb_2)

    all_ok = True
    all_ok &= check_close("grouped forward output", out_ref, out_grp, atol=1e-4)

    # Backward
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_grp.backward(grad_out)

    all_ok &= check_close("grouped grad_spheres_1", s1_ref.grad, s1_grp.grad, atol=1e-4)
    all_ok &= check_close("grouped grad_spheres_2", s2_ref.grad, s2_grp.grad, atol=1e-4)
    return all_ok


def benchmark_grouped(label, batch_shape, n1, n2, n_groups, act_dist=0.01, n_iters=100, n_warmup=10):
    """Benchmark grouped vs ungrouped."""
    from cutamp.costs_warp import sphere_to_sphere_overlap_grouped_warp, sphere_to_sphere_overlap_warp

    print(f"\n{'='*60}")
    print(f"Benchmark grouped: {label}")
    print(f"  batch_shape={batch_shape}, n1={n1}, n2={n2}, n_groups={n_groups}")
    print(f"{'='*60}")

    group_idx = torch.arange(n1, device="cuda", dtype=torch.int32) % n_groups

    def time_fn(fn, label):
        for _ in range(n_warmup):
            s1 = make_spheres((*batch_shape, n1, 4)).requires_grad_(True)
            s2 = make_spheres((*batch_shape, n2, 4)).requires_grad_(True)
            out = fn(s1, s2)
            out.sum().backward()

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(n_iters):
            s1 = make_spheres((*batch_shape, n1, 4)).requires_grad_(True)
            s2 = make_spheres((*batch_shape, n2, 4)).requires_grad_(True)
            out = fn(s1, s2)
            out.sum().backward()
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        per_iter_ms = elapsed_ms / n_iters
        print(f"  {label}: {per_iter_ms:.3f} ms/iter ({elapsed_ms:.1f} ms total)")
        return per_iter_ms

    def ungrouped_fn(s1, s2):
        return sphere_to_sphere_overlap_warp(s1, s2, act_dist)

    def grouped_fn(s1, s2):
        # Compute AABBs (included in timing since it's part of the real workload)
        s1_flat = s1.detach().reshape(-1, n1, 4)
        s2_flat = s2.detach().reshape(-1, n2, 4)

        group_aabb = torch.zeros(n_groups, 6, device="cuda")
        for g in range(n_groups):
            mask = group_idx == g
            g_spheres = s1_flat[:, mask]
            centers = g_spheres[..., :3]
            radii = g_spheres[..., 3]
            max_r = radii.max().item()
            mins = centers.min(dim=0).values.min(dim=0).values
            maxs = centers.max(dim=0).values.max(dim=0).values
            group_aabb[g, :3] = mins - max_r - act_dist
            group_aabb[g, 3:] = maxs + max_r + act_dist

        c2 = s2_flat[..., :3]
        r2 = s2_flat[..., 3]
        mr2 = r2.max(dim=-1).values
        batch_aabb_2 = torch.cat([
            c2.min(dim=1).values - mr2.unsqueeze(-1),
            c2.max(dim=1).values + mr2.unsqueeze(-1),
        ], dim=-1).reshape(*s2.shape[:-2], 6)

        return sphere_to_sphere_overlap_grouped_warp(s1, s2, act_dist, group_aabb, group_idx, batch_aabb_2)

    t_ungrouped = time_fn(ungrouped_fn, "Ungrouped")
    t_grouped = time_fn(grouped_fn, "Grouped  ")

    speedup = t_ungrouped / t_grouped
    print(f"  Speedup: {speedup:.2f}x")
    return speedup


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print("Sphere-to-sphere overlap: Warp vs PyTorch")
    print(f"Device: {torch.cuda.get_device_name()}")

    all_passed = True

    # Edge cases
    all_passed &= test_no_overlap()
    all_passed &= test_full_overlap()

    # Correctness at various sizes
    all_passed &= test_correctness("small", batch_shape=(2,), n1=3, n2=4)
    all_passed &= test_correctness("medium", batch_shape=(32, 2), n1=10, n2=50)
    all_passed &= test_correctness("realistic", batch_shape=(512, 8), n1=38, n2=200)
    all_passed &= test_correctness("large", batch_shape=(512, 8), n1=38, n2=800)

    # Grouped correctness — must match ungrouped exactly
    all_passed &= test_grouped_correctness("grouped small", batch_shape=(4,), n1=12, n2=20, n_groups=3)
    all_passed &= test_grouped_correctness("grouped realistic", batch_shape=(512, 8), n1=65, n2=200, n_groups=12)

    # Gradient finite-diff check
    all_passed &= test_gradcheck()

    # Benchmarks
    benchmark("small (baseline)", batch_shape=(32, 2), n1=10, n2=50)
    benchmark("realistic (200 sph/obj)", batch_shape=(512, 8), n1=38, n2=200)
    benchmark("large (800 sph/obj)", batch_shape=(512, 8), n1=38, n2=800)

    # Grouped benchmarks (simulating robot: 65 spheres, 12 links)
    benchmark_grouped("robot vs object (200 sph)", batch_shape=(512, 8), n1=65, n2=200, n_groups=12)

    print(f"\n{'='*60}")
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print(f"{'='*60}")
