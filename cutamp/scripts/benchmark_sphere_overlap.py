#!/usr/bin/env python3
"""Benchmark Warp sphere-to-sphere overlap against the PyTorch reference.

Correctness tests are in tests/test_sphere_overlap.py (run with pytest).

Usage:
    pixi shell
    python -m cutamp.scripts.benchmark_sphere_overlap
"""

import torch

from cutamp.costs_warp import sphere_to_sphere_overlap_warp


def _pytorch_reference(spheres_1, spheres_2, activation_distance):
    centers_1, radii_1 = spheres_1[..., :3], spheres_1[..., 3]
    centers_2, radii_2 = spheres_2[..., :3], spheres_2[..., 3]
    diff = centers_1.unsqueeze(-2) - centers_2.unsqueeze(-3)
    dist_sq = (diff * diff).sum(dim=-1)
    dist = torch.sqrt(dist_sq + 1e-8)
    radii_sum = radii_1.unsqueeze(-1) + radii_2.unsqueeze(-2)
    penetration = radii_sum - dist + activation_distance
    return torch.relu(penetration).sum((-2, -1))


def _make_spheres(shape, device="cuda", spread=1.0, radius_range=(0.01, 0.05)):
    centers = torch.randn(*shape[:-1], 3, device=device) * spread
    radii = torch.rand(*shape[:-1], 1, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    return torch.cat([centers, radii], dim=-1)


def benchmark(label, batch_shape, n1, n2, act_dist=0.01, n_iters=100, n_warmup=10):
    """Benchmark forward+backward for both implementations."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {label}")
    print(f"  batch_shape={batch_shape}, n1={n1}, n2={n2}, act_dist={act_dist}")
    print(f"  n_iters={n_iters}, n_warmup={n_warmup}")
    print(f"{'='*60}")

    shape_1 = (*batch_shape, n1, 4)
    shape_2 = (*batch_shape, n2, 4)

    def time_fn(fn, label):
        for _ in range(n_warmup):
            s1 = _make_spheres(shape_1).requires_grad_(True)
            s2 = _make_spheres(shape_2).requires_grad_(True)
            out = fn(s1, s2, act_dist)
            out.sum().backward()

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(n_iters):
            s1 = _make_spheres(shape_1).requires_grad_(True)
            s2 = _make_spheres(shape_2).requires_grad_(True)
            out = fn(s1, s2, act_dist)
            out.sum().backward()
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        per_iter_ms = elapsed_ms / n_iters
        print(f"  {label}: {per_iter_ms:.3f} ms/iter ({elapsed_ms:.1f} ms total)")
        return per_iter_ms

    t_pytorch = time_fn(_pytorch_reference, "PyTorch")
    t_warp = time_fn(sphere_to_sphere_overlap_warp, "Warp   ")

    speedup = t_pytorch / t_warp
    print(f"  Speedup: {speedup:.2f}x")
    return speedup


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print("Sphere-to-sphere overlap benchmarks: Warp vs PyTorch")
    print(f"Device: {torch.cuda.get_device_name()}")

    benchmark("small (baseline)", batch_shape=(32, 2), n1=10, n2=50)
    benchmark("realistic (200 sph/obj)", batch_shape=(512, 8), n1=38, n2=200)
    benchmark("large (800 sph/obj)", batch_shape=(512, 8), n1=38, n2=800)
