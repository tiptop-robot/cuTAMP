#!/usr/bin/env python3
"""Benchmark per-object Warp kernel launches vs. concatenating all objects into one call.

Tests the claim that launching one small kernel per movable object is faster than
stacking all object spheres and launching one large kernel, due to better early-exit
behavior in the Warp kernel when n2 is small.

Usage:
    pixi shell
    python -m cutamp.scripts.benchmark_per_object_vs_concat
"""

import torch

from cutamp.costs_warp import sphere_to_sphere_overlap_warp


def _make_spheres(shape, device="cuda", spread=1.0, radius_range=(0.01, 0.05)):
    centers = torch.randn(*shape[:-1], 3, device=device) * spread
    radii = torch.rand(*shape[:-1], 1, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    return torch.cat([centers, radii], dim=-1)


def benchmark(n_objects, batch_shape, n_robot, n_obj_spheres, act_dist=0.01, n_iters=200, n_warmup=20):
    print(f"\n{'='*70}")
    print(f"  {n_objects} objects, batch={batch_shape}, n_robot={n_robot}, n_obj_spheres={n_obj_spheres}")
    print(f"  n_iters={n_iters}")
    print(f"{'='*70}")

    robot_shape = (*batch_shape, n_robot, 4)
    obj_shape = (*batch_shape, n_obj_spheres, 4)

    def make_inputs(requires_grad):
        robot = _make_spheres(robot_shape).requires_grad_(requires_grad)
        objects = [_make_spheres(obj_shape).requires_grad_(requires_grad) for _ in range(n_objects)]
        return robot, objects

    def per_object_fn(robot, objects):
        return sum(sphere_to_sphere_overlap_warp(robot, obj, act_dist) for obj in objects)

    def concat_fn(robot, objects):
        stacked = torch.cat(objects, dim=-2)  # (*batch, n_objects * n_obj_spheres, 4)
        return sphere_to_sphere_overlap_warp(robot, stacked, act_dist)

    def time_fn(fn, label, requires_grad):
        for _ in range(n_warmup):
            robot, objects = make_inputs(requires_grad)
            out = fn(robot, objects)
            if requires_grad:
                out.sum().backward()

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(n_iters):
            robot, objects = make_inputs(requires_grad)
            out = fn(robot, objects)
            if requires_grad:
                out.sum().backward()
        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        per_iter_ms = elapsed_ms / n_iters
        print(f"  {label}: {per_iter_ms:.3f} ms/iter")
        return per_iter_ms

    for requires_grad, mode in [(False, "fwd only"), (True, "fwd+bwd")]:
        print(f"\n  --- {mode} ---")
        t_per = time_fn(per_object_fn, "Per-object launches", requires_grad)
        t_cat = time_fn(concat_fn, "Single concat call  ", requires_grad)
        winner = "per-object" if t_per < t_cat else "concat"
        ratio = max(t_per, t_cat) / min(t_per, t_cat)
        print(f"  Winner: {winner} ({ratio:.2f}x faster)")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print("Per-object vs. concatenated sphere overlap benchmark")
    print(f"Device: {torch.cuda.get_device_name()}")

    # Blocks workload: 3 objects, 50 spheres each, 38 robot spheres
    benchmark(n_objects=3, batch_shape=(512, 8), n_robot=38, n_obj_spheres=50)

    # 5 blocks
    benchmark(n_objects=5, batch_shape=(512, 8), n_robot=38, n_obj_spheres=50)

    # More spheres per object
    benchmark(n_objects=3, batch_shape=(512, 8), n_robot=38, n_obj_spheres=200)
    benchmark(n_objects=5, batch_shape=(512, 8), n_robot=38, n_obj_spheres=200)

    # Stress test: many objects
    benchmark(n_objects=10, batch_shape=(512, 8), n_robot=38, n_obj_spheres=50)
    benchmark(n_objects=10, batch_shape=(512, 8), n_robot=38, n_obj_spheres=200)
