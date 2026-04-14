#!/usr/bin/env python3
"""Benchmark the kinematic cost computation path.

Measures the Pose round-trip: get_matrix (FK Pose -> matrix) + from_matrix (matrix -> Pose)
+ distance, vs directly using the FK Pose and only converting the desired side.

Usage:
    pixi shell
    python -m cutamp.scripts.benchmark_kinematic_cost
"""

import torch
from curobo.types.math import Pose


def _random_pose(batch, device="cuda"):
    """Create a random Pose with position + quaternion."""
    position = torch.randn(batch, 3, device=device)
    quat = torch.randn(batch, 4, device=device)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    return Pose(position=position, quaternion=quat)


def _pose_error_from_mats(mat_a, mat_b):
    """Old path: convert both mats back to Pose, then distance."""
    return Pose.from_matrix(mat_a).distance(Pose.from_matrix(mat_b))


def benchmark_old(batch, n_iters=500, n_warmup=50):
    """Old path: Pose -> get_matrix -> store -> from_matrix -> distance."""
    device = "cuda"

    for _ in range(n_warmup):
        pose_a = _random_pose(batch, device)
        pose_b = _random_pose(batch, device)
        _pose_error_from_mats(pose_a.get_matrix(), pose_b.get_matrix())

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iters):
        pose_a = _random_pose(batch, device)
        pose_b = _random_pose(batch, device)
        # Simulate rollout storing a matrix + kinematic_costs rebuilding Pose on both sides.
        _pose_error_from_mats(pose_a.get_matrix(), pose_b.get_matrix())
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / n_iters


def benchmark_new(batch, n_iters=500, n_warmup=50):
    """New path: keep FK Pose, only convert desired side from matrix."""
    device = "cuda"

    for _ in range(n_warmup):
        pose_fk = _random_pose(batch, device)
        pose_desired = _random_pose(batch, device)
        mat_desired = pose_desired.get_matrix()
        pose_desired_rebuilt = Pose.from_matrix(mat_desired)
        pose_fk.distance(pose_desired_rebuilt)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iters):
        pose_fk = _random_pose(batch, device)
        pose_desired = _random_pose(batch, device)
        # Simulate rollout: desired side still needs matrix -> Pose conversion
        mat_desired = pose_desired.get_matrix()
        # Simulate kinematic_costs: FK Pose used directly, only desired side converted
        pose_desired_rebuilt = Pose.from_matrix(mat_desired)
        pose_fk.distance(pose_desired_rebuilt)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / n_iters


def benchmark_both_poses(batch, n_iters=500, n_warmup=50):
    """Best case: both sides already have Pose, no matrix conversion for distance."""
    device = "cuda"

    for _ in range(n_warmup):
        pose_a = _random_pose(batch, device)
        pose_b = _random_pose(batch, device)
        pose_a.distance(pose_b)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iters):
        pose_a = _random_pose(batch, device)
        pose_b = _random_pose(batch, device)
        pose_a.distance(pose_b)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / n_iters


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print("Kinematic cost benchmark: Pose round-trip overhead")
    print(f"Device: {torch.cuda.get_device_name()}")

    # Realistic sizes: 512 particles * 10 timesteps = 5120, or 512 * 12 = 6144
    for batch in [512 * 8, 512 * 10, 512 * 12]:
        print(f"\n{'='*60}")
        print(f"  batch={batch}")
        print(f"{'='*60}")

        t_old = benchmark_old(batch)
        t_new = benchmark_new(batch)
        t_both = benchmark_both_poses(batch)

        print(f"  Old (matrix round-trip):      {t_old:.3f} ms/iter")
        print(f"  New (keep FK Pose):           {t_new:.3f} ms/iter")
        print(f"  Both Poses (no conversion):   {t_both:.3f} ms/iter")
        print(f"  Speedup (new):      {t_old / t_new:.2f}x")
        print(f"  Speedup (both):     {t_old / t_both:.2f}x")
