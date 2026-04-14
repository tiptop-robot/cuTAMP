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

from cutamp.costs import curobo_pose_error


def _random_pose(batch, device="cuda"):
    """Create a random Pose with position + quaternion."""
    position = torch.randn(batch, 3, device=device)
    quat = torch.randn(batch, 4, device=device)
    quat = quat / quat.norm(dim=-1, keepdim=True)
    return Pose(position=position, quaternion=quat)


def benchmark_current(batch, n_iters=500, n_warmup=50):
    """Current path: Pose -> get_matrix -> store -> from_matrix -> distance."""
    device = "cuda"

    for _ in range(n_warmup):
        pose_a = _random_pose(batch, device)
        pose_b = _random_pose(batch, device)
        mat_a = pose_a.get_matrix()
        mat_b = pose_b.get_matrix()
        curobo_pose_error(mat_a, mat_b)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(n_iters):
        pose_a = _random_pose(batch, device)
        pose_b = _random_pose(batch, device)
        # Simulate rollout: convert Pose to matrix
        mat_a = pose_a.get_matrix()
        mat_b = pose_b.get_matrix()
        # Simulate kinematic_costs: convert matrix back to Pose + distance
        curobo_pose_error(mat_a, mat_b)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / n_iters


def benchmark_keep_fk_pose(batch, n_iters=500, n_warmup=50):
    """Proposed: keep FK Pose, only convert desired side from matrix."""
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

        t_current = benchmark_current(batch)
        t_keep_fk = benchmark_keep_fk_pose(batch)
        t_both = benchmark_both_poses(batch)

        print(f"  Current (matrix round-trip):  {t_current:.3f} ms/iter")
        print(f"  Keep FK Pose (1 from_matrix): {t_keep_fk:.3f} ms/iter")
        print(f"  Both Poses (no conversion):   {t_both:.3f} ms/iter")
        print(f"  Speedup (keep FK):  {t_current / t_keep_fk:.2f}x")
        print(f"  Speedup (both):     {t_current / t_both:.2f}x")
