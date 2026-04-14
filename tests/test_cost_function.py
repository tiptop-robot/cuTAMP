"""Tests for cutamp.cost_function."""

import os

import pytest
import torch
from curobo.types.math import Pose

from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import get_env_dir, load_env
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _run_blocks_activation_test(mask: bool) -> int:
    """Run planning on the blocks_activation_dist_test env and return num satisfying particles."""
    env = load_env(os.path.join(get_env_dir(), "blocks_activation_dist_test.yml"))
    config = TAMPConfiguration(
        num_particles=512,
        robot="fr3_robotiq",
        num_opt_steps=500,
        max_loop_dur=20.0,
        enable_visualizer=False,
        rr_spawn=False,
        enable_experiment_logging=False,
        world_activation_distance=0.0,
        mask_initial_movable_world_collision=mask,
    )
    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())
    _, num_satisfying, _ = run_cutamp(env, config, cost_reducer, constraint_checker)
    return num_satisfying


@gpu
def test_kinematic_costs_pose_parity():
    """The FK Pose path (build Pose from stored position+quaternion) must match the old
    matrix round-trip path (Pose.from_matrix(pose.get_matrix())) for pos/rot distance."""
    torch.manual_seed(0)
    device = "cuda"
    b, t = 8, 5

    ee_position = torch.randn(b, t, 3, device=device)
    raw_quat = torch.randn(b, t, 4, device=device)
    ee_quaternion = raw_quat / raw_quat.norm(dim=-1, keepdim=True)

    # Build desired poses as proper 4x4 matrices from random position+quaternion
    desired_pos = torch.randn(b, t, 3, device=device)
    desired_quat_raw = torch.randn(b, t, 4, device=device)
    desired_quat = desired_quat_raw / desired_quat_raw.norm(dim=-1, keepdim=True)
    world_from_ee_desired = (
        Pose(position=desired_pos.view(-1, 3), quaternion=desired_quat.view(-1, 4)).get_matrix().view(b, t, 4, 4)
    )

    # New path: Pose built directly from stored position + quaternion
    new_pose = Pose(
        position=ee_position.view(-1, 3),
        quaternion=ee_quaternion.view(-1, 4),
        normalize_rotation=False,
    )
    desired_pose = Pose.from_matrix(world_from_ee_desired.view(-1, 4, 4))
    new_pos, new_rot = new_pose.distance(desired_pose)

    # Old path: pose -> matrix -> Pose.from_matrix round-trip for the FK side too
    ee_matrix = new_pose.get_matrix()
    old_pose = Pose.from_matrix(ee_matrix)
    old_pos, old_rot = old_pose.distance(desired_pose)

    assert torch.allclose(new_pos, old_pos, atol=1e-5), f"pos_err mismatch: max diff {(new_pos - old_pos).abs().max()}"
    assert torch.allclose(new_rot, old_rot, atol=1e-5), f"rot_err mismatch: max diff {(new_rot - old_rot).abs().max()}"


@gpu
def test_movable_to_world_masking():
    """Masking initial movable-to-world collisions should find satisfying particles when blocks
    slightly penetrate the floor (simulating perception noise), while disabling masking should not."""
    assert _run_blocks_activation_test(mask=False) == 0
    assert _run_blocks_activation_test(mask=True) > 0
