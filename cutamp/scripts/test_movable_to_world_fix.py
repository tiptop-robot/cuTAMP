"""
Test script to verify the movable-to-world initial collision fix.

Creates an environment where blocks slightly penetrate the floor
(simulating perception noise), causing false initial collision costs.
With the old code (no masking), movable_to_world collision penalizes
all particles uniformly at pose_ts=0, making it much harder to find
a feasible plan. With the fix, initial timesteps are masked per-object.

Usage:
    pixi run python -m cutamp.scripts.test_movable_to_world_fix
"""

import logging
import os
import sys

from cutamp.algorithm import run_cutamp
from cutamp.config import TAMPConfiguration
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import load_env, get_env_dir
from cutamp.scripts.utils import default_constraint_to_mult, default_constraint_to_tol, setup_logging

_log = logging.getLogger(__name__)


def run_test(use_fix: bool, max_duration: float = 20.0, num_particles: int = 512) -> bool:
    """Run planning and return whether a plan was found."""
    env_path = os.path.join(get_env_dir(), "blocks_activation_dist_test.yml")
    env = load_env(env_path)

    config = TAMPConfiguration(
        num_particles=num_particles,
        robot="fr3_robotiq",
        num_opt_steps=500,
        max_loop_dur=max_duration,
        enable_visualizer=False,
        rr_spawn=False,
        enable_experiment_logging=False,
        # Default activation distance — initial collision is from blocks penetrating the floor
        # (simulating perception noise), not from the activation distance buffer
        world_activation_distance=0.0,
    )

    cost_reducer = CostReducer(default_constraint_to_mult.copy())
    constraint_checker = ConstraintChecker(default_constraint_to_tol.copy())

    # Monkey-patch to toggle the fix on/off
    if not use_fix:
        import torch
        from cutamp.cost_function import CostFunction

        _original_collision_costs = CostFunction.collision_costs

        def collision_costs_old(self, rollout, obj_to_spheres):
            """Original movable_to_world without per-object masking."""
            robot_spheres = rollout["robot_spheres"]
            coll_values = {"robot_to_world": self.world.collision_fn(robot_spheres)}

            # OLD CODE: concatenate all activated objects, no masking
            activated_movable_spheres = torch.cat(
                [obj_to_spheres[obj] for obj in self.activated_obj], dim=2
            )
            coll_values["movable_to_world"] = self.world.collision_fn(activated_movable_spheres)

            # Rest is identical — call original and overwrite the collision values we changed
            from cutamp.costs import sphere_to_sphere_overlap

            all_movable_spheres = torch.cat(list(obj_to_spheres.values()), dim=2)
            all_pose_ts = list(rollout["ts_to_pose_ts"].values())
            coll_values["robot_to_movables"] = sphere_to_sphere_overlap(
                robot_spheres,
                all_movable_spheres[:, all_pose_ts],
                activation_distance=self.config.gripper_activation_distance,
            )

            if self.movable_obj_pairs:
                obj_1_spheres_list = [obj_to_spheres[name1] for name1, _ in self.movable_obj_pairs]
                obj_2_spheres_list = [obj_to_spheres[name2] for _, name2 in self.movable_obj_pairs]
                obj_1_spheres_batched = torch.stack(obj_1_spheres_list, dim=0)
                obj_2_spheres_batched = torch.stack(obj_2_spheres_list, dim=0)

                collision_results = sphere_to_sphere_overlap(
                    obj_1_spheres_batched,
                    obj_2_spheres_batched,
                    activation_distance=self.config.movable_activation_distance,
                    use_aabb_check=True,
                )

                for idx, pair in enumerate(self.movable_obj_pairs):
                    pair_cost = collision_results[idx]
                    pose_ts = self.pair_to_first_pose_ts[pair]
                    pair_cost_filtered = pair_cost[:, pose_ts:]
                    name1, name2 = pair
                    coll_values[f"{name1}_to_{name2}"] = pair_cost_filtered

            coll_cost = {
                "type": "constraint",
                "constraints": self.cfree_constraints,
                "values": coll_values,
            }
            return coll_cost

        CostFunction.collision_costs = collision_costs_old

    _, num_satisfying, failure_reason = run_cutamp(env, config, cost_reducer, constraint_checker)

    # Restore original if we patched
    if not use_fix:
        CostFunction.collision_costs = _original_collision_costs

    _log.info(f"num_satisfying={num_satisfying}, failure_reason={failure_reason}")
    # failure_reason is None when the planner found satisfying particles (even without curobo_plan)
    return failure_reason is None


def main():
    setup_logging()

    _log.info("=" * 60)
    _log.info("Testing WITHOUT fix (old code - no masking)")
    _log.info("=" * 60)
    old_result = run_test(use_fix=False)
    _log.info(f"Old code result: {'PASS (plan found)' if old_result else 'FAIL (no plan)'}")

    _log.info("")
    _log.info("=" * 60)
    _log.info("Testing WITH fix (new code - per-object masking)")
    _log.info("=" * 60)
    new_result = run_test(use_fix=True)
    _log.info(f"New code result: {'PASS (plan found)' if new_result else 'FAIL (no plan)'}")

    _log.info("")
    _log.info("=" * 60)
    _log.info("SUMMARY")
    _log.info("=" * 60)
    _log.info(f"  Without fix: {'PASS' if old_result else 'FAIL'}")
    _log.info(f"  With fix:    {'PASS' if new_result else 'FAIL'}")

    if not old_result and new_result:
        _log.info("  ✓ Fix correctly resolves initial collision edge case")
        return 0
    elif old_result and new_result:
        _log.warning("  Both passed — activation distance may not be large enough to trigger the bug")
        return 1
    elif not old_result and not new_result:
        _log.warning("  Both failed — problem may be too hard or time limit too short")
        return 1
    else:
        _log.error("  Old passed but new failed — regression!")
        return 2


if __name__ == "__main__":
    sys.exit(main())
