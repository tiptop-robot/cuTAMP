# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import itertools
import logging
import warnings
from collections import defaultdict
from typing import Dict, Union

import roma
import torch
from einops import rearrange
from jaxtyping import Float

from curobo.rollout.cost.self_collision_cost import SelfCollisionCost, SelfCollisionCostConfig
from cutamp.config import TAMPConfiguration
from cutamp.costs import curobo_pose_error, dist_from_bounds_jit, sphere_to_sphere_overlap, trajectory_length
from cutamp.rollout import Rollout
from cutamp.tamp_world import TAMPWorld
from cutamp.task_planning import PlanSkeleton
from cutamp.task_planning.constraints import (
    Collision,
    CollisionFree,
    CollisionFreeGrasp,
    CollisionFreeHolding,
    CollisionFreePlacement,
    KinematicConstraint,
    Motion,
    StablePlacement,
    ValidPush,
    ValidPushStick,
)
from cutamp.task_planning.costs import GraspCost, TrajectoryLength
from cutamp.utils.common import transform_spheres
from cutamp.utils.obb import get_object_obb

_log = logging.getLogger(__name__)


class CostFunction:
    """
    Cost Function for a given plan skeleton. Given a rollout, we compute the constraints and costs.
    The __init__ function caches constraints and costs for quicker indexing during rollout evaluation.
    """

    def __init__(self, plan_skeleton: PlanSkeleton, world: TAMPWorld, config: TAMPConfiguration):
        if config.enable_traj:
            raise NotImplementedError("Trajectories not supported in cost function yet")

        self.plan_skeleton = plan_skeleton
        self.world = world
        self.config = config
        self._rollout_validated = False

        # Accumulate the constraints, so we can batch them up when computing the costs
        self.cfree_constraints = []
        self.kinematic_constraints = []
        self.motion_constraints = []
        self.stable_placement_constraints = []
        self.valid_push_constraints = []
        self.valid_push_stick_constraints = []
        self.traj_length_costs = []

        type_to_list = {
            KinematicConstraint.type: self.kinematic_constraints,
            Motion.type: self.motion_constraints,
            CollisionFree.type: self.cfree_constraints,
            CollisionFreeHolding.type: self.cfree_constraints,
            CollisionFreeGrasp.type: self.cfree_constraints,
            CollisionFreePlacement.type: self.cfree_constraints,
            StablePlacement.type: self.stable_placement_constraints,
            ValidPush.type: self.valid_push_constraints,
            ValidPushStick.type: self.valid_push_stick_constraints,
            TrajectoryLength.type: self.traj_length_costs,
        }
        warning_co = {GraspCost.type}
        for ground_op in plan_skeleton:
            for co in [*ground_op.constraints, *ground_op.costs]:
                if co.type not in type_to_list:
                    if co.type in warning_co:
                        warnings.warn(f"Cost {co} is not handled in the cost function")
                    else:
                        raise NotImplementedError(f"Unhandled constraint or cost: {co}")
                else:
                    type_to_list[co.type].append(co)

        # All conf parameters for motion constraints, we check rollout is subset
        self.motion_conf_params = set(
            iter(itertools.chain.from_iterable(con.params for con in self.motion_constraints))
        )

        # Setup self-collision cost
        self_collision_config = SelfCollisionCostConfig(
            self.world.tensor_args.to_device([1.0]),
            self.world.tensor_args,
            return_loss=True,
            self_collision_kin_config=self.world.kin_model.get_self_collision_config(),
        )
        self.self_collision_cost_fn = SelfCollisionCost(self_collision_config)
        # Should be using experimental kernel by default
        if not self.self_collision_cost_fn.self_collision_kin_config.experimental_kernel:
            raise ValueError("Expected self-collision cost to use experimental kernel")

        # Conf parameters for kinematic constraints, order in rollout should match
        self.kinematic_confs, self.kinematic_actions = zip(*(con.params for con in self.kinematic_constraints))
        self.kinematic_confs = list(self.kinematic_confs)
        self.kinematic_actions = list(self.kinematic_actions)

        # Compute the AABB and surface z-position for the placement surfaces
        self.surface_to_aabb = {}
        self.surface_to_obb = {}
        self.surface_to_target_z = {}
        self.surface_to_objs = defaultdict(list)
        for con in self.stable_placement_constraints:
            obj, _, _, surface = con.params
            self.surface_to_objs[surface].append(obj)
            if surface in self.surface_to_aabb:
                continue

            if self.config.placement_check == "aabb":
                aabb = world.get_aabb(surface)  # includes xyz
                aabb_xy = aabb[:, :2]
                self.surface_to_aabb[surface] = aabb_xy
                surface_z = aabb[1, 2]
            else:
                assert self.config.placement_check == "obb"
                surface_obj = world.get_object(surface)
                obb = get_object_obb(surface_obj, shrink_dist=config.placement_shrink_dist)
                self.surface_to_obb[surface] = obb
                surface_z = obb.surface_z

            # For target z, need to take collision activation distance into account
            target_z = surface_z + world.collision_activation_distance + 2e-3  # add some buffer
            self.surface_to_target_z[surface] = target_z

        # Store the button AABBs for ValidPush
        self.button_to_action = {}
        self.button_aabbs = []
        for con in self.valid_push_constraints:
            button, action = con.params
            if not self.world.has_object(button):
                raise ValueError(f"{button=} not found in world")
            if button in self.button_to_action:
                raise NotImplementedError(f"We only support pushing a button once right now")

            # Set z to be 2cm buffer above surface of the button
            aabb = self.world.get_aabb(button).clone()
            aabb[0, 2] = aabb[1, 2] + world.collision_activation_distance + 2e-3
            aabb[1, 2] = aabb[1, 2] + world.collision_activation_distance + 0.02
            self.button_aabbs.append(aabb)
            self.button_to_action[button] = action
        self.button_aabbs = torch.stack(self.button_aabbs) if self.button_aabbs else None

        # Store the buttons and sticks for ValidPushStick
        self.button_stick_actions = {}
        self.button_stick_aabbs = []
        self.button_to_stick = {}
        for con in self.valid_push_stick_constraints:
            button, stick, action = con.params
            if not self.world.has_object(button):
                raise ValueError(f"{button=} not found in world")
            if not self.world.has_object(stick):
                raise ValueError(f"{stick=} not found in world")
            if button in self.button_to_stick:
                raise NotImplementedError(f"We only support pushing a button once right now")

            # Set z to be 2cm buffer above surface of the button
            button_aabb = self.world.get_aabb(button).clone()
            button_aabb[0, 2] = button_aabb[1, 2] + world.collision_activation_distance + 2e-3
            button_aabb[1, 2] = button_aabb[1, 2] + world.collision_activation_distance + 0.02
            self.button_stick_aabbs.append(button_aabb)

            self.button_to_stick[button] = stick
            self.button_stick_actions[button] = action
        self.button_stick_aabbs = torch.stack(self.button_stick_aabbs) if self.button_stick_aabbs else None

        # All conf parameters for trajectory length costs, we check rollout matches
        # No support for trajectories right now
        self.traj_length_confs = []
        for cost in self.traj_length_costs:
            q_start, traj, q_end = cost.params
            self.traj_length_confs.append(q_start)
            self.traj_length_confs.append(q_end)
        self.traj_length_confs = list(dict.fromkeys(self.traj_length_confs))  # remove duplicates
        if self.traj_length_confs[0] != "q0":
            raise ValueError("Expected q0 to be the first conf")
        self.traj_length_confs = self.traj_length_confs[1:]

        # Identify which objects are manipulated in the plan
        self.activated_obj = set()
        self.obj_to_first_place = {}
        for ground_op in plan_skeleton:
            op_name = ground_op.operator.name
            if op_name == "Pick":
                obj = ground_op.values[0]
                self.activated_obj.add(obj)
            elif op_name == "Place":
                obj = ground_op.values[0]
                pose = ground_op.values[2]
                self.activated_obj.add(obj)
                if obj not in self.obj_to_first_place:
                    self.obj_to_first_place[obj] = pose
            elif op_name == "PushStick" or op_name == "Push":
                raise NotImplementedError(f"Haven't handled {op_name}")
            else:
                assert op_name == "MoveFree" or op_name == "MoveHolding"

        # Pre-compute movable object pairs for collision checking.
        # Only include pairs where at least one object is manipulated
        movable_names = [m.name for m in self.world.movables]
        all_movable_pairs = list(itertools.combinations(movable_names, 2)) if len(movable_names) > 1 else []
        self.movable_obj_pairs = [
            (obj_1, obj_2)
            for obj_1, obj_2 in all_movable_pairs
            if obj_1 in self.activated_obj or obj_2 in self.activated_obj
        ]
        pairs_msg = ", ".join(f"({o_1}, {o_2})" for o_1, o_2 in self.movable_obj_pairs)
        _log.debug(f"Checking movable collisions between {pairs_msg}")

        # Populated upon validating the rollout
        self.obj_to_first_pose_ts = {}
        self.pair_to_first_pose_ts = {}
        self._activated_objs = sorted(self.activated_obj)  # deterministic ordering for torch.stack
        self._movable_world_mask = None  # lazily built in collision_costs
        self._all_pose_ts = None

    def _validate_rollout(self, rollout: Rollout):
        """Checks structure of the rollout conforms to the assumptions we make in the cost function implementation."""
        if self._rollout_validated:
            return

        # Configurations should be subset of parameters involved in motion constraints
        if not set(rollout["conf_params"]).issubset(self.motion_conf_params):
            raise RuntimeError(
                f"Missing conf params in motion constraints: {rollout['conf_params'] - self.motion_conf_params}"
            )

        # Kinematic configuration and actions (i.e., poses) should match
        if self.kinematic_confs != rollout["conf_params"]:
            raise RuntimeError(f"Expected conf params {self.kinematic_confs} but got {rollout['conf_params']}")
        if self.kinematic_actions != rollout["action_params"]:
            raise RuntimeError(f"Expected action params {self.kinematic_actions} but got {rollout['action_params']}")

        # Trajectory length parameters should match
        if self.traj_length_confs != rollout["conf_params"]:
            raise RuntimeError(f"Expected conf params {self.traj_length_confs} but got {rollout['conf_params']}")

        # Bad-ish hack to get the first point at which the object is activated
        for obj, action in self.obj_to_first_place.items():
            self.obj_to_first_pose_ts[obj] = rollout["action_to_pose_ts"][action]
        for pair in self.movable_obj_pairs:
            obj_1, obj_2 = pair
            if obj_1 in self.obj_to_first_pose_ts and obj_2 in self.obj_to_first_pose_ts:
                self.pair_to_first_pose_ts[pair] = min(
                    self.obj_to_first_pose_ts[obj_1], self.obj_to_first_pose_ts[obj_2]
                )
            elif obj_1 in self.obj_to_first_pose_ts:
                self.pair_to_first_pose_ts[pair] = self.obj_to_first_pose_ts[obj_1]
            else:
                assert obj_2 in self.obj_to_first_place
                self.pair_to_first_pose_ts[pair] = self.obj_to_first_pose_ts[obj_2]

        self._all_pose_ts = list(rollout["ts_to_pose_ts"].values())

        # Mask for movable-to-world collisions: zero out timesteps before each object's first
        # placement (objects may initially be in collision with surfaces they rest on, e.g. due
        # to perception noise). Fixed per skeleton, so build once here.
        if self.config.mask_initial_movable_world_collision and self._activated_objs:
            t = len(self._all_pose_ts)
            mask = torch.ones(len(self._activated_objs), 1, t, device=self.world.device)
            for i, obj in enumerate(self._activated_objs):
                first_ts = self.obj_to_first_pose_ts[obj]
                if first_ts > 0:
                    mask[i, :, :first_ts] = 0.0
            self._movable_world_mask = mask

        self._rollout_validated = True

    def kinematic_costs(self, rollout: Rollout) -> Union[dict, None]:
        """Kinematic constraints - i.e., pose error between actual and desired end-effector poses."""
        pos_errs, rot_errs = curobo_pose_error(rollout["world_from_ee"], rollout["world_from_ee_desired"])
        kinematic_cost = {
            "type": "constraint",
            "constraints": self.kinematic_constraints,
            "values": {"pos_err": pos_errs, "rot_err": rot_errs},
        }
        return kinematic_cost

    def motion_costs(self, rollout: Rollout) -> Union[dict, None]:
        """Motion constraints - valid motions don't exceed joint limits or self-collide."""
        # Joint limits
        confs = rollout["confs"]
        dist_from_joint_lims = dist_from_bounds_jit(
            confs, self.world.robot_container.joint_limits[0], self.world.robot_container.joint_limits[1]
        )

        # Self collisions
        robot_spheres = rollout["robot_spheres"]
        with torch.profiler.record_function("coll::self_collision"):
            self_coll_vals = self.self_collision_cost_fn(robot_spheres)

        motion_cost = {
            "type": "constraint",
            "constraints": self.motion_constraints,
            "values": {"joint_limit": dist_from_joint_lims, "self_collision": self_coll_vals},
        }
        return motion_cost

    def valid_push_costs(
        self, rollout: Rollout, obj_to_spheres: Dict[str, Float[torch.Tensor, "b t n 4"]]
    ) -> Union[dict, None]:
        """Valid Push constraints - i.e., distance from the button for push actions."""
        if not (self.valid_push_constraints or self.valid_push_stick_constraints):
            return None

        valid_push_cost = {"type": "constraint", "constraints": [], "values": {}}

        if self.valid_push_constraints:
            ts_idxs = []  # get timestep for the button push actions
            for button, action in self.button_to_action.items():
                ts = rollout["action_to_ts"][action]
                ts_idxs.append(ts)
            tool_poses = rollout["world_from_tool_desired"][:, ts_idxs]
            tool_xyz = tool_poses[:, :, :3, 3]
            dist_from_button = dist_from_bounds_jit(tool_xyz, self.button_aabbs[:, 0], self.button_aabbs[:, 1])
            valid_push_cost["constraints"].extend(self.valid_push_constraints)
            valid_push_cost["values"]["dist_from_button"] = dist_from_button

        if self.valid_push_stick_constraints:
            # Get the stick spheres for each button push action
            all_stick_spheres = []
            for button, action in self.button_stick_actions.items():
                pose_ts = rollout["action_to_pose_ts"][action]
                stick_name = self.button_to_stick[button]
                stick_spheres = obj_to_spheres[stick_name][:, pose_ts]
                all_stick_spheres.append(stick_spheres)
            all_stick_spheres = torch.stack(all_stick_spheres, dim=1)

            # We'll consider bottom of spheres, so subtract the radius
            stick_xyz = all_stick_spheres[..., :3].clone()  # important! clone so we don't modify original tensor
            stick_xyz[..., 2] -= all_stick_spheres[..., 3]

            # Compute distance between all stick spheres and button AABB, take the minimum
            push_stick_cost = dist_from_bounds_jit(
                stick_xyz, self.button_stick_aabbs[:, None, 0], self.button_stick_aabbs[:, None, 1]
            )
            push_stick_cost = push_stick_cost.min(-1).values

            valid_push_cost["constraints"].extend(self.valid_push_stick_constraints)
            valid_push_cost["values"]["stick_dist_from_button"] = push_stick_cost

        return valid_push_cost

    def stable_placement_costs(
        self, rollout: Rollout, obj_to_spheres: Dict[str, Float[torch.Tensor, "b t n 4"]]
    ) -> Union[dict, None]:
        """
        Stable Placement constraints. Converted to two costs:
            1. Object sphere xy positions within surface AABB
            2. Minimum object sphere is supported by the surface (compute distance between surface and bottom of sphere)
        """
        if not self.stable_placement_constraints:
            return None

        # First collate the objects by placement surface.
        surface_to_obj = defaultdict(list)
        surface_to_spheres = defaultdict(list)
        for con in self.stable_placement_constraints:
            obj, _, placement, surface = con.params
            pose_ts = rollout["action_to_pose_ts"][placement]
            obj_spheres = obj_to_spheres[obj][:, pose_ts]
            surface_to_obj[surface].append(obj)
            surface_to_spheres[surface].append(obj_spheres)

        num_particles = rollout["num_particles"]
        support_vals = {}
        for surface, objs in surface_to_obj.items():
            # Create map of sphere index to object index
            sphere_idx_map = []
            for o_idx, (obj, spheres) in enumerate(zip(objs, surface_to_spheres[surface])):
                num_spheres = spheres.shape[1]
                sph_idxs = [o_idx] * num_spheres
                sphere_idx_map.extend(sph_idxs)
            sphere_idx_map = torch.tensor(sphere_idx_map, dtype=torch.int64, device=self.world.device)
            sphere_idx_map_expand = sphere_idx_map[None].expand(num_particles, -1)  # expand by batch size

            # Since objects can have different numbers of spheres, we need to concatenate instead of stack
            spheres = torch.cat(surface_to_spheres[surface], dim=1)
            spheres_xy = spheres[..., :2]

            # Within goal xy bounds, need to gather by the spheres for each object
            if self.config.placement_check == "aabb":
                in_goal_xy = dist_from_bounds_jit(spheres_xy, *self.surface_to_aabb[surface])
                obj_in_goal_xy = torch.zeros(
                    (num_particles, len(objs)), dtype=in_goal_xy.dtype, device=in_goal_xy.device
                )
                obj_in_goal_xy.scatter_add_(1, sphere_idx_map_expand, in_goal_xy)
                support_vals[f"{surface}_in_xy"] = obj_in_goal_xy
            else:
                # Check spheres are within the OBB's xy plane by transforming to OBB local frame
                obb = self.surface_to_obb[surface]

                # Transform sphere centers to OBB local frame using cached rotation matrix
                sphere_centers = spheres[..., :3]  # (b, n, 3)
                centers_relative = sphere_centers - obb.center  # translate to OBB origin
                centers_local = centers_relative @ obb.rot_matrix_inv.T  # rotate to OBB frame

                # Now everything's in local frame just use AABB check
                centers_xy = centers_local[..., :2]  # (b, n, 2)
                # com_xy = centers_xy.mean(1)[:, None]
                obb_xy_lower = -obb.half_extents[:2]
                obb_xy_upper = obb.half_extents[:2]
                in_goal_xy = dist_from_bounds_jit(centers_xy, obb_xy_lower, obb_xy_upper)

                # Accumulate per-object distances
                obj_in_goal_xy = torch.zeros(
                    (num_particles, len(objs)), dtype=in_goal_xy.dtype, device=in_goal_xy.device
                )
                obj_in_goal_xy.scatter_add_(1, sphere_idx_map_expand, in_goal_xy)
                support_vals[f"{surface}_in_xy"] = obj_in_goal_xy

            # Distance between bottom of spheres and z-position of the surface
            spheres_bottom = spheres[..., 2] - spheres[..., 3]
            obj_bottom = torch.full(
                (num_particles, len(objs)), float("inf"), dtype=spheres_bottom.dtype, device=spheres_bottom.device
            )
            obj_bottom.scatter_reduce_(1, sphere_idx_map_expand, spheres_bottom, reduce="amin")
            target_z = self.surface_to_target_z[surface]
            support_vals[f"{surface}_support"] = torch.abs(obj_bottom - target_z)

        stable_placement_cost = {
            "type": "constraint",
            "constraints": self.stable_placement_constraints,
            "values": support_vals,
        }
        return stable_placement_cost

    def trajectory_costs(self, rollout: Rollout) -> dict:
        """Trajectory costs, just joint space distance between configurations for now."""
        traj_cost = {
            "type": "cost",
            "costs": self.traj_length_costs,
            "values": {"traj_length": trajectory_length(rollout["confs"])},
        }
        return traj_cost

    def collision_costs(self, rollout: Rollout, obj_to_spheres: Dict[str, Float[torch.Tensor, "b t n 4"]]) -> dict:
        """Collision costs."""
        # Robot to world
        robot_spheres = rollout["robot_spheres"]
        with torch.profiler.record_function("coll::robot_to_world"):
            coll_values = {"robot_to_world": self.world.collision_fn(robot_spheres)}

        # Collision between movables and world — batch all activated objects in one collision_fn call,
        # then mask out timesteps before each object's first placement. The motion solver handles this
        # analogously by temporarily detaching the object from the robot when the grasped object's
        # spheres cause an invalid start state during retract planning.
        with torch.profiler.record_function("coll::movable_to_world"):
            stacked = torch.stack([obj_to_spheres[obj] for obj in self._activated_objs])
            coll = self.world.collision_fn(rearrange(stacked, "objs b t n d -> (objs b) t n d"))
            coll = rearrange(coll, "(objs b) t -> objs b t", objs=len(self._activated_objs))
            if self._movable_world_mask is not None:
                coll = coll * self._movable_world_mask
            coll_values["movable_to_world"] = coll.sum(dim=0)

        with torch.profiler.record_function("coll::robot_to_movables"):
            act_dist = self.config.gripper_activation_distance
            # Per-object Warp kernel launches outperformed a single concatenated call in profiling:
            # smaller n2 per launch improves the kernel's early-exit behavior (most robot-sphere ↔
            # distant-object-sphere pairs are rejected on the threshold check before sqrt), and
            # launch overhead is negligible for the handful of movables we have. Revisit if we
            # scale to many more movables.
            coll_values["robot_to_movables"] = sum(
                sphere_to_sphere_overlap(robot_spheres, obj_s[:, self._all_pose_ts], activation_distance=act_dist)
                for obj_s in obj_to_spheres.values()
            )

        # Collision between movable objects
        if self.movable_obj_pairs:
            with torch.profiler.record_function("coll::movable_to_movable"):
                # Stack into (num_pairs, b, t, n_spheres, 4)
                obj_1_spheres_list = [obj_to_spheres[name1] for name1, _ in self.movable_obj_pairs]
                obj_2_spheres_list = [obj_to_spheres[name2] for _, name2 in self.movable_obj_pairs]
                obj_1_spheres_batched = torch.stack(obj_1_spheres_list, dim=0)
                obj_2_spheres_batched = torch.stack(obj_2_spheres_list, dim=0)

                collision_results = sphere_to_sphere_overlap(
                    obj_1_spheres_batched,
                    obj_2_spheres_batched,
                    activation_distance=self.config.movable_activation_distance,
                    use_aabb_check=True,
                )  # (num_pairs, b, t)

            for idx, pair in enumerate(self.movable_obj_pairs):
                pair_cost = collision_results[idx]
                pose_ts = self.pair_to_first_pose_ts[pair]
                # Only consider costs from when the action was activated. This allows us to handle objects that
                # are initially in-collision (perhaps due to bad perception)
                pair_cost_filtered = pair_cost[:, pose_ts:]
                name1, name2 = pair
                coll_values[f"{name1}_to_{name2}"] = pair_cost_filtered

        coll_cost = {
            "type": "constraint",
            "constraints": self.cfree_constraints,
            "values": coll_values,
        }
        return coll_cost

    def soft_costs(self, rollout: Rollout) -> dict:
        """Soft costs defined on the goal state."""
        # last object pose
        last_obj_position = [v[:, -1, :3, 3] for v in rollout["obj_to_pose"].values()]
        last_obj_position = torch.stack(last_obj_position, dim=1)

        if self.config.soft_cost == "dist_from_origin":
            dist_from_origin = last_obj_position.norm(dim=-1)
            dist_from_origin = -dist_from_origin.sum(dim=-1)
            values = {"dist_from_origin": dist_from_origin}
        elif self.config.soft_cost == "max_obj_dist" or self.config.soft_cost == "min_obj_dist":
            all_obj_dists = torch.cdist(last_obj_position, last_obj_position, p=2)  # (b, n, n)
            mask = torch.triu(torch.ones_like(all_obj_dists), diagonal=1) == 1
            obj_dists = all_obj_dists[mask].view(mask.shape[0], -1)  # reshape into num pairs
            dists_sum = obj_dists.sum(-1)
            if self.config.soft_cost == "max_obj_dist":
                values = {"max_obj_dist": -dists_sum}
            else:
                values = {"min_obj_dist": dists_sum}
        elif self.config.soft_cost == "min_y" or self.config.soft_cost == "max_y":
            last_obj_y = last_obj_position[..., 1]
            last_y = last_obj_y.sum(dim=-1)
            if self.config.soft_cost == "min_y":
                values = {"min_y": last_y}
            else:
                values = {"max_y": -last_y}
        elif self.config.soft_cost == "align_yaw":
            last_obj_mat3x3 = [v[:, -1, :3, :3] for v in rollout["obj_to_pose"].values()]
            last_obj_mat3x3 = torch.stack(last_obj_mat3x3, dim=1)
            last_obj_rpy = roma.rotmat_to_euler("XYZ", last_obj_mat3x3)
            last_obj_yaw = last_obj_rpy[..., 2]

            # Compute pairwise yaw differences and normalize to be between -pi and pi
            yaw_diffs = last_obj_yaw[:, :, None] - last_obj_yaw[:, None, :]
            yaw_diffs = torch.atan2(torch.sin(yaw_diffs), torch.cos(yaw_diffs)).abs()
            mask = torch.triu(torch.ones_like(yaw_diffs), diagonal=1) == 1
            yaw_diffs = yaw_diffs[mask].view(mask.shape[0], -1)  # reshape into num pairs
            yaw_diffs = yaw_diffs.sum(-1)
            values = {"align_yaw": yaw_diffs}
        else:
            raise ValueError(f"Unsupported soft cost: {self.config.soft_cost}")

        return {"type": "cost", "constraints": [], "values": values}

    def __call__(self, rollout: Rollout) -> Dict[str, dict]:
        self._validate_rollout(rollout)
        cost_dict = {}

        def add_cost(k_, v_):
            if v_ is not None:
                cost_dict[k_] = v_

        # Trajectory cost
        with torch.profiler.record_function("cost::trajectory"):
            traj_cost = self.trajectory_costs(rollout)
        add_cost(TrajectoryLength.type, traj_cost)

        # Get collision spheres for movable objects
        with torch.profiler.record_function("cost::transform_spheres"):
            obj_to_spheres = {}
            for idx, obj in enumerate(self.world.movables):
                if obj.name in obj_to_spheres:
                    raise RuntimeError(f"Object {obj.name} already in obj_to_spheres")
                obj_pose = rollout["obj_to_pose"][obj.name]
                obj_spheres = transform_spheres(self.world.get_collision_spheres(obj), obj_pose)
                obj_to_spheres[obj.name] = obj_spheres

        # Collision costs
        with torch.profiler.record_function("cost::collision"):
            collision_cost = self.collision_costs(rollout, obj_to_spheres)
        add_cost(Collision.type, collision_cost)

        # Valid Push constraints
        with torch.profiler.record_function("cost::valid_push"):
            valid_push_cost = self.valid_push_costs(rollout, obj_to_spheres)
        add_cost(ValidPush.type, valid_push_cost)

        # Stable placement cost
        with torch.profiler.record_function("cost::stable_placement"):
            stable_placement_cost = self.stable_placement_costs(rollout, obj_to_spheres)
        add_cost(StablePlacement.type, stable_placement_cost)

        # Valid motions don't exceed joint limits
        with torch.profiler.record_function("cost::motion"):
            motion_cost = self.motion_costs(rollout)
        add_cost(Motion.type, motion_cost)

        # Kinematic costs
        with torch.profiler.record_function("cost::kinematic"):
            kinematic_cost = self.kinematic_costs(rollout)
        add_cost(KinematicConstraint.type, kinematic_cost)

        # Soft costs
        if self.config.soft_cost is not None:
            with torch.profiler.record_function("cost::soft"):
                soft_cost = self.soft_costs(rollout)
            add_cost("soft", soft_cost)

        return cost_dict
