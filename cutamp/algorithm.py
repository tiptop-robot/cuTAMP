# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Core cuTAMP algorithm implementation."""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Union, Optional, Tuple
from unittest.mock import Mock

import torch

from curobo.types.base import TensorDeviceType
from curobo.wrap.reacher.ik_solver import IKSolver
from curobo.wrap.reacher.motion_gen import MotionGen
from cutamp.config import TAMPConfiguration, validate_tamp_config
from cutamp.constraint_checker import ConstraintChecker
from cutamp.cost_function import CostFunction
from cutamp.cost_reduction import CostReducer
from cutamp.envs.utils import TAMPEnvironment
from cutamp.experiment_logger import ExperimentLogger
from cutamp.motion_solver import solve_curobo, MotionPlanningError
from cutamp.optimize_plan import ParticleOptimizer
from cutamp.particle_initialization import ParticleInitializer
from cutamp.robots import get_q_home, load_robot_container
from cutamp.rollout import RolloutFunction
from cutamp.tamp_domain import all_tamp_operators
from cutamp.tamp_world import TAMPWorld, check_tamp_world_not_in_collision
from cutamp.task_planning import PlanSkeleton, task_plan_generator
from cutamp.utils.common import get_world_cfg
from cutamp.utils.timer import TorchTimer
from cutamp.utils.visualizer import RerunVisualizer, MockVisualizer, Visualizer

_log = logging.getLogger(__name__)


def heuristic_fn(
    plan_skeleton: PlanSkeleton, cost_dict: dict, constraint_checker: ConstraintChecker, verbose: bool = True
) -> float:
    """
    Get a single heuristic value for a cost dict corresponding to a rollout.

    We first compute the success rate of each constraint. If the constraint has zero success, we assign it a penalty
    of -num_particles. We then compute the mean success rate across all constraints, and use the failure rate as the
    heuristic (lower the better).
    """
    full_mask = constraint_checker.get_full_mask(cost_dict)
    successes = []
    num_particles = None
    for con_type, con_info in full_mask.items():
        for name, mask in con_info.items():
            if mask.ndim == 2:
                satisfying = mask.sum(0)
            else:
                satisfying = mask.sum()

            if num_particles is None:
                num_particles = mask.shape[0]
            else:
                assert num_particles == mask.shape[0]

            # replace zeros with -num_particles
            satisfying[satisfying == 0] = -num_particles
            successes.extend(satisfying.tolist())
            if verbose:
                _log.debug(f"{con_type} {name} {satisfying.tolist()}")
    success_mean = sum(successes) / len(successes)
    success_rate = success_mean / num_particles
    failure_rate = 1 - success_rate
    heuristic = 100 * failure_rate

    # We have a preference for shorter plans
    heuristic += len(plan_skeleton)
    return heuristic


def get_best_particle(
    plan_info: dict, config: TAMPConfiguration, constraint_checker: ConstraintChecker, cost_reducer: CostReducer
) -> dict:
    """Get the particle that satisfies the constraints and has the best soft cost."""
    particles, rollout_fn, cost_fn = plan_info["particles"], plan_info["rollout_fn"], plan_info["cost_fn"]
    with torch.no_grad():
        rollout = rollout_fn(particles)
        cost_dict = cost_fn(rollout)

    # Take the best particle that is satisfying and has the best soft cost
    satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
    if not satisfying_mask.any():
        raise RuntimeError("No satisfying particles found")

    soft_costs = cost_reducer.soft_costs(cost_dict)
    satisfying_costs = soft_costs[satisfying_mask]
    best_satisfying_idx = satisfying_costs.argmin()
    indices = torch.arange(config.num_particles, device=satisfying_costs.device)
    best_idx = indices[satisfying_mask][best_satisfying_idx]
    best_particle = {k: v[best_idx].detach().clone() for k, v in particles.items() if v is not None}
    return best_particle


def _visualize_best_particle(
    visualizer: Visualizer,
    rollout: dict,
    best_idx: int,
    world: TAMPWorld,
    config: TAMPConfiguration,
) -> None:
    """Visualize the rollout for the best ranked particle."""
    visualizer.set_time_sequence("rollout_best", 0)
    visualizer.set_joint_positions(world.q_init.tolist())
    for obj in world.movables:
        obj_pose = world.get_object_pose(obj).cpu()
        visualizer.log_mat4x4(f"world/{obj.name}", obj_pose)

    for ts in range(len(rollout["conf_params"])):
        visualizer.set_time_sequence("rollout_best", ts + 1)
        q = rollout["confs"][best_idx, ts]

        gripper_close = rollout["gripper_close"][ts]
        if config.robot == "ur5":
            gripper_joints = [0.4] if gripper_close else [0.0]
        elif config.robot == "panda":
            gripper_joints = [0.01, 0.01] if gripper_close else [0.04, 0.04]
        else:
            gripper_joints = []
        visualizer.set_joint_positions(q.tolist() + gripper_joints)

        world_from_ee = rollout["world_from_ee"][best_idx, ts].cpu()
        visualizer.log_mat4x4("rollout/ee_pose", world_from_ee)

        robot_spheres = rollout["robot_spheres"][best_idx, ts].cpu()
        visualizer.log_spheres("rollout/robot_spheres", robot_spheres)

        pose_ts = rollout["ts_to_pose_ts"][ts]
        for obj in world.movables:
            obj_pose = rollout["obj_to_pose"][obj.name][best_idx, pose_ts].cpu()
            visualizer.log_mat4x4(f"world/{obj.name}", obj_pose)


def get_ranked_satisfying_particles(
    plan_info: dict,
    config: TAMPConfiguration,
    constraint_checker: ConstraintChecker,
    cost_reducer: CostReducer,
    visualizer: Visualizer | None = None,
) -> dict[str, torch.Tensor]:
    """Get the satisfying particles ranked by grasp confidence (if available) or soft costs."""
    particles, rollout_fn, cost_fn = plan_info["particles"], plan_info["rollout_fn"], plan_info["cost_fn"]
    with torch.no_grad():
        rollout = rollout_fn(particles)
        cost_dict = cost_fn(rollout)

    # Get all satisfying particles
    satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
    if not satisfying_mask.any():
        raise RuntimeError("No satisfying particles found")

    soft_costs = cost_reducer.soft_costs(cost_dict)
    satisfying_costs = soft_costs[satisfying_mask]

    # Sum grasp confidences across all grasp parameters
    grasp_keys = [k for k in particles if k.startswith("grasp") and k.endswith("_confidences")]
    grasp_confs = None
    for grasp_key in grasp_keys:
        if (conf := particles[grasp_key]) is not None:
            grasp_confs = conf if grasp_confs is None else (grasp_confs + conf)
    satisfying_grasp_confs = grasp_confs[satisfying_mask] if grasp_confs is not None else None

    # Rank satisfying particles by grasp confidence (if available) or soft costs
    indices = torch.arange(config.num_particles, device=satisfying_costs.device)
    satisfying_idxs = indices[satisfying_mask]
    use_grasp_ranking = satisfying_grasp_confs is not None
    if use_grasp_ranking:
        sorted_idxs = satisfying_grasp_confs.argsort(descending=True)  # Higher confidence is better
    else:
        sorted_idxs = satisfying_costs.argsort()  # Lower cost is better
    ranked_idxs = satisfying_idxs[sorted_idxs]
    ranked_particles = {k: v[ranked_idxs].detach().clone() for k, v in particles.items() if v is not None}

    # Visualize the best particle
    if visualizer is not None:
        best_idx = ranked_idxs[0]
        _visualize_best_particle(visualizer, rollout, best_idx, rollout_fn.world, config)

    return ranked_particles


def sample_plan_skeleton(
    plan_gen,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    plan_count: int,
    constraint_checker: ConstraintChecker,
    cost_reducer: CostReducer,
    particle_initializer: ParticleInitializer,
) -> Tuple[Union[dict, None], bool]:
    """
    Try sampling a plan skeleton (if any remain), then its particles and compute the heuristic.
    Returns the plan_info dict and whether any satisfying particles were found upon initialization.
    """
    with timer.time("sample_task_plan"):
        plan_skeleton = next(plan_gen)
    if not plan_skeleton:
        return None, False

    plan_str = [op.name for op in plan_skeleton]
    _log.debug(f"[Plan {plan_count + 1}] Sampled plan {plan_str}")

    # Sample particles
    with timer.time("initialize_particles"):
        plan_particles = particle_initializer(plan_skeleton)
    if plan_particles is None:  # failed subgraph
        return None, False

    # Rollout particles and compute costs
    rollout_fn = RolloutFunction(plan_skeleton, world, config)
    cost_fn = CostFunction(plan_skeleton, world, config)
    with timer.time("measure_heuristic"), torch.no_grad():
        rollout = rollout_fn(plan_particles)
        cost_dict = cost_fn(rollout)
        heuristic = heuristic_fn(plan_skeleton, cost_dict, constraint_checker)

    # Number of satisfying particles
    with timer.time("get_satisfying_mask"):
        satisfying_mask = constraint_checker.get_mask(cost_dict)
    num_satisfying = satisfying_mask.sum().item()

    if config.stick_button_experiment and num_satisfying > 0:
        # Custom logic in stick button for breaking early for sampling baseline
        heuristic -= 100
        print(f"Found satisfying plan: {plan_str} heuristic -= 100")

    # Best cost initially
    with timer.time("compute_best_cost"):
        consider_types = {"constraint"}
        if config.optimize_soft_costs:
            consider_types.add("cost")
        costs = cost_reducer(cost_dict, consider_types=consider_types)
        if satisfying_mask.any():
            best_cost = costs[satisfying_mask].min().item()
            best_soft_cost = cost_reducer.soft_costs(cost_dict)[satisfying_mask].min().item()
        else:
            best_cost, best_soft_cost = float("inf"), float("inf")

    plan_info = {
        "idx": plan_count,
        "plan_skeleton": plan_skeleton,
        "particles": plan_particles,
        "rollout_fn": rollout_fn,
        "cost_fn": cost_fn,
        "heuristic": heuristic,
        "num_satisfying": num_satisfying,
        "best_cost": best_cost,
        "best_soft_cost": best_soft_cost,
    }

    _log.debug(
        f"[Plan {plan_count + 1}] {plan_info['num_satisfying']}/{config.num_particles} satisfying, "
        f"heuristic = {plan_info['heuristic']}"
    )
    return plan_info, num_satisfying > 0


def resample_plan_info(
    plan_info: dict,
    world: TAMPWorld,
    config: TAMPConfiguration,
    timer: TorchTimer,
    cost_reducer: CostReducer,
    constraint_checker: ConstraintChecker,
    particle_initializer: ParticleInitializer,
) -> int:
    """
    Sample particles again in-place for a plan info container with a plan skeleton. This can be used for rejection
    sampling strategy (for the sampling baseline), or for random restarts.

    Returns number of satisfying particles after re-sampling.
    """
    with timer.time("initialize_particles"), timer.time("resample_particles"):
        plan_particles = particle_initializer(plan_info["plan_skeleton"], verbose=False)

    # Rollout new particles and compute costs
    with timer.time("measure_heuristic"), torch.no_grad():
        rollout = plan_info["rollout_fn"](plan_particles)
        cost_dict = plan_info["cost_fn"](rollout)
        heuristic = heuristic_fn(plan_info["plan_skeleton"], cost_dict, constraint_checker, verbose=False)

    # Number of satisfying particles
    with timer.time("get_satisfying_mask"):
        satisfying_mask = constraint_checker.get_mask(cost_dict, verbose=False)
    num_satisfying = satisfying_mask.sum().item()

    # Best cost
    with timer.time("compute_best_cost"):
        consider_types = {"constraint"}
        if config.optimize_soft_costs:
            consider_types.add("cost")
        costs = cost_reducer(cost_dict, consider_types=consider_types)
        if satisfying_mask.any():
            best_cost = costs[satisfying_mask].min().item()  # note: should consider satisfying mask?
            soft_costs = cost_reducer.soft_costs(cost_dict)
            best_soft_cost = soft_costs[satisfying_mask].min().item()
            indices = torch.arange(config.num_particles, device=soft_costs.device)
            best_idx = indices[satisfying_mask][costs[satisfying_mask].argmin()]
            best_soft_idx = indices[satisfying_mask][soft_costs[satisfying_mask].argmin()]
        else:
            best_cost, best_soft_cost = float("inf"), float("inf")
            best_idx = None
            best_soft_idx = None

    # Update plan info
    plan_info["particles"] = plan_particles
    plan_info["heuristic"] = heuristic
    plan_info["num_satisfying"] = num_satisfying
    plan_info["best_cost"] = best_cost
    plan_info["best_soft_cost"] = best_soft_cost
    plan_info["rollout"] = rollout
    plan_info["best_idx"] = best_idx
    plan_info["best_soft_idx"] = best_soft_idx
    return num_satisfying


def setup_cutamp(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    q_init: Optional[List[float]] = None,
    experiment_id: Optional[str] = None,
    ik_solver: Optional[IKSolver] = None,
    experiment_dir: Optional[Path] = None,
):
    # Validate args and setup experiment logger
    validate_tamp_config(config)
    if experiment_id is None:
        experiment_id = datetime.now().isoformat().split(".")[0]

    exp_logger = (
        ExperimentLogger(name=experiment_id, config=config, experiment_dir=experiment_dir)
        if config.enable_experiment_logging
        else Mock()
    )
    exp_logger.save_env(env)

    # Loading robot can be done offline, so doesn't count towards timing
    tensor_args = TensorDeviceType()
    robot_container = load_robot_container(config.robot, tensor_args)
    if q_init is None:
        q_init = get_q_home(config.robot)
    q_init = tensor_args.to_device(q_init)

    # Load TAMP world and warmup IK solver
    timer = TorchTimer()
    with timer.time("load_tamp_world", log_callback=_log.info):
        world = TAMPWorld(
            env,
            tensor_args,
            robot=robot_container,
            q_init=q_init,
            collision_activation_distance=config.world_activation_distance,
            coll_n_spheres=config.coll_n_spheres,
            coll_sphere_radius=config.coll_sphere_radius,
            ik_solver=ik_solver,
        )
        check_tamp_world_not_in_collision(world, movable_activation_dist=config.movable_activation_distance)

    if config.warmup_ik:
        with timer.time("warmup_ik_solver", log_callback=_log.info):
            world.warmup_ik_solver(config.num_particles)

    # Setup visualizer (doesn't count towards timing)
    visualizer = (
        RerunVisualizer(config, q_init, application_id=env.name, recording_id=experiment_id, spawn=config.rr_spawn)
        if config.enable_visualizer
        else MockVisualizer()
    )
    visualizer.log_tamp_world(world)
    return exp_logger, visualizer, timer, world


def run_cutamp(
    env: TAMPEnvironment,
    config: TAMPConfiguration,
    cost_reducer: CostReducer,
    constraint_checker: ConstraintChecker,
    q_init: Optional[List[float]] = None,
    experiment_id: Optional[str] = None,
    ik_solver: Optional[IKSolver] = None,
    grasps: Optional[dict] = None,
    motion_gen: Optional[MotionGen] = None,
    experiment_dir: Optional[Path] = None,
):
    """Overall cuTAMP algorithm implementation."""
    if config.m2t2_grasps and not grasps:
        _log.warning(f"M2T2 grasps enabled but no grasps provided! Falling back to grasp_dof={config.grasp_dof}")

    # Setup all the things and load the world
    exp_logger, visualizer, timer, world = setup_cutamp(env, config, q_init, experiment_id, ik_solver, experiment_dir)
    particle_initializer = ParticleInitializer(world, config, grasps)

    # Task plan generator
    _log.info(f"Initial State: {world.initial_state}")
    _log.info(f"Goal State: {world.goal_state}")
    with timer.time("get_plan_generator", log_callback=_log.info):
        plan_gen = task_plan_generator(
            world.initial_state,
            world.goal_state,
            operators=all_tamp_operators,
            explored_state_check=config.explored_state_check,
        )

    # Sample initial plans and particles
    found_solution_initially = False
    num_skipped_plans = 0
    with timer.time("sample_initial_plans", log_callback=_log.info):
        plan_queue: List[dict] = []
        plan_count = 0
        for idx in range(config.num_initial_plans):
            try:
                plan_info, has_solution = sample_plan_skeleton(
                    plan_gen, world, config, timer, idx, constraint_checker, cost_reducer, particle_initializer
                )
                if plan_info is None:
                    _log.debug("failed subgraph, skipping...")
                    num_skipped_plans += 1
                    continue
            except StopIteration:
                _log.info("Ran out of plans to sample")
                break
            plan_queue.append(plan_info)
            if has_solution:
                found_solution_initially = True
                break
            plan_count += 1

    # Sort plans by heuristic
    def sort_plans():
        with timer.time("sort_plans"):
            plan_queue.sort(key=lambda x: x["heuristic"])

    sort_plans()
    _log.info(f"Num plans: {len(plan_queue)}, num skipped: {num_skipped_plans}")

    curobo_plan = None
    failure_reason = None
    overall_metrics = {
        "num_optimized_plans": 0,
        "num_initial_plans": plan_count,
        "num_skipped_plans": num_skipped_plans,
        "num_satisfying_final": 0,
        "num_particles": config.num_particles,
        "best_cost": float("inf"),
        "best_soft_cost": float("inf"),
    }
    found_solution = False
    particle_optimizer = ParticleOptimizer(config, cost_reducer, constraint_checker)
    timer.start("first_solution")
    if found_solution_initially:
        found_solution = True
        timer.stop("first_solution")

    # Optimization loop for each skeleton and its particles
    timer.start("start_optimization")
    for idx, plan_info in enumerate(plan_queue):
        opt_iter = idx + 1
        should_break = False
        plan_skeleton = plan_info["plan_skeleton"]
        _log.info(f"[Opt {opt_iter}] Optimizing plan {[op.name for op in plan_skeleton]}")
        _log.info(
            f"[Opt {opt_iter}] plan idx = {plan_info['idx']}, heuristic = {plan_info['heuristic']:.2f}, "
            f"num satisfying = {plan_info['num_satisfying']}"
        )
        best_particle = None

        if config.approach == "optimization":
            has_satisfying, metrics, time_exceeded = particle_optimizer(plan_info, timer, visualizer)

            # For the sake of printing out debug info
            with torch.no_grad():
                rollout = plan_info["rollout_fn"](plan_info["particles"])
                cost_dict = plan_info["cost_fn"](rollout)
                _ = heuristic_fn(plan_skeleton, cost_dict, constraint_checker, verbose=True)

            if metrics["best_cost"] is not None:
                overall_metrics["best_cost"] = min(overall_metrics["best_cost"], metrics["best_cost"])
            if metrics["best_soft_cost"] is not None:
                overall_metrics["best_soft_cost"] = min(overall_metrics["best_soft_cost"], metrics["best_soft_cost"])
            if time_exceeded:
                _log.info(f"Max loop duration reached, stopping optimization")
                should_break = True
            exp_logger.log_dict(f"optimization/opt_{opt_iter:04d}", metrics)
            if has_satisfying:
                best_particle = get_best_particle(plan_info, config, constraint_checker, cost_reducer)
        else:
            # This is the parallelized sampling baseline
            assert config.approach == "sampling"
            num_resample_attempts = 0
            resample_dur = 0.0
            has_satisfying = plan_info["num_satisfying"] > 0
            total_num_satisfying = plan_info["num_satisfying"]
            best_particle = None
            best_soft_costs = []
            elapsed = []

            if not has_satisfying or not config.break_on_satisfying:
                timer.start("resample_duration")
                for resample_idx in range(config.num_resampling_attempts):
                    if config.max_loop_dur is not None and timer.elapsed("start_optimization") >= config.max_loop_dur:
                        _log.info(f"Max loop duration reached, stopping resampling")
                        should_break = True
                        break
                    timer.start("resample_plan_info")
                    num_satisfying = resample_plan_info(
                        plan_info,
                        world,
                        config,
                        timer,
                        cost_reducer,
                        constraint_checker,
                        particle_initializer,
                    )
                    total_num_satisfying += num_satisfying
                    if plan_info["best_soft_cost"] < overall_metrics["best_soft_cost"]:
                        best_soft_idx = plan_info["best_soft_idx"]
                        best_particle = {
                            k: v[best_soft_idx].detach().clone() for k, v in plan_info["particles"].items()
                        }

                    overall_metrics["best_cost"] = min(overall_metrics["best_cost"], plan_info["best_cost"])
                    overall_metrics["best_soft_cost"] = min(
                        overall_metrics["best_soft_cost"], plan_info["best_soft_cost"]
                    )

                    # Keep track of the best soft cost since start of resampling
                    best_soft_costs.append(overall_metrics["best_soft_cost"])
                    elapsed.append(timer.elapsed("start_optimization"))

                    resample_plan_info_dur = timer.stop("resample_plan_info")
                    _log.debug(
                        f"[Plan {plan_info['idx'] + 1}] Resample attempt {resample_idx + 1}/{config.num_resampling_attempts}, "
                        f"{num_satisfying}/{config.num_particles} satisfying particles. Total satisfying {total_num_satisfying}. "
                        f"Took {resample_plan_info_dur:.2f}s"
                    )
                    has_satisfying = num_satisfying > 0
                    num_resample_attempts += 1

                    # Visualize best particle rollout state
                    rollout = plan_info["rollout"]
                    best_soft_idx = plan_info["best_soft_idx"]
                    if best_soft_idx is None:
                        best_soft_idx = 0
                    visualizer.set_time_sequence(f"samp", num_resample_attempts)
                    q_last = rollout["confs"][best_soft_idx, -1].tolist()
                    visualizer.set_joint_positions(q_last)
                    for obj in rollout["obj_to_pose"]:
                        mat4x4_last = rollout["obj_to_pose"][obj][best_soft_idx, -1]
                        visualizer.log_mat4x4(f"world/{obj}", mat4x4_last)

                    if has_satisfying:
                        if timer.has_timer("first_solution"):
                            time_to_first_sol = timer.stop("first_solution")
                            _log.info(f"Found first solution in {time_to_first_sol:.2f}s after sampling plans")
                        if config.break_on_satisfying:
                            should_break = True
                            break
                resample_dur = timer.stop("resample_duration")
                _log.info(f"Total resample duration: {resample_dur:.2f}s")
            else:
                _log.info("Already has satisfying particles, skipping resampling")
                overall_metrics["best_cost"] = min(overall_metrics["best_cost"], plan_info["best_cost"])
                overall_metrics["best_soft_cost"] = min(overall_metrics["best_soft_cost"], plan_info["best_soft_cost"])
                if config.break_on_satisfying:
                    should_break = True

            metrics = {
                "plan_skeleton": [str(op) for op in plan_skeleton],
                "num_particles": config.num_particles,
                "num_resample_attempts": num_resample_attempts,
                "resample_duration": resample_dur,
                "num_satisfying_final": total_num_satisfying,
                "total_num_particles": config.num_particles * (num_resample_attempts + 1),
                "best_cost": overall_metrics["best_cost"],
                "best_soft_cost": overall_metrics["best_soft_cost"],
                "best_soft_costs": best_soft_costs,
                "elapsed": elapsed,
            }
            exp_logger.log_dict(f"sampling/samp_{opt_iter:04d}", metrics)
            has_satisfying = total_num_satisfying > 0
            overall_metrics["num_satisfying_final"] = total_num_satisfying

            # Log best particle as last
            if best_particle is not None:
                rollout = plan_info["rollout_fn"]({k: v[None] for k, v in best_particle.items()})
                visualizer.set_time_sequence(f"samp", num_resample_attempts)
                q_last = rollout["confs"][0, -1].tolist()
                visualizer.set_joint_positions(q_last)

                for obj in rollout["obj_to_pose"]:
                    mat4x4_last = rollout["obj_to_pose"][obj][0, -1]
                    visualizer.log_mat4x4(f"world/{obj}", mat4x4_last)

        # Now we've either optimized or resampled
        overall_metrics["num_optimized_plans"] += 1
        if has_satisfying:
            found_solution = True
            ranked_particles = get_ranked_satisfying_particles(
                plan_info, config, constraint_checker, cost_reducer, visualizer
            )
            if config.curobo_plan:
                # Need to cache initial pose as cuRobo dynamically updates during planning which sucks ass
                obj_to_initial_pose = {obj.name: world.get_object_pose(obj) for obj in world.movables}

                # IMPORTANT: this line comes after the previous as it messes with cuRobo internal memory,
                # also call it only once!
                if motion_gen is not None:
                    all_world_cfg = get_world_cfg(world.env, include_movables=True)
                    motion_gen.update_world(all_world_cfg)
                    _log.info(f"Updated motion gen with world cfg")

                num_satisfying = ranked_particles["q0"].shape[0]
                max_attempts = min(config.max_motion_refine_attempts or num_satisfying, num_satisfying)
                for curr_idx in range(max_attempts):
                    _log.info(f"Trying cuRobo planning with satisfying particle {curr_idx + 1}/{max_attempts} ({num_satisfying} total satisfying)")
                    curr_particle = {k: v[curr_idx] for k, v in ranked_particles.items()}
                    try:
                        curobo_plan = solve_curobo(
                            plan_info,
                            curr_particle,
                            world,
                            config,
                            timer,
                            visualizer,
                            obj_to_initial_pose=obj_to_initial_pose,
                            timeline=f"curobo_{curr_idx}",
                            motion_gen=motion_gen,
                        )
                        _log.info("Successful plan found!")
                        failure_reason = None
                        break
                    except MotionPlanningError as e:
                        _log.warning(f"Failed to motion plan: {e}")
                else:
                    # All attempted particles failed motion planning
                    if curobo_plan is None:
                        failure_reason = (
                            f"Motion planning failed for {max_attempts}/{num_satisfying} satisfying particle(s)"
                        )

            overall_metrics["num_satisfying_final"] = metrics["num_satisfying_final"]
            overall_metrics["final_plan_skeleton"] = [str(op) for op in plan_skeleton]
            _log.debug(f"Total num satisfying {metrics['num_satisfying_final']}")
            if config.curobo_plan and curobo_plan is None:
                # Motion refinement failed, try next skeleton. Intentionally overrides should_break
                # set by break_on_satisfying during resampling — we don't want to stop on a skeleton
                # where motion planning failed. The max_loop_dur timeout will still be checked at the
                # start of the next skeleton's resampling loop.
                _log.info(f"Motion refinement failed for skeleton {[op.name for op in plan_skeleton]}, trying next")
                should_break = False
                failure_reason = None
            elif config.break_on_satisfying:
                should_break = True

        if should_break:
            break

        # TODO: complete version of our algorithm that adds additional skeletons to the queue, resorts, revisits
        #  skeletons, etc.
        # new_plan_info = sample_plan_skeleton()
        # if new_plan_info is not None:
        #     plan_queue.append(new_plan_info)
        #     sort_plans()

    opt_elapsed = timer.stop("start_optimization")
    _log.debug(f"Optimization loop took roughly {opt_elapsed:.2f}s")
    if not found_solution:
        if len(plan_queue) == 0:
            if num_skipped_plans > 0:
                failure_reason = f"All {num_skipped_plans} plan skeleton(s) failed particle initialization"
            else:
                failure_reason = "No valid plan skeletons found for the given goal"
        elif failure_reason is None:
            # Had plans but no satisfying particles (or timed out)
            optimized = overall_metrics["num_optimized_plans"]
            total = len(plan_queue)
            if optimized < total:
                failure_reason = (
                    f"No satisfying particles found after optimizing "
                    f"{optimized}/{total} plan(s) (time budget {config.max_loop_dur}s exceeded)"
                )
            else:
                failure_reason = (
                    f"No satisfying particles found after optimizing all {total} plan(s)"
                )
        _log.warning(failure_reason)
    _log.debug(f"Best cost: {overall_metrics['best_cost']:.4f}, soft cost: {overall_metrics['best_soft_cost']:.4f}")

    # Dump metrics out
    overall_metrics["found_solution"] = found_solution
    exp_logger.log_dict("overall_metrics", overall_metrics)
    exp_logger.log_dict("timer_metrics", timer.get_summaries())

    # Log constraint and cost multipliers
    exp_logger.log_dict("multipliers", cost_reducer.cost_config)
    exp_logger.log_dict("tolerances", constraint_checker.constraint_config)
    return curobo_plan, overall_metrics["num_satisfying_final"], failure_reason
