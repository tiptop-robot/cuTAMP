"""Test a robot configuration by loading it, sampling joint configs, running FK/IK, and planning motions.

Similar to cuRobo's robot configuration tutorial but integrated with cuTAMP's registry and Rerun visualization.
See: https://curobo.org/tutorials/1_robot_configuration.html#test-robot-configuration

Usage:
    cutamp-test-robot --robot fr3_robotiq
    cutamp-test-robot --robot fr3_robotiq --num-samples 5
    cutamp-test-robot --robot-yml path/to/cfg.yml
"""

import argparse
import time

import rerun as rr
import torch
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

from cutamp.robots.registry import (
    get_curobo_cfg,
    get_robot_definition,
    list_robots,
    load_rerun_robot,
    load_robot_container,
)
from cutamp.scripts.viz_spheres import _log_frame_axes, _log_spheres_as_ellipsoids


def _sample_valid_joints(container, num_samples: int, max_attempts: int = 1000) -> torch.Tensor:
    """Sample joint configurations within joint limits."""
    lower = container.joint_limits[0]
    upper = container.joint_limits[1]
    samples = []
    for _ in range(max_attempts):
        if len(samples) >= num_samples:
            break
        q = lower + (upper - lower) * torch.rand_like(lower)
        samples.append(q)
    return torch.stack(samples[:num_samples])


def test_robot(
    robot_name: str,
    robot_yml: str | None = None,
    num_samples: int = 3,
    plan_motion: bool = True,
):
    rr.init(f"cutamp-test-robot/{robot_name or 'custom'}", spawn=True)

    tensor_args = TensorDeviceType()
    empty_world = WorldConfig()

    # Load via registry or raw YML
    if robot_yml is not None:
        from curobo.util_file import load_yaml

        robot_cfg_dict = load_yaml(robot_yml)
        robot_cfg = RobotConfig.from_dict(robot_cfg_dict["robot_cfg"])
        from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

        kin_model = CudaRobotModel(robot_cfg.kinematics)
        joint_limits = kin_model.kinematics_config.joint_limits.position
        print(f"Loaded robot from YML: {robot_yml}")
        print(f"  Joint limits shape: {joint_limits.shape}")
        print(f"  EE link: {robot_cfg.kinematics.ee_link}")

        # Build IK solver
        ik_config = IKSolverConfig.load_from_robot_config(robot_cfg, empty_world, num_seeds=12)
        ik_solver = IKSolver(ik_config)

        # Build MotionGen if requested
        motion_gen = None
        if plan_motion:
            mg_config = MotionGenConfig.load_from_robot_config(
                robot_cfg=robot_cfg_dict, world_model=empty_world, use_cuda_graph=False
            )
            motion_gen = MotionGen(mg_config)

        # For raw YML, we use retract config as home
        retract = robot_cfg_dict["robot_cfg"]["kinematics"]["cspace"]["retract_config"]
        q_home = torch.tensor(retract, dtype=torch.float32, device=tensor_args.device)

        # No rerun robot or gripper spheres for raw YML
        rerun_robot = None
        container = None
    else:
        defn = get_robot_definition(robot_name)
        container = load_robot_container(robot_name, tensor_args)
        kin_model = container.kin_model
        joint_limits = container.joint_limits
        q_home = torch.tensor(list(defn.q_home), dtype=torch.float32, device=tensor_args.device)

        print(f"Robot: {robot_name}")
        print(f"  Arm joints: {defn.n_arm_joints}")
        print(f"  Joint limits shape: {joint_limits.shape}")
        print(f"  Home config: {list(defn.q_home)}")

        # Build IK solver and MotionGen via registry config
        cfg_dict = get_curobo_cfg(robot_name)
        robot_cfg = RobotConfig.from_dict(cfg_dict["robot_cfg"])
        ik_config = IKSolverConfig.load_from_robot_config(robot_cfg, empty_world, num_seeds=12)
        ik_solver = IKSolver(ik_config)

        motion_gen = None
        if plan_motion:
            mg_config = MotionGenConfig.load_from_robot_config(
                robot_cfg=cfg_dict, world_model=empty_world, use_cuda_graph=False
            )
            motion_gen = MotionGen(mg_config)

        rerun_robot = load_rerun_robot(robot_name)

    # --- Test 1: FK at home config ---
    print("\n--- Test 1: FK at home config ---")
    state = kin_model.get_state(q_home[None])
    ee_pose = state.ee_pose
    ee_pos = ee_pose.position[0].cpu().tolist()
    print(f"  EE position at home: [{ee_pos[0]:.4f}, {ee_pos[1]:.4f}, {ee_pos[2]:.4f}]")

    if rerun_robot is not None:
        rerun_robot.set_joint_positions(q_home.cpu().tolist())

    # Visualize collision spheres at home
    link_spheres = state.link_spheres_tensor[0].cpu()
    valid = link_spheres[:, 3] > 0
    link_spheres = link_spheres[valid]
    _log_spheres_as_ellipsoids("collision/link_spheres", link_spheres, color=(100, 149, 237, 120))
    print(f"  Link collision spheres: {link_spheres.shape[0]}")

    # Visualize EE frame and gripper spheres
    world_from_ee = state.ee_pose.get_matrix()[0]
    _log_frame_axes("frames/ee_frame", world_from_ee, length=0.08)

    if container is not None:
        tool_from_ee = container.tool_from_ee
        ee_from_tool = torch.inverse(tool_from_ee)
        world_from_tool = world_from_ee @ ee_from_tool
        _log_frame_axes("frames/tool_frame", world_from_tool, length=0.06)

        gs = container.gripper_spheres
        gs_world = (world_from_tool[:3, :3] @ gs[:, :3].T).T + world_from_tool[:3, 3]
        gs_world = torch.cat([gs_world, gs[:, 3:]], dim=1)
        _log_spheres_as_ellipsoids("collision/gripper_spheres", gs_world, color=(255, 165, 0, 150))
        print(f"  Gripper spheres: {gs.shape[0]}")

    # --- Test 2: Sample configs and FK/IK roundtrip ---
    print(f"\n--- Test 2: FK/IK roundtrip ({num_samples} samples) ---")
    lower = joint_limits[0]
    upper = joint_limits[1]
    ik_successes = 0

    for i in range(num_samples):
        q_sample = lower + (upper - lower) * torch.rand_like(lower)
        fk_state = kin_model.get_state(q_sample[None])
        target_pose = fk_state.ee_pose
        target_pos = target_pose.position[0].cpu().tolist()

        ik_result = ik_solver.solve_single(target_pose)
        if ik_result.success[0]:
            ik_successes += 1
            # Check position error
            ik_q = ik_result.solution[0, 0]  # [batch, seeds, dof] -> first batch, first seed
            ik_fk = kin_model.get_state(ik_q[None]).ee_pose
            pos_err = (target_pose.position - ik_fk.position).norm().item()
            status = f"OK (pos err: {pos_err:.6f}m)"
        else:
            status = "FAILED"

        print(f"  Sample {i + 1}: EE=[{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}] -> IK {status}")

    print(f"  IK success rate: {ik_successes}/{num_samples}")

    # --- Test 3: Motion planning ---
    if motion_gen is not None:
        print(f"\n--- Test 3: Motion planning ({num_samples} targets) ---")
        print("  Warming up MotionGen...")
        t0 = time.perf_counter()
        motion_gen.warmup()
        print(f"  Warmup took {time.perf_counter() - t0:.2f}s")

        plan_config = MotionGenPlanConfig(timeout=2.0, enable_finetune_trajopt=False)
        start_js = JointState.from_position(q_home[None])
        plan_successes = 0

        for i in range(num_samples):
            q_sample = lower + (upper - lower) * torch.rand_like(lower)
            target_state = kin_model.get_state(q_sample[None])
            target_pose = Pose.from_matrix(target_state.ee_pose.get_matrix()[0])

            t0 = time.perf_counter()
            result = motion_gen.plan_single(start_js, target_pose, plan_config)
            plan_dur = time.perf_counter() - t0

            if result.success:
                plan_successes += 1
                plan = result.get_interpolated_plan()
                dt = result.interpolation_dt

                # Animate trajectory in Rerun
                if rerun_robot is not None:
                    for t_idx, q_step in enumerate(plan.position):
                        rr.set_time_seconds("motion", t_idx * dt)
                        rerun_robot.set_joint_positions(q_step.cpu().tolist())

                print(f"  Plan {i + 1}: OK ({plan.position.shape[0]} steps, {plan_dur:.3f}s)")
            else:
                print(f"  Plan {i + 1}: FAILED ({plan_dur:.3f}s) - {result.status}")

        print(f"  Motion planning success rate: {plan_successes}/{num_samples}")
    else:
        print("\n--- Test 3: Motion planning (skipped) ---")

    print("\nDone! Check the Rerun viewer for visualizations.")


def entrypoint():
    parser = argparse.ArgumentParser(
        description="Test a robot configuration with FK/IK roundtrip and motion planning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--robot", choices=list_robots(), help="Robot name from registry")
    group.add_argument("--robot-yml", type=str, help="Path to a cuRobo robot YAML config file")
    parser.add_argument("--num-samples", type=int, default=3, help="Number of random configs to test")
    parser.add_argument("--no-motion-plan", action="store_true", help="Skip motion planning test")
    args = parser.parse_args()

    test_robot(
        robot_name=args.robot,
        robot_yml=args.robot_yml,
        num_samples=args.num_samples,
        plan_motion=not args.no_motion_plan,
    )


if __name__ == "__main__":
    entrypoint()
