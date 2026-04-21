"""Visualize robot collision spheres, gripper spheres, tool frame, and approach direction in Rerun.

Usage:
    cutamp-viz-spheres --robot fr3_robotiq
    cutamp-viz-spheres --robot fr3_robotiq --q 0 -0.6 0 -2.5 0 1.9 0
"""

import argparse

import numpy as np
import rerun as rr
import torch
from curobo.types.base import TensorDeviceType

from cutamp.robots.registry import get_robot_definition, list_robots, load_robot_container, load_rerun_robot


def _log_frame_axes(path: str, transform: torch.Tensor, length: float = 0.05):
    """Log a coordinate frame as RGB arrows (R=X, G=Y, B=Z)."""
    origin = transform[:3, 3].cpu().numpy()
    rot = transform[:3, :3].cpu().numpy()
    origins = np.tile(origin, (3, 1))
    vectors = rot.T * length  # columns are x, y, z axes
    colors = np.array([[255, 0, 0], [0, 255, 0], [0, 0, 255]], dtype=np.uint8)
    rr.log(path, rr.Arrows3D(origins=origins, vectors=vectors, colors=colors))


def _log_spheres_as_ellipsoids(path: str, spheres: torch.Tensor, color: tuple[int, int, int, int]):
    """Log spheres as proper Ellipsoids3D (not Points3D) for accurate size rendering."""
    centers = spheres[:, :3].cpu().numpy()
    radii = spheres[:, 3].cpu().numpy()
    # Ellipsoids3D uses half-sizes for each axis
    half_sizes = np.stack([radii, radii, radii], axis=1)
    rr.log(path, rr.Ellipsoids3D(centers=centers, half_sizes=half_sizes, colors=[color] * len(centers)))


def viz_spheres(robot_name: str, q: list[float] | None = None):
    rr.init(f"cutamp-viz-spheres/{robot_name}", spawn=True)

    tensor_args = TensorDeviceType()
    defn = get_robot_definition(robot_name)
    container = load_robot_container(robot_name, tensor_args)

    # Use provided joint config or home config
    if q is not None:
        q_tensor = torch.tensor(q, dtype=torch.float32, device=tensor_args.device)
    else:
        q_tensor = torch.tensor(list(defn.q_home), dtype=torch.float32, device=tensor_args.device)

    # Visualize robot mesh
    rerun_robot = load_rerun_robot(robot_name)
    rerun_robot.set_joint_positions(q_tensor.cpu().tolist())

    # Forward kinematics
    state = container.kin_model.get_state(q_tensor[None])

    # 1. Link collision spheres from cuRobo (in world frame at the given config)
    link_spheres = state.link_spheres_tensor[0].cpu()
    valid = link_spheres[:, 3] > 0
    link_spheres = link_spheres[valid]
    _log_spheres_as_ellipsoids("collision/link_spheres", link_spheres, color=(100, 149, 237, 120))

    # 2. EE frame
    world_from_ee = state.ee_pose.get_matrix()[0]
    _log_frame_axes("frames/ee_frame", world_from_ee, length=0.08)

    # 3. Tool frame (= EE @ inv(tool_from_ee))
    tool_from_ee = container.tool_from_ee
    ee_from_tool = torch.inverse(tool_from_ee)
    world_from_tool = world_from_ee @ ee_from_tool
    _log_frame_axes("frames/tool_frame", world_from_tool, length=0.06)

    # 4. Gripper spheres in world frame
    # Gripper spheres are defined in the tool frame (z-up origin)
    gripper_spheres = container.gripper_spheres  # [n, 4]
    gs_positions = gripper_spheres[:, :3]
    gs_radii = gripper_spheres[:, 3:]

    # Transform positions to world frame via the tool frame
    gs_world = (world_from_tool[:3, :3] @ gs_positions.T).T + world_from_tool[:3, 3]
    gs_world = torch.cat([gs_world, gs_radii], dim=1)
    _log_spheres_as_ellipsoids("collision/gripper_spheres", gs_world, color=(255, 165, 0, 150))

    # 5. Approach direction arrow (negative Z in EE frame, used by motion_solver.py)
    ee_origin = world_from_ee[:3, 3].cpu().numpy()
    ee_z = world_from_ee[:3, 2].cpu().numpy()
    approach_vec = -ee_z * 0.1  # 10cm arrow in approach direction
    rr.log(
        "frames/approach_direction",
        rr.Arrows3D(origins=[ee_origin], vectors=[approach_vec], colors=[[255, 255, 0]]),
    )

    # Print summary
    print(f"Robot: {robot_name}")
    print(f"Joint config: {q_tensor.cpu().tolist()}")
    print(f"EE position: {world_from_ee[:3, 3].cpu().tolist()}")
    print(f"Link collision spheres: {link_spheres.shape[0]}")
    print(f"Gripper spheres: {gripper_spheres.shape[0]}")
    print(f"tool_from_ee translation: {tool_from_ee[:3, 3].cpu().tolist()}")


def entrypoint():
    parser = argparse.ArgumentParser(
        description="Visualize robot collision spheres and frames in Rerun.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--robot", required=True, choices=list_robots(), help="Robot name from registry")
    parser.add_argument("--q", nargs="+", type=float, default=None, help="Joint configuration (space-separated)")
    args = parser.parse_args()
    viz_spheres(args.robot, args.q)


if __name__ == "__main__":
    entrypoint()
