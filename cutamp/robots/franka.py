# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import logging
import os
from pathlib import Path

import torch
from jaxtyping import Float
from yourdfpy import URDF

from curobo.types.base import TensorDeviceType

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

from curobo.geom.types import WorldConfig
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_configs_path, join_path, load_yaml, get_assets_path
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from cutamp.robots.registry import RobotDefinition, register_robot
from cutamp.robots.utils import RerunRobot

_log = logging.getLogger(__name__)

franka_neutral_joint_positions = (-0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398, 0.04)


def franka_curobo_cfg():
    return load_yaml(join_path(get_robot_configs_path(), "franka.yml"))


def _franka_cfg_dict() -> dict:
    return franka_curobo_cfg()["robot_cfg"]


def get_franka_kinematics_model() -> CudaRobotModel:
    """cuRobo robot kinematics model."""
    robot_cfg = _franka_cfg_dict()
    robot_cfg = RobotConfig.from_dict(robot_cfg)
    kinematics_model = CudaRobotModel(robot_cfg.kinematics)
    return kinematics_model


def get_franka_ik_solver(
    world_cfg: WorldConfig,
    num_seeds: int = 12,
    self_collision_opt: bool = False,
    self_collision_check: bool = True,
    use_particle_opt: bool = False,
) -> IKSolver:
    """
    cuRobo IK solver for Franka Panda. 12 seeds is sufficient according to Bala for the Franka.
    The other default settings give good performance for initializing configuration seeds to warm start optimization.
    """
    robot_cfg_dict = _franka_cfg_dict()
    robot_cfg = RobotConfig.from_dict(robot_cfg_dict)
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        num_seeds=num_seeds,
        self_collision_opt=self_collision_opt,
        self_collision_check=self_collision_check,
        use_particle_opt=use_particle_opt,
    )
    ik_solver = IKSolver(ik_config)
    return ik_solver


def get_franka_gripper_spheres(
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> Float[torch.Tensor, "num_spheres 4"]:
    """
    Get the collision spheres for the Franka gripper.
    IMPORTANT: note they are in the origin frame with z-up (not the conventional z-down gripper frame).
    """
    assets_dir = Path(__file__).parent / "assets"
    spheres_pt = assets_dir / "franka_gripper_spheres.pt"
    if not spheres_pt.exists():
        raise FileNotFoundError(f"Franka gripper spheres file not found at {spheres_pt}")
    spheres = torch.load(spheres_pt, map_location=tensor_args.device, weights_only=True)
    assert spheres.ndim == 2 and spheres.shape[1] == 4, f"Invalid shape for Franka gripper spheres: {spheres.shape}"

    # It turns out there are some spheres with negative radii, so ignore those
    spheres = spheres[spheres[:, 3] > 0]
    return spheres


def load_franka_rerun(load_mesh: bool = True) -> RerunRobot:
    robot_cfg = franka_curobo_cfg()["robot_cfg"]
    urdf_rel_path = robot_cfg["kinematics"]["urdf_path"]
    urdf_path = join_path(get_assets_path(), urdf_rel_path)
    urdf = URDF.load(urdf_path)
    return RerunRobot("panda", urdf, q_neutral=(*franka_neutral_joint_positions, 0.04), load_mesh=load_mesh)


fr3_franka_neutral_joint_positions = (0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0)


def fr3_franka_curobo_cfg():
    assets_dir = Path(__file__).parent / "assets"
    cfg = load_yaml(str(assets_dir / "fr3_franka.yml"))
    keys = ["external_asset_path", "external_robot_configs_path"]
    for key in keys:
        if key not in cfg["robot_cfg"]["kinematics"]:
            cfg["robot_cfg"]["kinematics"][key] = str(assets_dir)
    return cfg


def _fr3_franka_cfg_dict() -> dict:
    return fr3_franka_curobo_cfg()["robot_cfg"]


def get_fr3_franka_kinematics_model() -> CudaRobotModel:
    """cuRobo robot kinematics model."""
    robot_cfg = RobotConfig.from_dict(_fr3_franka_cfg_dict())
    return CudaRobotModel(robot_cfg.kinematics)


def get_fr3_franka_ik_solver(
    world_cfg: WorldConfig,
    num_seeds: int = 12,
    self_collision_opt: bool = False,
    self_collision_check: bool = True,
    use_particle_opt: bool = False,
) -> IKSolver:
    """cuRobo IK solver for FR3 with Panda native hand."""
    robot_cfg = RobotConfig.from_dict(_fr3_franka_cfg_dict())
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        num_seeds=num_seeds,
        self_collision_opt=self_collision_opt,
        self_collision_check=self_collision_check,
        use_particle_opt=use_particle_opt,
    )
    return IKSolver(ik_config)


def get_fr3_franka_gripper_spheres(
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> Float[torch.Tensor, "num_spheres 4"]:
    """
    Get the collision spheres for the Panda native gripper.
    IMPORTANT: note they are in the origin frame with z-up (not the conventional z-down gripper frame).
    """
    return get_franka_gripper_spheres(tensor_args)


def load_fr3_franka_rerun(load_mesh: bool = True) -> RerunRobot:
    assets_dir = Path(__file__).parent / "assets"
    urdf_path = str(assets_dir / "fr3_franka.urdf")

    def _locate_curobo_asset(fname: str) -> str:
        if fname.startswith("package://"):
            return os.path.join(get_assets_path(), "robot", fname.replace("package://", ""))
        return os.path.join(Path(__file__).parent, fname)

    urdf = URDF.load(urdf_path, filename_handler=_locate_curobo_asset)
    return RerunRobot(
        "panda",
        urdf,
        q_neutral=(*fr3_franka_neutral_joint_positions, 0.04, 0.04),
        load_mesh=load_mesh,
    )


## ---------------------------------------------------------------------------
## Robot registrations
## ---------------------------------------------------------------------------

register_robot(
    RobotDefinition(
        name="panda",
        curobo_cfg_fn=franka_curobo_cfg,
        gripper_spheres_fn=get_franka_gripper_spheres,
        rerun_fn=load_franka_rerun,
        q_home=franka_neutral_joint_positions[:7],
        tool_from_ee_quat=(0.0, 1.0, 0.0, 0.0),
        tool_from_ee_translation=(0.0, 0.0, 0.105),
    )
)

register_robot(
    RobotDefinition(
        name="fr3_franka",
        curobo_cfg_fn=fr3_franka_curobo_cfg,
        gripper_spheres_fn=get_fr3_franka_gripper_spheres,
        rerun_fn=load_fr3_franka_rerun,
        q_home=fr3_franka_neutral_joint_positions,
        tool_from_ee_quat=(0.0, 1.0, 0.0, 0.0),
        tool_from_ee_translation=(0.0, 0.0, 0.105),
    )
)


if __name__ == "__main__":
    import rerun as rr

    rr.init("franka_test", spawn=True)
    franka = load_franka_rerun()

    kin_model = get_franka_kinematics_model()

    q = torch.tensor(franka_neutral_joint_positions[:7], device="cuda")[None]
    state = kin_model.get_state(q)

    spheres = state.link_spheres_tensor[0].cpu()
    xyz, radii = spheres[:, :3], spheres[:, 3]
    rr.log("spheres", rr.Points3D(positions=xyz, radii=radii))

    gripper_spheres = get_franka_gripper_spheres().cpu()
    rr.log("gripper_spheres", rr.Points3D(positions=gripper_spheres[:, :3], radii=gripper_spheres[:, 3]))
