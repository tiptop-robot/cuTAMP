# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from dataclasses import dataclass

import torch
from jaxtyping import Float

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType

from .utils import RerunRobot as RerunRobot  # noqa: F401


@dataclass(frozen=True)
class RobotContainer:
    name: str
    kin_model: CudaRobotModel
    joint_limits: Float[torch.Tensor, "2 d"]
    # Note: in tool frame, not end-effector
    gripper_spheres: Float[torch.Tensor, "n 4"]
    # Transformation from tool pose to end-effector (defined in cuRobo config)
    tool_from_ee: Float[torch.Tensor, "4 4"]


# Import robot modules to trigger their register_robot() calls.
# This must happen after RobotContainer is defined (registry imports it).
from . import franka as _franka  # noqa: E402, F401
from . import franka_robotiq as _franka_robotiq  # noqa: E402, F401
from . import ur5 as _ur5  # noqa: E402, F401

# Re-export registry functions as the primary public API
from .registry import (  # noqa: E402, F401
    RobotDefinition as RobotDefinition,
    get_curobo_cfg as get_curobo_cfg,
    get_ik_solver as get_ik_solver,
    get_kinematics_model as get_kinematics_model,
    get_q_home as get_q_home,
    get_robot_definition as get_robot_definition,
    list_robots as list_robots,
    load_rerun_robot as load_rerun_robot,
    load_robot_container as load_robot_container,
    register_robot as register_robot,
)

# ---------------------------------------------------------------------------
# Backward-compatible re-exports from robot modules
# (consumed by tiptop and other callers that import from cutamp.robots directly)
# ---------------------------------------------------------------------------
from .franka import (  # noqa: E402, F401
    fr3_franka_curobo_cfg as fr3_franka_curobo_cfg,
    fr3_franka_neutral_joint_positions as fr3_franka_neutral_joint_positions,
    franka_curobo_cfg as franka_curobo_cfg,
    franka_neutral_joint_positions as franka_neutral_joint_positions,
    get_fr3_franka_gripper_spheres as get_fr3_franka_gripper_spheres,
    get_fr3_franka_ik_solver as get_fr3_franka_ik_solver,
    get_fr3_franka_kinematics_model as get_fr3_franka_kinematics_model,
    get_franka_gripper_spheres as get_franka_gripper_spheres,
    get_franka_ik_solver as get_franka_ik_solver,
    get_franka_kinematics_model as get_franka_kinematics_model,
    load_fr3_franka_rerun as load_fr3_franka_rerun,
    load_franka_rerun as load_franka_rerun,
)
from .franka_robotiq import (  # noqa: E402, F401
    fr3_robotiq_curobo_cfg as fr3_robotiq_curobo_cfg,
    fr3_robotiq_neutral_joint_positions as fr3_robotiq_neutral_joint_positions,
    get_fr3_robotiq_gripper_spheres as get_fr3_robotiq_gripper_spheres,
    get_fr3_robotiq_ik_solver as get_fr3_robotiq_ik_solver,
    get_fr3_robotiq_kinematics_model as get_fr3_robotiq_kinematics_model,
    get_panda_robotiq_gripper_spheres as get_panda_robotiq_gripper_spheres,
    get_panda_robotiq_ik_solver as get_panda_robotiq_ik_solver,
    get_panda_robotiq_kinematics_model as get_panda_robotiq_kinematics_model,
    load_fr3_robotiq_rerun as load_fr3_robotiq_rerun,
    load_panda_robotiq_rerun as load_panda_robotiq_rerun,
    panda_robotiq_curobo_cfg as panda_robotiq_curobo_cfg,
    panda_robotiq_neutral_joint_positions as panda_robotiq_neutral_joint_positions,
)
from .ur5 import (  # noqa: E402, F401
    get_ur5_gripper_spheres as get_ur5_gripper_spheres,
    get_ur5_ik_solver as get_ur5_ik_solver,
    get_ur5_kinematics_model as get_ur5_kinematics_model,
    load_ur5_rerun as load_ur5_rerun,
    ur5_curobo_cfg as ur5_curobo_cfg,
    ur5_home as ur5_home,
)

# Backward-compatible per-robot container loaders.
# New code should use load_robot_container("robot_name", tensor_args) instead.


def load_panda_container(tensor_args: TensorDeviceType) -> RobotContainer:
    return load_robot_container("panda", tensor_args)


def load_fr3_franka_container(tensor_args: TensorDeviceType) -> RobotContainer:
    return load_robot_container("fr3_franka", tensor_args)


def load_panda_robotiq_container(tensor_args: TensorDeviceType) -> RobotContainer:
    return load_robot_container("panda_robotiq", tensor_args)


def load_fr3_robotiq_container(tensor_args: TensorDeviceType) -> RobotContainer:
    return load_robot_container("fr3_robotiq", tensor_args)


def load_ur5_container(tensor_args: TensorDeviceType) -> RobotContainer:
    return load_robot_container("ur5", tensor_args)
