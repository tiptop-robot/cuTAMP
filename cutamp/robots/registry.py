"""
Robot registry for cuTAMP. Provides a declarative way to register robot configurations
so that adding a new robot only requires defining a RobotDefinition and calling register_robot().
"""

from dataclasses import dataclass
from typing import Callable, Sequence

import roma
import torch
from jaxtyping import Float

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.transform import quaternion_to_matrix
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig


@dataclass(frozen=True)
class RobotDefinition:
    """Declarative definition of a robot for cuTAMP.

    Each robot module creates one of these and calls register_robot() to make the robot
    available system-wide. This eliminates per-robot boilerplate for kinematics models,
    IK solvers, containers, etc.
    """

    name: str
    curobo_cfg_fn: Callable[[], dict]
    gripper_spheres_fn: Callable[[TensorDeviceType], Float[torch.Tensor, "n 4"]]
    rerun_fn: Callable[[bool], "RerunRobot"]  # forward ref to avoid circular import
    q_home: Sequence[float]
    n_arm_joints: int = 7

    # tool_from_ee transform: specify either rpy or quat (wxyz), plus translation
    tool_from_ee_rpy: tuple[float, float, float] | None = None
    tool_from_ee_quat: tuple[float, float, float, float] | None = None
    tool_from_ee_translation: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        if self.tool_from_ee_rpy is None and self.tool_from_ee_quat is None:
            raise ValueError(f"Robot '{self.name}': must specify either tool_from_ee_rpy or tool_from_ee_quat")
        if self.tool_from_ee_rpy is not None and self.tool_from_ee_quat is not None:
            raise ValueError(f"Robot '{self.name}': specify only one of tool_from_ee_rpy or tool_from_ee_quat")


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: dict[str, RobotDefinition] = {}


def register_robot(defn: RobotDefinition):
    """Register a robot definition. Called at module-import time by each robot module."""
    if defn.name in _registry:
        raise ValueError(f"Robot '{defn.name}' is already registered")
    _registry[defn.name] = defn


def get_robot_definition(name: str) -> RobotDefinition:
    """Get the definition for a registered robot."""
    if name not in _registry:
        raise ValueError(f"Unknown robot: {name}. Registered robots: {list_robots()}")
    return _registry[name]


def list_robots() -> list[str]:
    """Return names of all registered robots."""
    return list(_registry.keys())


# ---------------------------------------------------------------------------
# Generic helpers derived from RobotDefinition
# ---------------------------------------------------------------------------


def _build_tool_from_ee(defn: RobotDefinition, tensor_args: TensorDeviceType) -> Float[torch.Tensor, "4 4"]:
    """Build the 4x4 tool_from_ee transform from a RobotDefinition."""
    tool_from_ee = torch.eye(4, device=tensor_args.device)
    if defn.tool_from_ee_rpy is not None:
        rpy = tensor_args.to_device(list(defn.tool_from_ee_rpy))
        tool_from_ee[:3, :3] = roma.euler_to_rotmat("XYZ", rpy)
    else:
        quat = tensor_args.to_device(list(defn.tool_from_ee_quat))
        tool_from_ee[:3, :3] = quaternion_to_matrix(quat[None])[0]
    tool_from_ee[:3, 3] = tensor_args.to_device(list(defn.tool_from_ee_translation))
    return tool_from_ee


def get_curobo_cfg(name: str) -> dict:
    """Get the cuRobo configuration dict for a registered robot."""
    defn = get_robot_definition(name)
    return defn.curobo_cfg_fn()


def get_kinematics_model(name: str) -> CudaRobotModel:
    """Get the cuRobo kinematics model for a registered robot."""
    cfg_dict = get_curobo_cfg(name)["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(cfg_dict)
    return CudaRobotModel(robot_cfg.kinematics)


def get_ik_solver(
    name: str,
    world_cfg: WorldConfig,
    num_seeds: int = 12,
    self_collision_opt: bool = False,
    self_collision_check: bool = True,
    use_particle_opt: bool = False,
) -> IKSolver:
    """Get a cuRobo IK solver for a registered robot."""
    cfg_dict = get_curobo_cfg(name)["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(cfg_dict)
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        num_seeds=num_seeds,
        self_collision_opt=self_collision_opt,
        self_collision_check=self_collision_check,
        use_particle_opt=use_particle_opt,
    )
    return IKSolver(ik_config)


# Import here to avoid circular dependency at module level
# (RobotContainer is defined in __init__.py which imports from robot modules)
def _get_robot_container_class():
    from cutamp.robots import RobotContainer
    return RobotContainer


def load_robot_container(name: str, tensor_args: TensorDeviceType) -> "RobotContainer":
    """Load a RobotContainer for a registered robot. Generic replacement for per-robot load_*_container functions."""
    defn = get_robot_definition(name)

    kin_model = get_kinematics_model(name)
    joint_limits = kin_model.kinematics_config.joint_limits.position
    expected_shape = (2, defn.n_arm_joints)
    assert joint_limits.shape == expected_shape, (
        f"Invalid joint limits shape for {name}: {joint_limits.shape}, expected {expected_shape}"
    )

    gripper_spheres = defn.gripper_spheres_fn(tensor_args)
    tool_from_ee = _build_tool_from_ee(defn, tensor_args)

    RobotContainer = _get_robot_container_class()
    return RobotContainer(name, kin_model, joint_limits, gripper_spheres, tool_from_ee)


def load_rerun_robot(name: str, load_mesh: bool = True) -> "RerunRobot":
    """Load a RerunRobot for visualization."""
    from cutamp.robots.utils import RerunRobot

    defn = get_robot_definition(name)
    rerun_robot = defn.rerun_fn(load_mesh)
    if not isinstance(rerun_robot, RerunRobot):
        raise TypeError(f"Expected RerunRobot, got {type(rerun_robot)}")
    return rerun_robot


def get_q_home(name: str) -> Sequence[float]:
    """Get the home joint positions for a registered robot."""
    defn = get_robot_definition(name)
    return defn.q_home
