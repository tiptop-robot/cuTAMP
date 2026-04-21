import logging
import math
import os
from pathlib import Path

import torch
from jaxtyping import Float
from yourdfpy import URDF

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import join_path, load_yaml, get_assets_path
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from cutamp.robots.registry import RobotDefinition, register_robot
from cutamp.robots.utils import RerunRobot, get_robotiq_2f_85_gripper_spheres

_log = logging.getLogger(__name__)

fr3_robotiq_neutral_joint_positions = (0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0)


def fr3_robotiq_curobo_cfg():
    assets_dir = Path(__file__).parent / "assets"
    cfg = load_yaml(str(assets_dir / "fr3_robotiq_2f_85.yml"))
    # Set some asset paths so cuRobo can load our URDF and meshes
    keys = ["external_asset_path", "external_robot_configs_path"]
    for key in keys:
        if key not in cfg["robot_cfg"]["kinematics"]:
            cfg["robot_cfg"]["kinematics"][key] = str(assets_dir)
    return cfg


def _fr3_robotiq_cfg_dict() -> dict:
    return fr3_robotiq_curobo_cfg()["robot_cfg"]


def get_fr3_robotiq_kinematics_model() -> CudaRobotModel:
    """cuRobo robot kinematics model."""
    robot_cfg = _fr3_robotiq_cfg_dict()
    robot_cfg = RobotConfig.from_dict(robot_cfg)
    kinematics_model = CudaRobotModel(robot_cfg.kinematics)
    return kinematics_model


def get_fr3_robotiq_ik_solver(
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
    robot_cfg_dict = _fr3_robotiq_cfg_dict()
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


def get_fr3_robotiq_gripper_spheres(
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> Float[torch.Tensor, "num_spheres 4"]:
    """
    Get the collision spheres for the Franka gripper.
    IMPORTANT: note they are in the origin frame with z-up (not the conventional z-down gripper frame).
    """
    return get_robotiq_2f_85_gripper_spheres(tensor_args)


def load_fr3_robotiq_rerun(load_mesh: bool = True) -> RerunRobot:
    robot_cfg = fr3_robotiq_curobo_cfg()["robot_cfg"]
    urdf_rel_path = robot_cfg["kinematics"]["urdf_path"]
    urdf_path = os.path.join(robot_cfg["kinematics"].get("external_asset_path", ""), urdf_rel_path) or join_path(
        get_assets_path(), urdf_rel_path
    )

    def _locate_curobo_asset(fname: str) -> str:
        if fname.startswith("package://"):
            return os.path.join(get_assets_path(), "robot", fname.replace("package://", ""))

        assets_dir = Path(__file__).parent
        return os.path.join(assets_dir, fname)

    urdf = URDF.load(urdf_path, filename_handler=_locate_curobo_asset)
    return RerunRobot("panda", urdf, q_neutral=(*fr3_robotiq_neutral_joint_positions, 0.04), load_mesh=load_mesh)


panda_robotiq_neutral_joint_positions = (0.0, -0.628, 0.0, -2.513, 0.0, 1.885, 0.0)


def panda_robotiq_curobo_cfg():
    assets_dir = Path(__file__).parent / "assets"
    cfg = load_yaml(str(assets_dir / "panda_robotiq_2f_85.yml"))
    keys = ["external_asset_path", "external_robot_configs_path"]
    for key in keys:
        if key not in cfg["robot_cfg"]["kinematics"]:
            cfg["robot_cfg"]["kinematics"][key] = str(assets_dir)
    return cfg


def _panda_robotiq_cfg_dict() -> dict:
    return panda_robotiq_curobo_cfg()["robot_cfg"]


def get_panda_robotiq_kinematics_model() -> CudaRobotModel:
    """cuRobo robot kinematics model."""
    robot_cfg = RobotConfig.from_dict(_panda_robotiq_cfg_dict())
    return CudaRobotModel(robot_cfg.kinematics)


def get_panda_robotiq_ik_solver(
    world_cfg: WorldConfig,
    num_seeds: int = 12,
    self_collision_opt: bool = False,
    self_collision_check: bool = True,
    use_particle_opt: bool = False,
) -> IKSolver:
    """cuRobo IK solver for Panda with Robotiq 2F-85 gripper."""
    robot_cfg = RobotConfig.from_dict(_panda_robotiq_cfg_dict())
    ik_config = IKSolverConfig.load_from_robot_config(
        robot_cfg,
        world_cfg,
        num_seeds=num_seeds,
        self_collision_opt=self_collision_opt,
        self_collision_check=self_collision_check,
        use_particle_opt=use_particle_opt,
    )
    return IKSolver(ik_config)


def get_panda_robotiq_gripper_spheres(
    tensor_args: TensorDeviceType = TensorDeviceType(),
) -> Float[torch.Tensor, "num_spheres 4"]:
    """
    Get the collision spheres for the Robotiq 2F-85 gripper.
    IMPORTANT: note they are in the origin frame with z-up (not the conventional z-down gripper frame).
    """
    return get_robotiq_2f_85_gripper_spheres(tensor_args)


def load_panda_robotiq_rerun(load_mesh: bool = True) -> RerunRobot:
    robot_cfg = panda_robotiq_curobo_cfg()["robot_cfg"]
    urdf_rel_path = robot_cfg["kinematics"]["urdf_path"]
    assets_dir = Path(__file__).parent / "assets"
    urdf_path = str(assets_dir / urdf_rel_path)

    def _locate_asset(fname: str) -> str:
        if fname.startswith("package://"):
            return os.path.join(get_assets_path(), "robot", fname.replace("package://", ""))
        # Panda arm meshes (e.g. "meshes/visual/link0.dae") live in cuRobo's franka_description
        if fname.startswith("meshes/"):
            return os.path.join(get_assets_path(), "robot", "franka_description", fname)
        # "assets/robotiq_description/..." and "assets/camera_mounts/..." are relative to the robots/ dir
        robots_dir = Path(__file__).parent
        return str(robots_dir / fname)

    urdf = URDF.load(urdf_path, filename_handler=_locate_asset)
    return RerunRobot("panda", urdf, q_neutral=(*panda_robotiq_neutral_joint_positions, 0.04), load_mesh=load_mesh)


## ---------------------------------------------------------------------------
## Robot registrations
## ---------------------------------------------------------------------------

register_robot(
    RobotDefinition(
        name="fr3_robotiq",
        curobo_cfg_fn=fr3_robotiq_curobo_cfg,
        gripper_spheres_fn=get_fr3_robotiq_gripper_spheres,
        rerun_fn=load_fr3_robotiq_rerun,
        q_home=fr3_robotiq_neutral_joint_positions,
        tool_from_ee_rpy=(math.pi, 0, math.pi / 2),
        tool_from_ee_translation=(0.0, 0.0, 0.015),
    )
)

register_robot(
    RobotDefinition(
        name="panda_robotiq",
        curobo_cfg_fn=panda_robotiq_curobo_cfg,
        gripper_spheres_fn=get_panda_robotiq_gripper_spheres,
        rerun_fn=load_panda_robotiq_rerun,
        q_home=panda_robotiq_neutral_joint_positions,
        tool_from_ee_rpy=(math.pi, 0, math.pi / 2),
        tool_from_ee_translation=(0.0, 0.0, 0.015),
    )
)


if __name__ == "__main__":
    import rerun as rr

    rr.init("fr3_robotiq_test", spawn=True)
    franka = load_fr3_robotiq_rerun()

    kin_model = get_fr3_robotiq_kinematics_model()

    q = torch.tensor(fr3_robotiq_neutral_joint_positions, device="cuda")[None]
    state = kin_model.get_state(q)

    spheres = state.link_spheres_tensor[0].cpu()
    xyz, radii = spheres[:, :3], spheres[:, 3]
    rr.log("spheres", rr.Points3D(positions=xyz, radii=radii))

    # Visualize gripper spheres
    gripper_spheres = get_fr3_robotiq_gripper_spheres().cpu()
    rr.log("gripper_spheres", rr.Points3D(positions=gripper_spheres[:, :3], radii=gripper_spheres[:, 3]))
