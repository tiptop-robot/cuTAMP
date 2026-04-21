"""Tests for the robot registry and container loading."""

import pytest
import torch

from cutamp.robots.registry import (
    RobotDefinition,
    get_curobo_cfg,
    get_ik_solver,
    get_kinematics_model,
    get_q_home,
    get_robot_definition,
    list_robots,
    load_rerun_robot,
    load_robot_container,
)

gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires GPU")


def _dummy_world_cfg():
    """WorldConfig with a far-away cuboid so cuRobo's collision checker doesn't complain."""
    from curobo.geom.types import Cuboid, WorldConfig

    return WorldConfig(cuboid=[Cuboid(name="dummy", dims=[0.01, 0.01, 0.01], pose=[99, 99, 99, 1, 0, 0, 0])])

EXPECTED_ROBOTS = ["panda", "fr3_franka", "fr3_robotiq", "panda_robotiq", "ur5"]


def test_all_expected_robots_registered():
    registered = list_robots()
    for name in EXPECTED_ROBOTS:
        assert name in registered, f"Robot '{name}' not registered"


@pytest.mark.parametrize("robot_name", EXPECTED_ROBOTS)
def test_get_robot_definition(robot_name):
    defn = get_robot_definition(robot_name)
    assert isinstance(defn, RobotDefinition)
    assert defn.name == robot_name
    assert len(defn.q_home) == defn.n_arm_joints or len(defn.q_home) >= defn.n_arm_joints
    assert defn.tool_from_ee_rpy is not None or defn.tool_from_ee_quat is not None


@pytest.mark.parametrize("robot_name", EXPECTED_ROBOTS)
def test_get_q_home(robot_name):
    q_home = get_q_home(robot_name)
    defn = get_robot_definition(robot_name)
    assert len(q_home) >= defn.n_arm_joints


@pytest.mark.parametrize("robot_name", EXPECTED_ROBOTS)
def test_get_curobo_cfg(robot_name):
    cfg = get_curobo_cfg(robot_name)
    assert isinstance(cfg, dict)
    assert "robot_cfg" in cfg
    assert "kinematics" in cfg["robot_cfg"]


@gpu
@pytest.mark.parametrize("robot_name", EXPECTED_ROBOTS)
def test_load_robot_container(robot_name):
    from curobo.types.base import TensorDeviceType

    tensor_args = TensorDeviceType()
    container = load_robot_container(robot_name, tensor_args)
    defn = get_robot_definition(robot_name)

    assert container.name == robot_name
    assert container.joint_limits.shape == (2, defn.n_arm_joints)
    assert container.gripper_spheres.ndim == 2
    assert container.gripper_spheres.shape[1] == 4
    assert container.tool_from_ee.shape == (4, 4)

    # Verify tool_from_ee translation matches definition
    expected_t = list(defn.tool_from_ee_translation)
    actual_t = container.tool_from_ee[:3, 3].cpu().tolist()
    for e, a in zip(expected_t, actual_t):
        assert abs(e - a) < 1e-5, f"tool_from_ee translation mismatch for {robot_name}: {expected_t} vs {actual_t}"


@gpu
@pytest.mark.parametrize("robot_name", EXPECTED_ROBOTS)
def test_get_kinematics_model(robot_name):
    kin_model = get_kinematics_model(robot_name)
    defn = get_robot_definition(robot_name)

    q_home = torch.tensor(list(defn.q_home)[:defn.n_arm_joints], device="cuda", dtype=torch.float32)
    state = kin_model.get_state(q_home[None])
    ee_pos = state.ee_pose.position[0]
    assert ee_pos.shape == (3,)
    # EE should be at a reasonable position (not at origin or wildly far away)
    assert ee_pos.norm().item() > 0.1, f"EE position too close to origin for {robot_name}"
    assert ee_pos.norm().item() < 3.0, f"EE position unreasonably far for {robot_name}"


@gpu
@pytest.mark.parametrize("robot_name", EXPECTED_ROBOTS)
def test_fk_ik_roundtrip(robot_name):
    """FK at home config, then IK back - should recover a valid solution."""
    kin_model = get_kinematics_model(robot_name)
    defn = get_robot_definition(robot_name)
    q_home = torch.tensor(list(defn.q_home)[:defn.n_arm_joints], device="cuda", dtype=torch.float32)

    # FK
    state = kin_model.get_state(q_home[None])
    target_pose = state.ee_pose

    # IK
    ik_solver = get_ik_solver(robot_name, _dummy_world_cfg())
    result = ik_solver.solve_single(target_pose)
    assert result.success[0], f"IK failed for {robot_name} at home config"

    # Verify position error is small. result.solution has shape [batch, seeds, dof]
    ik_q = result.solution[0, 0]  # first batch, first seed
    ik_state = kin_model.get_state(ik_q[None])
    pos_err = (target_pose.position - ik_state.ee_pose.position).norm().item()
    assert pos_err < 0.01, f"IK position error too large for {robot_name}: {pos_err:.6f}m"


@gpu
@pytest.mark.parametrize("robot_name", EXPECTED_ROBOTS)
def test_backward_compat_container_loaders(robot_name):
    """Verify backward-compatible load_*_container functions still work."""
    from curobo.types.base import TensorDeviceType
    from cutamp.robots import (
        load_fr3_franka_container,
        load_fr3_robotiq_container,
        load_panda_container,
        load_panda_robotiq_container,
        load_ur5_container,
    )

    tensor_args = TensorDeviceType()
    compat_fns = {
        "panda": load_panda_container,
        "fr3_franka": load_fr3_franka_container,
        "fr3_robotiq": load_fr3_robotiq_container,
        "panda_robotiq": load_panda_robotiq_container,
        "ur5": load_ur5_container,
    }
    fn = compat_fns[robot_name]
    container = fn(tensor_args)
    assert container.name == robot_name


def test_unknown_robot_raises():
    with pytest.raises(ValueError, match="Unknown robot"):
        get_robot_definition("nonexistent_robot")

    with pytest.raises(ValueError, match="Unknown robot"):
        get_q_home("nonexistent_robot")
