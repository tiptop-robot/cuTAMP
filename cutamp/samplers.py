# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Optional

import torch
from curobo.geom.types import Obstacle, Cuboid, Mesh
from jaxtyping import Float

from cutamp.utils.common import approximate_goal_aabb, pose_list_to_mat4x4, transform_points
from cutamp.utils.obb import get_object_obb
from cutamp.utils.shapes import MultiSphere

Grasp4DOF = Place4DOF = Float[torch.Tensor, "n 4"]
Grasp6DOF = Place6DOF = Float[torch.Tensor, "n 6"]


def sample_yaw(num_samples: int, num_faces: Optional[int], device: torch.device):
    """Sample yaws. Continuous if num_faces is None, otherwise discrete."""
    if num_faces is None:
        yaw = torch.rand(num_samples, device=device) * 2 * torch.pi  # [0, 2pi)
    else:
        assert num_faces >= 1
        two_pi = 2 * torch.pi
        endpoint = two_pi - (two_pi / num_faces)
        yaw_choices = torch.linspace(0, endpoint, num_faces, device=device)
        yaw_idxs = torch.randint(0, num_faces, (num_samples,), device=device)
        yaw = yaw_choices[yaw_idxs]
    return yaw


def sample_stick_grasps(num_samples: int, stick: MultiSphere) -> Grasp4DOF:
    """Sample 4-DOF grasps for a stick."""
    spheres = stick.spheres
    if not (spheres[:, 1:3] == 0.0).all():
        raise ValueError(f"Expected stick spheres to have y and z positions of 0")

    # Randomly sample x-coordinate of the sphere
    sphere_x = spheres[:, 0]
    x_idxs = torch.randint(0, len(sphere_x), (num_samples,), device=spheres.device)
    sampled_x = sphere_x[x_idxs]

    # Sample yaws, use two faces since we just want original orientation and mirrored 180 degrees
    yaw = sample_yaw(num_samples, num_faces=2, device=spheres.device)

    # Create 4-DOF grasp!
    grasp_4dof = torch.zeros((num_samples, 4), device=spheres.device)
    grasp_4dof[:, 0] = sampled_x
    grasp_4dof[:, 3] = yaw
    return grasp_4dof


def grasp_4dof_sampler(
    num_samples: int, obj: Obstacle, obj_spheres: Float[torch.Tensor, "n 4"], num_faces: Optional[int] = None
) -> Grasp4DOF:
    """
    Sample 4-DOF grasps for the given object in the object's coordinate frame.
    This could be made a lot more sophisticated.
    """
    # Handle the stick as a special case
    if obj.name == "stick":
        obj: MultiSphere
        return sample_stick_grasps(num_samples, obj)

    # Determine point to grasp from the top of the object
    if isinstance(obj, Cuboid):
        obj_half_z = max(0.0, obj.dims[2] / 2 - 0.02)  # grasp 2cm from top of object
    elif isinstance(obj, MultiSphere):
        obj_half_z = 0.0
    else:
        max_z = obj_spheres[:, 2].max()
        obj_half_z = max_z - 0.02  # 2cm from top of object

    # Assume zero translation for now in x and y-axes
    translation = torch.zeros(num_samples, 3, device=obj.tensor_args.device)
    translation[:, 2] = obj_half_z

    # Sample yaw
    yaw = sample_yaw(num_samples, num_faces, obj.tensor_args.device)

    # Form full 4-DOF grasp
    grasp_4dof = torch.cat([translation, yaw.unsqueeze(-1)], dim=1)
    return grasp_4dof


def grasp_6dof_sampler(num_samples: int, obj: Obstacle, num_faces: Optional[int] = None) -> Grasp6DOF:
    """
    Sample 6-DOF grasps for the given object in the object's coordinate frame.
    Note: this is a very simple sampler which was written for the bookshelf domain and isn't general enough.
    """
    assert isinstance(obj, Cuboid), "only Cuboid objects supported for 6-dof grasps right now"
    # Sample roll from discrete choices
    roll_choices = torch.tensor(
        [-torch.pi / 2, -torch.pi / 3, -torch.pi / 4, torch.pi / 4, torch.pi / 3, torch.pi / 2],
        device=obj.tensor_args.device,
    )
    roll_idxs = torch.randint(0, len(roll_choices), (num_samples,), device=obj.tensor_args.device)
    roll = roll_choices[roll_idxs]

    # Let pitch be zero for now
    pitch = torch.zeros(num_samples, device=obj.tensor_args.device)

    # Sample yaw from discrete choices
    yaw_choices = torch.tensor([-torch.pi / 2, torch.pi / 2], device=obj.tensor_args.device)
    yaw_idxs = torch.randint(0, 2, (num_samples,), device=obj.tensor_args.device)
    yaw = yaw_choices[yaw_idxs]

    # Stack rpy
    rpy = torch.stack([roll, pitch, yaw], dim=1)

    # Compute offsets for gripper translation in object frame
    half_extents = obj.tensor_args.to_device([dim / 2 for dim in obj.dims])
    gripper_offset = 0.01
    upper = (half_extents - gripper_offset).clamp(min=0.0)
    lower = (obj.tensor_args.to_device(3 * [gripper_offset])).clamp(max=upper)
    lower[0] = upper[0] = 0.0  # remove translation in x-axis

    # Sample translation between bounds
    translation = torch.rand(num_samples, 3, device=obj.tensor_args.device)
    translation = lower + (upper - lower) * translation

    # Form 6-DOF grasps
    grasp_6dof = torch.cat([translation, rpy], dim=1)
    return grasp_6dof


def place_4dof_sampler(
    num_samples: int,
    obj: Obstacle,
    obj_spheres: Float[torch.Tensor, "n 4"],
    surface: Obstacle,
    surface_rep: str,
    shrink_dist: float | None,
    collision_activation_dist: float,
) -> Place4DOF:
    """Sample 4-DOF placement poses in the world frame. This does not yet fully support surfaces with yaw."""
    if not isinstance(surface, (Cuboid, Mesh)):
        raise NotImplementedError(f"Only Cuboid or Mesh surfaces supported for now, not {type(surface)}")

    device = obj.tensor_args.device

    # Compute object bottom offset (how much to raise object so bottom aligns with surface)
    sph_bottom = obj_spheres[:, 2] - obj_spheres[:, 3]
    obj_bottom = sph_bottom.min()
    obj_z_delta = -obj_bottom

    # Sample random z-offset above surface (to avoid exact contact)
    z_lower, z_upper = 1e-3, 1e-2
    z_offset = torch.rand(num_samples, 1, device=device)
    z_offset = z_lower + (z_upper - z_lower) * z_offset

    # Sample xyz based on surface representation in the world frame
    if surface_rep == "obb":
        obb = get_object_obb(surface, shrink_dist)

        # Compute maximum radial extent of object spheres (in object frame)
        # Since yaw is sampled, the object can be rotated - use radial distance for safety
        radial_distances = torch.sqrt(obj_spheres[:, 0] ** 2 + obj_spheres[:, 1] ** 2) + obj_spheres[:, 3]
        max_xy_extent = radial_distances.max()

        # Shrink OBB bounds to ensure object spheres stay within surface OBB
        # If object is too large, clamp to prevent negative bounds and use full OBB
        sampling_half_extents = (obb.half_extents[:2] - max_xy_extent).clamp(min=0.0)
        if (sampling_half_extents == 0.0).any():
            sampling_half_extents = obb.half_extents[:2]

        # Sample in OBB's local (axis-aligned) frame
        xy_local = torch.rand(num_samples, 2, device=device) * 2 - 1  # [-1, 1]
        xy_local = xy_local * sampling_half_extents

        # Transform xy from OBB local frame to world frame
        xyz_local = torch.cat([xy_local, torch.zeros(num_samples, 1, device=device)], dim=1)
        xy_world = xyz_local @ obb.rot_matrix.T  # rotate to world frame
        xy_world = xy_world[:, :2] + obb.center[:2]  # translate by OBB center

        # Compute z in world frame (obb.surface_z is already in world frame)
        z_world = z_offset + obj_z_delta + obb.surface_z + collision_activation_dist
        xyz_world = torch.cat([xy_world, z_world], dim=1)
    elif surface_rep == "aabb" and isinstance(surface, Cuboid):
        # AABB + Cuboid: sample in surface local frame, then transform to world
        assert shrink_dist is None
        aabb_xy_local = surface.tensor_args.to_device(
            [[-surface.dims[0] / 2, -surface.dims[1] / 2], [surface.dims[0] / 2, surface.dims[1] / 2]]
        )
        xy_local = torch.rand(num_samples, 2, device=device)
        xy_local = aabb_xy_local[0] + xy_local * (aabb_xy_local[1] - aabb_xy_local[0])

        # Compute z in local frame
        surface_z_local = surface.dims[2] / 2
        z_local = z_offset + obj_z_delta + surface_z_local + collision_activation_dist
        xyz_local = torch.cat([xy_local, z_local], dim=1)

        # Transform from surface local frame to world frame
        surface_mat4x4 = pose_list_to_mat4x4(surface.pose).to(device)
        xyz_world = transform_points(xyz_local, surface_mat4x4)
    elif surface_rep == "aabb" and isinstance(surface, Mesh):
        # AABB + Mesh: sample directly in world frame (approximate_goal_aabb returns world frame)
        assert shrink_dist is None
        aabb_world = approximate_goal_aabb(surface).to(device)
        aabb_xy_world = aabb_world[:, :2]
        surface_z_world = aabb_world[1, 2]

        # Sample in world frame
        xy_world = torch.rand(num_samples, 2, device=device)
        xy_world = aabb_xy_world[0] + xy_world * (aabb_xy_world[1] - aabb_xy_world[0])

        # Compute z in world frame
        z_world = z_offset + obj_z_delta + surface_z_world + collision_activation_dist
        xyz_world = torch.cat([xy_world, z_world], dim=1)
    else:
        raise ValueError(f"Unsupported combination: surface_rep={surface_rep}, surface type={type(surface)}")

    # Sample yaw and create final 4-DOF placement (xyz in world frame + yaw)
    yaw = sample_yaw(num_samples, None, device)
    place_4dof = torch.cat([xyz_world, yaw.unsqueeze(-1)], dim=1)
    return place_4dof
