from functools import cached_property
from typing import Optional

import cv2
import numpy as np
import torch
from jaxtyping import Float

from curobo.geom.transform import quaternion_to_matrix, matrix_to_quaternion
from curobo.geom.types import Obstacle, Cuboid
from curobo.types.base import TensorDeviceType
from cutamp.utils.common import transform_points
from cutamp.utils.shapes import MultiSphere


class OrientedBoundingBox(torch.nn.Module):
    center: Float[torch.Tensor, "3"]
    half_extents: Float[torch.Tensor, "3"]
    quat_wxyz: Float[torch.Tensor, "4"]
    surface_z: float

    def __init__(
        self,
        center: Float[torch.Tensor, "3"],
        half_extents: Float[torch.Tensor, "3"],
        quat_wxyz: Float[torch.Tensor, "4"],
        surface_z: float,
    ):
        super().__init__()
        if not (center.ndim == half_extents.ndim == quat_wxyz.ndim):
            raise ValueError("All inputs must have the same number of dimensions")
        self.register_buffer("center", center)
        self.register_buffer("half_extents", half_extents)
        self.register_buffer("quat_wxyz", quat_wxyz)
        self.surface_z = surface_z

    def __repr__(self):
        return (
            f"OrientedBoundingBox(center.shape={self.center.shape}, half_extents.shape={self.half_extents.shape}, "
            f"quat_wxyz.shape={self.quat_wxyz.shape})"
        )

    @cached_property
    def rot_matrix(self) -> Float[torch.Tensor, "3 3"]:
        """Get the rotation matrix from OBB local frame to world frame (cached)."""
        return quaternion_to_matrix(self.quat_wxyz[None])[0]

    @cached_property
    def rot_matrix_inv(self) -> Float[torch.Tensor, "3 3"]:
        """Get the rotation matrix from world frame to OBB local frame (cached)."""
        return self.rot_matrix.T


def get_object_obb(obj: Obstacle, shrink_dist: Optional[float] = None) -> OrientedBoundingBox:
    """
    Get the oriented bounding box of an object in the world frame.
    Note: this method assumes that the object is planar with z-up.
    """
    if isinstance(obj, (Cuboid, MultiSphere)):
        obj = obj.get_mesh()
    tensor_args = TensorDeviceType()
    mat4x4 = torch.eye(4, device=tensor_args.device)
    mat4x4[:3, 3] = tensor_args.to_device(obj.pose[:3])
    mat4x4[:3, :3] = quaternion_to_matrix(tensor_args.to_device(obj.pose[3:])[None])[0]

    vertices_world = transform_points(points=tensor_args.to_device(obj.vertices), transform=mat4x4)

    # Fit a 2D rectangle to the xy vertices
    vertices_xy = vertices_world[:, :2].cpu().numpy().astype(np.float32)
    rect = cv2.minAreaRect(vertices_xy)
    box_points = cv2.boxPoints(rect)  # 4 corners in consistent winding order

    # Compute edge vectors explicitly to avoid angle convention issues
    edge0 = box_points[1] - box_points[0]
    edge1 = box_points[2] - box_points[1]
    len0, len1 = np.linalg.norm(edge0), np.linalg.norm(edge1)

    # Define local x-axis along edge0, y-axis along edge1
    x_len, y_len = len0, len1
    angle_rad = np.arctan2(edge0[1], edge0[0])

    cx, cy = rect[0]
    z_min, z_max = vertices_world[:, 2].min().item(), vertices_world[:, 2].max().item()
    z_len = z_max - z_min
    cz = (z_max + z_min) / 2.0

    center = tensor_args.to_device([cx, cy, cz])
    extents = tensor_args.to_device([x_len, y_len, z_len])

    # OBB rotation matrix (z-axis rotation)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    mat3x3 = np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    mat3x3 = tensor_args.to_device(mat3x3)

    quat_wxyz = matrix_to_quaternion(mat3x3[None])[0]
    surface_z = z_max

    half_extents = extents / 2
    if shrink_dist is not None:
        half_extents[:2] -= shrink_dist
    if (half_extents <= 0.0).any():
        raise ValueError(f"Shrunk OBB for {obj.name} has half extents <= 0 ({half_extents})")

    return OrientedBoundingBox(center=center, half_extents=half_extents, quat_wxyz=quat_wxyz, surface_z=surface_z)
