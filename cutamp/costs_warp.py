"""Warp-accelerated sphere-to-sphere overlap computation.

Replaces the PyTorch implementation in costs.py with a fused GPU kernel that avoids
materializing the O(n1 * n2) pairwise intermediate tensor. See docs/profiling-analysis.md
for the motivation.
"""

import torch
import warp as wp

wp.config.quiet = True
wp.init()
wp.set_module_options({"fast_math": False})


@wp.kernel
def _sphere_overlap_fwd_kernel_1(
    spheres_1: wp.array(dtype=wp.float32, ndim=2),
    spheres_2: wp.array(dtype=wp.float32, ndim=2),
    activation_distance: wp.float32,
    n1: wp.int32,
    n2: wp.int32,
    partial_cost: wp.array(dtype=wp.float32, ndim=2),
    grad_spheres_1: wp.array(dtype=wp.float32, ndim=2),
):
    """Forward kernel 1: one thread per (batch_elem, sphere_1_idx).

    Loops over all spheres in set 2 to compute partial cost and gradient for spheres_1.
    spheres_1/spheres_2 are flattened to (flat_batch * n, 4).
    partial_cost is (flat_batch, n1). grad_spheres_1 is (flat_batch * n1, 4).
    """
    tid = wp.tid()
    flat_batch = partial_cost.shape[0]
    if tid >= flat_batch * n1:
        return

    batch_idx = tid / n1
    i = tid - batch_idx * n1

    # Read sphere 1
    s1_base = batch_idx * n1 + i
    c1x = spheres_1[s1_base, 0]
    c1y = spheres_1[s1_base, 1]
    c1z = spheres_1[s1_base, 2]
    r1 = spheres_1[s1_base, 3]

    cost_accum = float(0.0)
    gc1x = float(0.0)
    gc1y = float(0.0)
    gc1z = float(0.0)
    gr1 = float(0.0)

    s2_base_start = batch_idx * n2
    for j in range(n2):
        s2_idx = s2_base_start + j
        c2x = spheres_2[s2_idx, 0]
        c2y = spheres_2[s2_idx, 1]
        c2z = spheres_2[s2_idx, 2]
        r2 = spheres_2[s2_idx, 3]

        dx = c1x - c2x
        dy = c1y - c2y
        dz = c1z - c2z
        dist_sq = dx * dx + dy * dy + dz * dz

        # Early exit: if squared distance exceeds threshold, no overlap possible
        threshold = r1 + r2 + activation_distance
        if dist_sq >= threshold * threshold:
            continue

        dist = wp.sqrt(dist_sq + 1.0e-8)
        penetration = threshold - dist  # = r1 + r2 + act_dist - dist

        if penetration > 0.0:
            cost_accum += penetration
            # Gradient: d(pen)/d(c1) = -(c1 - c2) / dist
            inv_dist = 1.0 / dist
            gc1x -= dx * inv_dist
            gc1y -= dy * inv_dist
            gc1z -= dz * inv_dist
            gr1 += 1.0

    partial_cost[batch_idx, i] = cost_accum
    grad_spheres_1[tid, 0] = gc1x
    grad_spheres_1[tid, 1] = gc1y
    grad_spheres_1[tid, 2] = gc1z
    grad_spheres_1[tid, 3] = gr1


@wp.kernel
def _sphere_overlap_fwd_kernel_2(
    spheres_1: wp.array(dtype=wp.float32, ndim=2),
    spheres_2: wp.array(dtype=wp.float32, ndim=2),
    activation_distance: wp.float32,
    n1: wp.int32,
    n2: wp.int32,
    grad_spheres_2: wp.array(dtype=wp.float32, ndim=2),
):
    """Forward kernel 2: one thread per (batch_elem, sphere_2_idx).

    Loops over all spheres in set 1 to compute gradient for spheres_2.
    """
    tid = wp.tid()
    flat_batch = grad_spheres_2.shape[0] / n2
    if tid >= flat_batch * n2:
        return

    batch_idx = tid / n2
    j = tid - batch_idx * n2

    # Read sphere 2
    s2_base = batch_idx * n2 + j
    c2x = spheres_2[s2_base, 0]
    c2y = spheres_2[s2_base, 1]
    c2z = spheres_2[s2_base, 2]
    r2 = spheres_2[s2_base, 3]

    gc2x = float(0.0)
    gc2y = float(0.0)
    gc2z = float(0.0)
    gr2 = float(0.0)

    s1_base_start = batch_idx * n1
    for i in range(n1):
        s1_idx = s1_base_start + i
        c1x = spheres_1[s1_idx, 0]
        c1y = spheres_1[s1_idx, 1]
        c1z = spheres_1[s1_idx, 2]
        r1 = spheres_1[s1_idx, 3]

        dx = c1x - c2x
        dy = c1y - c2y
        dz = c1z - c2z
        dist_sq = dx * dx + dy * dy + dz * dz

        threshold = r1 + r2 + activation_distance
        if dist_sq >= threshold * threshold:
            continue

        dist = wp.sqrt(dist_sq + 1.0e-8)
        penetration = threshold - dist

        if penetration > 0.0:
            # Gradient: d(pen)/d(c2) = +(c1 - c2) / dist (opposite sign from c1)
            inv_dist = 1.0 / dist
            gc2x += dx * inv_dist
            gc2y += dy * inv_dist
            gc2z += dz * inv_dist
            gr2 += 1.0

    grad_spheres_2[tid, 0] = gc2x
    grad_spheres_2[tid, 1] = gc2y
    grad_spheres_2[tid, 2] = gc2z
    grad_spheres_2[tid, 3] = gr2


# --- Grouped AABB variants ---
# These kernels accept per-group AABBs for spheres_1 and per-batch AABBs for spheres_2.
# Each thread in kernel_1 checks its group's AABB against spheres_2's AABB before entering
# the inner loop. This allows skipping entire groups (e.g. robot links) that are far from
# the object, without boolean indexing overhead.


@wp.kernel
def _sphere_overlap_grouped_fwd_kernel_1(
    spheres_1: wp.array(dtype=wp.float32, ndim=2),
    spheres_2: wp.array(dtype=wp.float32, ndim=2),
    activation_distance: wp.float32,
    n1: wp.int32,
    n2: wp.int32,
    # Group AABB for spheres_1: (n_groups, 6) where each row is [min_x, min_y, min_z, max_x, max_y, max_z]
    group_aabb_1: wp.array(dtype=wp.float32, ndim=2),
    # Per-sphere group index for spheres_1: (n1,) mapping sphere index -> group index
    group_idx_1: wp.array(dtype=wp.int32, ndim=1),
    # Per-batch AABB for spheres_2: (flat_batch, 6)
    batch_aabb_2: wp.array(dtype=wp.float32, ndim=2),
    partial_cost: wp.array(dtype=wp.float32, ndim=2),
    grad_spheres_1: wp.array(dtype=wp.float32, ndim=2),
):
    """Forward kernel 1 with group-level AABB gating on spheres_1."""
    tid = wp.tid()
    flat_batch = partial_cost.shape[0]
    if tid >= flat_batch * n1:
        return

    batch_idx = tid / n1
    i = tid - batch_idx * n1

    # Check group AABB of this sphere against batch AABB of spheres_2
    gid = group_idx_1[i]
    g_min_x = group_aabb_1[gid, 0]
    g_min_y = group_aabb_1[gid, 1]
    g_min_z = group_aabb_1[gid, 2]
    g_max_x = group_aabb_1[gid, 3]
    g_max_y = group_aabb_1[gid, 4]
    g_max_z = group_aabb_1[gid, 5]

    b_min_x = batch_aabb_2[batch_idx, 0]
    b_min_y = batch_aabb_2[batch_idx, 1]
    b_min_z = batch_aabb_2[batch_idx, 2]
    b_max_x = batch_aabb_2[batch_idx, 3]
    b_max_y = batch_aabb_2[batch_idx, 4]
    b_max_z = batch_aabb_2[batch_idx, 5]

    # AABB overlap test (already expanded by activation_distance + max_radius on Python side)
    if g_min_x > b_max_x or g_max_x < b_min_x:
        partial_cost[batch_idx, i] = 0.0
        grad_spheres_1[tid, 0] = 0.0
        grad_spheres_1[tid, 1] = 0.0
        grad_spheres_1[tid, 2] = 0.0
        grad_spheres_1[tid, 3] = 0.0
        return
    if g_min_y > b_max_y or g_max_y < b_min_y:
        partial_cost[batch_idx, i] = 0.0
        grad_spheres_1[tid, 0] = 0.0
        grad_spheres_1[tid, 1] = 0.0
        grad_spheres_1[tid, 2] = 0.0
        grad_spheres_1[tid, 3] = 0.0
        return
    if g_min_z > b_max_z or g_max_z < b_min_z:
        partial_cost[batch_idx, i] = 0.0
        grad_spheres_1[tid, 0] = 0.0
        grad_spheres_1[tid, 1] = 0.0
        grad_spheres_1[tid, 2] = 0.0
        grad_spheres_1[tid, 3] = 0.0
        return

    # AABBs overlap — do full sphere-to-sphere computation
    s1_base = batch_idx * n1 + i
    c1x = spheres_1[s1_base, 0]
    c1y = spheres_1[s1_base, 1]
    c1z = spheres_1[s1_base, 2]
    r1 = spheres_1[s1_base, 3]

    cost_accum = float(0.0)
    gc1x = float(0.0)
    gc1y = float(0.0)
    gc1z = float(0.0)
    gr1 = float(0.0)

    s2_base_start = batch_idx * n2
    for j in range(n2):
        s2_idx = s2_base_start + j
        c2x = spheres_2[s2_idx, 0]
        c2y = spheres_2[s2_idx, 1]
        c2z = spheres_2[s2_idx, 2]
        r2 = spheres_2[s2_idx, 3]

        dx = c1x - c2x
        dy = c1y - c2y
        dz = c1z - c2z
        dist_sq = dx * dx + dy * dy + dz * dz

        threshold = r1 + r2 + activation_distance
        if dist_sq >= threshold * threshold:
            continue

        dist = wp.sqrt(dist_sq + 1.0e-8)
        penetration = threshold - dist

        if penetration > 0.0:
            cost_accum += penetration
            inv_dist = 1.0 / dist
            gc1x -= dx * inv_dist
            gc1y -= dy * inv_dist
            gc1z -= dz * inv_dist
            gr1 += 1.0

    partial_cost[batch_idx, i] = cost_accum
    grad_spheres_1[tid, 0] = gc1x
    grad_spheres_1[tid, 1] = gc1y
    grad_spheres_1[tid, 2] = gc1z
    grad_spheres_1[tid, 3] = gr1


@wp.kernel
def _sphere_overlap_grouped_fwd_kernel_2(
    spheres_1: wp.array(dtype=wp.float32, ndim=2),
    spheres_2: wp.array(dtype=wp.float32, ndim=2),
    activation_distance: wp.float32,
    n1: wp.int32,
    n2: wp.int32,
    group_aabb_1: wp.array(dtype=wp.float32, ndim=2),
    group_idx_1: wp.array(dtype=wp.int32, ndim=1),
    batch_aabb_2: wp.array(dtype=wp.float32, ndim=2),
    grad_spheres_2: wp.array(dtype=wp.float32, ndim=2),
):
    """Forward kernel 2 with group-level AABB gating. For each sphere_2 thread,
    skips sphere_1 groups whose AABBs don't overlap with this batch's spheres_2 AABB.

    Note: we gate at the sphere_1 group level inside the inner loop.
    """
    tid = wp.tid()
    flat_batch = grad_spheres_2.shape[0] / n2
    if tid >= flat_batch * n2:
        return

    batch_idx = tid / n2
    j = tid - batch_idx * n2

    s2_base = batch_idx * n2 + j
    c2x = spheres_2[s2_base, 0]
    c2y = spheres_2[s2_base, 1]
    c2z = spheres_2[s2_base, 2]
    r2 = spheres_2[s2_base, 3]

    gc2x = float(0.0)
    gc2y = float(0.0)
    gc2z = float(0.0)
    gr2 = float(0.0)

    # Get this batch element's AABB for spheres_2
    b_min_x = batch_aabb_2[batch_idx, 0]
    b_min_y = batch_aabb_2[batch_idx, 1]
    b_min_z = batch_aabb_2[batch_idx, 2]
    b_max_x = batch_aabb_2[batch_idx, 3]
    b_max_y = batch_aabb_2[batch_idx, 4]
    b_max_z = batch_aabb_2[batch_idx, 5]

    s1_base_start = batch_idx * n1
    for i in range(n1):
        # Check if this sphere_1's group AABB overlaps with spheres_2's batch AABB
        gid = group_idx_1[i]
        if group_aabb_1[gid, 0] > b_max_x or group_aabb_1[gid, 3] < b_min_x:
            continue
        if group_aabb_1[gid, 1] > b_max_y or group_aabb_1[gid, 4] < b_min_y:
            continue
        if group_aabb_1[gid, 2] > b_max_z or group_aabb_1[gid, 5] < b_min_z:
            continue

        s1_idx = s1_base_start + i
        c1x = spheres_1[s1_idx, 0]
        c1y = spheres_1[s1_idx, 1]
        c1z = spheres_1[s1_idx, 2]
        r1 = spheres_1[s1_idx, 3]

        dx = c1x - c2x
        dy = c1y - c2y
        dz = c1z - c2z
        dist_sq = dx * dx + dy * dy + dz * dz

        threshold = r1 + r2 + activation_distance
        if dist_sq >= threshold * threshold:
            continue

        dist = wp.sqrt(dist_sq + 1.0e-8)
        penetration = threshold - dist

        if penetration > 0.0:
            inv_dist = 1.0 / dist
            gc2x += dx * inv_dist
            gc2y += dy * inv_dist
            gc2z += dz * inv_dist
            gr2 += 1.0

    grad_spheres_2[tid, 0] = gc2x
    grad_spheres_2[tid, 1] = gc2y
    grad_spheres_2[tid, 2] = gc2z
    grad_spheres_2[tid, 3] = gr2


class SphereOverlapWarp(torch.autograd.Function):
    """Warp-accelerated sphere-to-sphere overlap with analytical gradients."""

    @staticmethod
    def forward(ctx, spheres_1, spheres_2, activation_distance):
        # spheres_1: (*batch, n1, 4), spheres_2: (*batch, n2, 4)
        batch_shape = spheres_1.shape[:-2]
        n1 = spheres_1.shape[-2]
        n2 = spheres_2.shape[-2]
        flat_batch = 1
        for s in batch_shape:
            flat_batch *= s

        device = spheres_1.device
        torch_stream = torch.cuda.current_stream(device)
        stream = wp.stream_from_torch(torch_stream)

        # Flatten to (flat_batch * n, 4) for warp — reshape is a view, no copy
        s1_flat = spheres_1.detach().reshape(flat_batch * n1, 4)
        s2_flat = spheres_2.detach().reshape(flat_batch * n2, 4)

        # Allocate outputs
        partial_cost = torch.zeros(flat_batch, n1, device=device, dtype=torch.float32)
        grad_s1 = torch.zeros(flat_batch * n1, 4, device=device, dtype=torch.float32)
        grad_s2 = torch.zeros(flat_batch * n2, 4, device=device, dtype=torch.float32)

        act_dist = 0.0 if activation_distance is None else float(activation_distance)

        # Kernel 1: cost + grad_spheres_1
        wp.launch(
            kernel=_sphere_overlap_fwd_kernel_1,
            dim=flat_batch * n1,
            inputs=[
                wp.from_torch(s1_flat, dtype=wp.float32),
                wp.from_torch(s2_flat, dtype=wp.float32),
                act_dist,
                n1,
                n2,
                wp.from_torch(partial_cost, dtype=wp.float32),
                wp.from_torch(grad_s1, dtype=wp.float32),
            ],
            stream=stream,
        )

        # Kernel 2: grad_spheres_2
        wp.launch(
            kernel=_sphere_overlap_fwd_kernel_2,
            dim=flat_batch * n2,
            inputs=[
                wp.from_torch(s1_flat, dtype=wp.float32),
                wp.from_torch(s2_flat, dtype=wp.float32),
                act_dist,
                n1,
                n2,
                wp.from_torch(grad_s2, dtype=wp.float32),
            ],
            stream=stream,
        )

        # Sum partial costs across n1 to get per-batch-element cost
        cost = partial_cost.sum(dim=-1)  # (flat_batch,)
        cost = cost.reshape(batch_shape)

        # Reshape gradients back to input shapes
        grad_s1 = grad_s1.reshape(*batch_shape, n1, 4)
        grad_s2 = grad_s2.reshape(*batch_shape, n2, 4)

        ctx.save_for_backward(grad_s1, grad_s2)
        return cost

    @staticmethod
    def backward(ctx, grad_output):
        grad_s1_stored, grad_s2_stored = ctx.saved_tensors
        # grad_output: (*batch,) -> expand to (*batch, n, 4)
        grad_expand = grad_output[..., None, None]  # (*batch, 1, 1)
        grad_spheres_1 = grad_s1_stored * grad_expand if ctx.needs_input_grad[0] else None
        grad_spheres_2 = grad_s2_stored * grad_expand if ctx.needs_input_grad[1] else None
        return grad_spheres_1, grad_spheres_2, None


class SphereOverlapGroupedWarp(torch.autograd.Function):
    """Warp sphere-to-sphere overlap with per-group AABB gating on spheres_1.

    Used when spheres_1 has a known grouping (e.g. robot link spheres) and we can
    compute tight per-group AABBs to skip groups that are far from spheres_2.
    """

    @staticmethod
    def forward(ctx, spheres_1, spheres_2, activation_distance, group_aabb_1, group_idx_1, batch_aabb_2):
        batch_shape = spheres_1.shape[:-2]
        n1 = spheres_1.shape[-2]
        n2 = spheres_2.shape[-2]
        flat_batch = 1
        for s in batch_shape:
            flat_batch *= s

        device = spheres_1.device
        torch_stream = torch.cuda.current_stream(device)
        stream = wp.stream_from_torch(torch_stream)

        s1_flat = spheres_1.detach().reshape(flat_batch * n1, 4)
        s2_flat = spheres_2.detach().reshape(flat_batch * n2, 4)

        partial_cost = torch.zeros(flat_batch, n1, device=device, dtype=torch.float32)
        grad_s1 = torch.zeros(flat_batch * n1, 4, device=device, dtype=torch.float32)
        grad_s2 = torch.zeros(flat_batch * n2, 4, device=device, dtype=torch.float32)

        act_dist = 0.0 if activation_distance is None else float(activation_distance)

        # Flatten batch_aabb_2 from (*batch, 6) to (flat_batch, 6)
        batch_aabb_2_flat = batch_aabb_2.detach().reshape(flat_batch, 6)

        wp_s1 = wp.from_torch(s1_flat, dtype=wp.float32)
        wp_s2 = wp.from_torch(s2_flat, dtype=wp.float32)
        wp_group_aabb = wp.from_torch(group_aabb_1.detach(), dtype=wp.float32)
        wp_group_idx = wp.from_torch(group_idx_1.detach(), dtype=wp.int32)
        wp_batch_aabb = wp.from_torch(batch_aabb_2_flat, dtype=wp.float32)

        # Kernel 1: cost + grad_spheres_1 (with group AABB gating)
        wp.launch(
            kernel=_sphere_overlap_grouped_fwd_kernel_1,
            dim=flat_batch * n1,
            inputs=[
                wp_s1, wp_s2, act_dist, n1, n2,
                wp_group_aabb, wp_group_idx, wp_batch_aabb,
                wp.from_torch(partial_cost, dtype=wp.float32),
                wp.from_torch(grad_s1, dtype=wp.float32),
            ],
            stream=stream,
        )

        # Kernel 2: grad_spheres_2 (with group AABB gating in inner loop)
        wp.launch(
            kernel=_sphere_overlap_grouped_fwd_kernel_2,
            dim=flat_batch * n2,
            inputs=[
                wp_s1, wp_s2, act_dist, n1, n2,
                wp_group_aabb, wp_group_idx, wp_batch_aabb,
                wp.from_torch(grad_s2, dtype=wp.float32),
            ],
            stream=stream,
        )

        cost = partial_cost.sum(dim=-1).reshape(batch_shape)
        grad_s1 = grad_s1.reshape(*batch_shape, n1, 4)
        grad_s2 = grad_s2.reshape(*batch_shape, n2, 4)

        ctx.save_for_backward(grad_s1, grad_s2)
        return cost

    @staticmethod
    def backward(ctx, grad_output):
        grad_s1_stored, grad_s2_stored = ctx.saved_tensors
        grad_expand = grad_output[..., None, None]
        grad_spheres_1 = grad_s1_stored * grad_expand if ctx.needs_input_grad[0] else None
        grad_spheres_2 = grad_s2_stored * grad_expand if ctx.needs_input_grad[1] else None
        # No grad for activation_distance, group_aabb_1, group_idx_1, batch_aabb_2
        return grad_spheres_1, grad_spheres_2, None, None, None, None


def _sphere_to_sphere_overlap_pytorch(spheres_1, spheres_2, activation_distance):
    """PyTorch fallback for shapes that need broadcasting."""
    centers_1, radii_1 = spheres_1[..., :3], spheres_1[..., 3]
    centers_2, radii_2 = spheres_2[..., :3], spheres_2[..., 3]
    diff = centers_1.unsqueeze(-2) - centers_2.unsqueeze(-3)
    dist_sq = (diff * diff).sum(dim=-1)
    dist = torch.sqrt(dist_sq + 1e-8)
    radii_sum = radii_1.unsqueeze(-1) + radii_2.unsqueeze(-2)
    penetration = radii_sum - dist + activation_distance
    return torch.relu(penetration).sum((-2, -1))


def sphere_to_sphere_overlap_warp(
    spheres_1: torch.Tensor,
    spheres_2: torch.Tensor,
    activation_distance: float,
) -> torch.Tensor:
    """Drop-in replacement for _sphere_to_sphere_overlap using Warp kernels.

    Same signature and semantics: takes two sets of spheres (*batch, n, 4) and returns
    the total penetration (*batch,).

    Falls back to PyTorch when batch dims don't match (broadcasting needed).
    """
    # Warp kernel requires both inputs to have the same batch dims
    if spheres_1.shape[:-2] != spheres_2.shape[:-2]:
        return _sphere_to_sphere_overlap_pytorch(spheres_1, spheres_2, activation_distance)
    return SphereOverlapWarp.apply(spheres_1, spheres_2, activation_distance)


def sphere_to_sphere_overlap_grouped_warp(
    spheres_1: torch.Tensor,
    spheres_2: torch.Tensor,
    activation_distance: float,
    group_aabb_1: torch.Tensor,
    group_idx_1: torch.Tensor,
    batch_aabb_2: torch.Tensor,
) -> torch.Tensor:
    """Sphere-to-sphere overlap with per-group AABB gating on spheres_1.

    Args:
        spheres_1: (*batch, n1, 4) — grouped spheres (e.g. robot link spheres)
        spheres_2: (*batch, n2, 4) — target spheres (e.g. object spheres)
        activation_distance: collision activation distance
        group_aabb_1: (n_groups, 6) — per-group AABB [min_xyz, max_xyz], already expanded
            by activation_distance + max sphere radius in that group
        group_idx_1: (n1,) int32 — maps each sphere in spheres_1 to its group index
        batch_aabb_2: (*batch, 6) — per-batch-element AABB of spheres_2 [min_xyz, max_xyz],
            already expanded by max sphere radius
    """
    if spheres_1.shape[:-2] != spheres_2.shape[:-2]:
        return _sphere_to_sphere_overlap_pytorch(spheres_1, spheres_2, activation_distance)
    return SphereOverlapGroupedWarp.apply(
        spheres_1, spheres_2, activation_distance, group_aabb_1, group_idx_1, batch_aabb_2
    )
