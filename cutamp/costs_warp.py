"""Warp-accelerated sphere-to-sphere overlap computation.

Replaces the PyTorch implementation in costs.py with a fused GPU kernel that avoids
materializing the O(n1 * n2) pairwise intermediate tensor. See docs/profiling-analysis.md
for the motivation.
"""

import torch
import warp as wp
from jaxtyping import Float

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


class _SphereOverlapWarp(torch.autograd.Function):
    """Warp-accelerated sphere-to-sphere overlap with analytical gradients.

    Inputs must have matching batch dims. The ``need_grad`` check uses ``requires_grad`` on the
    inputs (not ``torch.is_grad_enabled()``) because PyTorch disables autograd inside
    ``Function.forward``, so ``is_grad_enabled()`` is always False here.
    """

    @staticmethod
    def forward(
        ctx,
        spheres_1: Float[torch.Tensor, "*batch n1 4"],
        spheres_2: Float[torch.Tensor, "*batch n2 4"],
        activation_distance: float,
    ) -> Float[torch.Tensor, "*batch"]:
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
        act_dist = 0.0 if activation_distance is None else float(activation_distance)
        need_grad = spheres_1.requires_grad or spheres_2.requires_grad

        # Kernel 1: cost + grad_spheres_1 (grad_s1 is always written by the kernel as a byproduct)
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

        # Kernel 2: grad_spheres_2 (only needed when spheres_2 requires grad)
        grad_s2 = None
        if spheres_2.requires_grad:
            grad_s2 = torch.zeros(flat_batch * n2, 4, device=device, dtype=torch.float32)
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

        # Reshape and save gradients for backward
        if need_grad:
            grad_s1 = grad_s1.reshape(*batch_shape, n1, 4)
            grad_s2 = grad_s2.reshape(*batch_shape, n2, 4) if grad_s2 is not None else torch.zeros(*batch_shape, n2, 4, device=device)
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


def sphere_to_sphere_overlap_warp(
    spheres_1: Float[torch.Tensor, "*batch n1 4"],
    spheres_2: Float[torch.Tensor, "*batch n2 4"],
    activation_distance: float,
) -> Float[torch.Tensor, "*batch"]:
    """Warp-accelerated sphere-to-sphere overlap. Requires matching batch dims."""
    return _SphereOverlapWarp.apply(spheres_1, spheres_2, activation_distance)
