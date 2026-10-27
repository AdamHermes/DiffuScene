"""Inference-only physical energies for DiffuScene (no EchoScene imports).

Public loss functions take [B, N, 7] boxes: full sizes (XYZ), centers
(XYZ), yaw in radians, plus a [B, N] boolean furniture mask. Y is up.
Yaw follows DiffuScene's row-vector mesh transform: X'=X*cos+Z*sin.
Losses sum objects (or average pairs) within each scene, then average scenes.
"""

import json
import math

import torch


def _footprint_corners(boxes):
    signs = boxes.new_tensor([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    local = boxes[..., [0, 2]].unsqueeze(-2) * signs * 0.5
    u, v = local.unbind(-1)
    c, s = boxes[..., 6].cos().unsqueeze(-1), boxes[..., 6].sin().unsqueeze(-1)
    return torch.stack((u * c + v * s, -u * s + v * c), -1) + boxes[..., [3, 5]].unsqueeze(-2)


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _intersection_area(corners_a, corners_b):
    """Differentiable rectangle intersection from corners and edge crossings.

    Discrete inside/sort decisions are piecewise constant; coordinates retain
    their gradients. Safe denominators also cover parallel and disjoint boxes.
    """
    def inside(points, polygon):
        edges = polygon.roll(-1, -2) - polygon
        offsets = points.unsqueeze(-2) - polygon.unsqueeze(-3)
        return (_cross(edges.unsqueeze(-3), offsets) >= -1e-7).all(-1)

    a = corners_a.unsqueeze(-2)
    b = corners_b.unsqueeze(-3)
    ea = (corners_a.roll(-1, -2) - corners_a).unsqueeze(-2)
    eb = (corners_b.roll(-1, -2) - corners_b).unsqueeze(-3)
    denominator = _cross(ea, eb)
    nonparallel = denominator.abs() > 1e-8
    safe_denominator = torch.where(nonparallel, denominator, torch.ones_like(denominator))
    t = _cross(b - a, eb) / safe_denominator
    u = _cross(b - a, ea) / safe_denominator
    crossing = nonparallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
    intersections = a + t.unsqueeze(-1) * ea
    vertices = torch.cat((corners_a, corners_b, intersections.flatten(-3, -2)), -2)
    valid = torch.cat((inside(corners_a, corners_b), inside(corners_b, corners_a), crossing.flatten(-2)), -1)
    count = valid.sum(-1)
    centroid = (vertices * valid.unsqueeze(-1)).sum(-2) / count.clamp_min(1).unsqueeze(-1)
    offsets = vertices - centroid.unsqueeze(-2)
    # atan2(0, 0) is not needed for the discrete sorting decision.
    angles = torch.atan2(offsets[..., 1].detach(), offsets[..., 0].detach())
    order = angles.masked_fill(~valid, 10.0).argsort(-1)
    sorted_vertices = vertices.gather(-2, order.unsqueeze(-1).expand_as(vertices))
    positions = torch.arange(vertices.shape[-2], device=vertices.device)
    next_positions = (positions + 1) % count.clamp_min(1).unsqueeze(-1)
    following = sorted_vertices.gather(-2, next_positions.unsqueeze(-1).expand_as(vertices))
    area = (_cross(sorted_vertices, following) * (positions < count.unsqueeze(-1))).sum(-1).abs() * 0.5
    return torch.where(count >= 3, area, torch.zeros_like(area))


def compute_collision_guidance_loss(boxes, objectness, iou_weight=0.5,
                                    penetration_weight=2.0, iou_threshold=0.02,
                                    penetration_threshold=0.0):
    """OBB 3D IoU plus the rotated-AABB penetration proxy used by EchoScene."""
    n = boxes.shape[1]
    pairs = torch.triu_indices(n, n, offset=1, device=boxes.device)
    a, b = boxes[:, pairs[0]], boxes[:, pairs[1]]
    valid = objectness[:, pairs[0]] & objectness[:, pairs[1]]
    corners = _footprint_corners(boxes)
    area = _intersection_area(corners[:, pairs[0]], corners[:, pairs[1]])
    lower = torch.maximum(a[..., 4] - a[..., 1] / 2, b[..., 4] - b[..., 1] / 2)
    upper = torch.minimum(a[..., 4] + a[..., 1] / 2, b[..., 4] + b[..., 1] / 2)
    intersection = area * (upper - lower).clamp_min(0)
    union = a[..., :3].prod(-1) + b[..., :3].prod(-1) - intersection
    iou = (intersection / union.clamp_min(1e-8)).clamp(0, 1)

    sizes = boxes[..., :3]
    c, s = boxes[..., 6].cos().abs(), boxes[..., 6].sin().abs()
    enclosing_sizes = torch.stack((sizes[..., 0] * c + sizes[..., 2] * s,
                                   sizes[..., 1], sizes[..., 0] * s + sizes[..., 2] * c), -1)
    overlap = ((enclosing_sizes[:, pairs[0]] + enclosing_sizes[:, pairs[1]]) / 2
               - (a[..., 3:6] - b[..., 3:6]).abs()).clamp_min(0)
    penetration = overlap.prod(-1)
    penalty = iou_weight * (iou - iou_threshold).clamp_min(0)
    penalty = penalty + penetration_weight * (penetration - penetration_threshold).clamp_min(0)
    return ((penalty * valid).sum(-1) / valid.sum(-1).clamp_min(1)).mean()


def compute_room_boundary_loss(boxes, objectness, room_bounds):
    """L1 excess of rotated footprints outside [xmin, zmin, xmax, zmax]."""
    corners = _footprint_corners(boxes)
    lower, upper = corners.amin(-2), corners.amax(-2)
    penalty = (room_bounds[:, None, :2] - lower).clamp_min(0).sum(-1)
    penalty = penalty + (upper - room_bounds[:, None, 2:]).clamp_min(0).sum(-1)
    return (penalty * objectness).sum(-1).mean()


def compute_walkability_loss(boxes, objectness, room_bounds, type="center_penalty",
                             sigma=0.5, robot_hight_real=1.5, sigma_scale=0.5,
                             heatmap_weight=0.8, repulsion_weight=0.2, grid_resolution=64):
    """EchoScene center penalty or edge-Gaussian surrogate (not path connectivity).

    Center penalty uses the room's midpoint and masks empty slots. Edge Gaussian
    uses actual room bounds rather than a fixed 6m square. Like EchoScene, its
    height cutoff is based on the object's center Y coordinate.
    """
    centers = boxes[..., [3, 5]]
    if type == "center_penalty":
        midpoint = (room_bounds[:, :2] + room_bounds[:, 2:]) / 2
        distance_sq = (centers - midpoint[:, None]).square().sum(-1)
        return (torch.exp(-distance_sq / sigma) * objectness).sum(-1).mean()
    if type != "edge_gaussian":
        raise ValueError("walkable.type must be 'center_penalty' or 'edge_gaussian'")

    valid = objectness & (boxes[..., 4] < robot_hight_real)
    half_sizes = boxes[..., [0, 2]] / 2
    sigmas = boxes[..., [0, 2]].mean(-1) * sigma_scale
    c, s = boxes[..., 6].cos(), boxes[..., 6].sin()

    def field(delta):
        dx, dz = delta.unbind(-1)
        x_local = dx * c[:, :, None] - dz * s[:, :, None]
        z_local = dx * s[:, :, None] + dz * c[:, :, None]
        local = torch.stack((x_local, z_local), -1)
        distance_sq = (local.abs() - half_sizes[:, :, None]).clamp_min(0).square().sum(-1)
        return distance_sq

    # Object j's center in object i's local frame, with the same size-j proxy
    # as EchoScene. This term is a soft repulsion energy, not exact OBB distance.
    distance = (field(centers[:, None] - centers[:, :, None]) + 1e-8).sqrt()
    clearance = (distance - half_sizes.mean(-1)[:, None]).clamp_min(0)
    pairwise = torch.exp(-clearance.square() / (2 * sigmas[:, :, None].square() + 1e-8))
    pair_mask = valid[:, :, None] & valid[:, None, :]
    pair_mask = pair_mask & ~torch.eye(boxes.shape[1], device=boxes.device, dtype=torch.bool)
    repulsion = (pairwise * pair_mask).sum((-1, -2)) / 2

    unit = torch.linspace(0, 1, grid_resolution, device=boxes.device, dtype=boxes.dtype)
    gx, gz = torch.meshgrid(unit, unit, indexing="ij")
    grid = torch.stack((gx.flatten(), gz.flatten()), -1)
    grid = room_bounds[:, None, :2] + grid * (room_bounds[:, None, 2:] - room_bounds[:, None, :2])
    distances_sq = field(grid[:, None] - centers[:, :, None])
    heatmap = torch.exp(-distances_sq / (2 * sigmas[:, :, None].square() + 1e-8))
    heatmap = (heatmap * valid[:, :, None]).sum(1).mean(-1)
    return (heatmap_weight * heatmap + repulsion_weight * repulsion).mean()


class PhysicalGuidance:
    """Differentiate physical energies only w.r.t. translations and yaw."""

    def __init__(self, config, network_config, train_stats_file):
        self.config = config
        self.angle_dim = network_config.get("angle_dim", 1)
        self.bbox_dim = 6 + self.angle_dim
        self.class_dim = network_config.get("class_dim", 21)
        self.objectness_dim = network_config.get("objectness_dim", 1)
        if network_config.get("translation_dim", 3) != 3 or network_config.get("size_dim", 3) != 3:
            raise ValueError("Physical guidance requires 3D translations and sizes")
        if self.angle_dim not in (1, 2):
            raise ValueError("Physical guidance requires angle_dim 1 or 2")
        if not train_stats_file:
            raise ValueError("Physical guidance requires diffusion_kwargs.train_stats_file")
        with open(train_stats_file, "r") as handle:
            self.stats = json.load(handle)
        self.constraints = config.get("constraints", {})
        unknown = set(self.constraints) - {"collision", "room_boundary", "walkable"}
        if unknown:
            raise ValueError("Unknown physical guidance constraints: {}".format(sorted(unknown)))
        for key in ("strength", "grad_clip"):
            if not math.isfinite(float(config.get(key, 0))) or float(config.get(key, 0)) < 0:
                raise ValueError("guidance.{} must be finite and nonnegative".format(key))
        for name, cfg in self.constraints.items():
            weight = float(cfg.get("weight", 0))
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("{}.weight must be finite and nonnegative".format(name))
            start = float(cfg.get("start_ratio", config.get("start_ratio", 0.95)))
            if not 0 <= start <= 1 or int(cfg.get("interval", config.get("interval", 1))) < 1:
                raise ValueError("Invalid guidance start_ratio or interval")
        walkable = self.constraints.get("walkable", {})
        if walkable.get("type", "center_penalty") not in ("center_penalty", "edge_gaussian"):
            raise ValueError("Unknown walkable.type")
        for key, default in (("sigma", 0.5), ("sigma_scale", 0.5)):
            if float(walkable.get(key, default)) <= 0:
                raise ValueError("walkable.{} must be positive".format(key))
        if int(walkable.get("grid_resolution", 64)) < 2:
            raise ValueError("walkable.grid_resolution must be >= 2")
        self.step_stats = []

    def active_constraints(self, timestep, num_timesteps):
        step = num_timesteps - 1 - timestep
        progress = step / max(num_timesteps - 1, 1)
        return {name: cfg for name, cfg in self.constraints.items()
                if cfg.get("enabled", True) and float(cfg.get("weight", 0)) > 0
                and progress >= float(cfg.get("start_ratio", self.config.get("start_ratio", 0.95)))
                and step % int(cfg.get("interval", self.config.get("interval", 1))) == 0}

    def decode_boxes(self, prediction):
        def descale(values, key):
            bounds = values.new_tensor(self.stats[key])
            return (values + 1) / 2 * (bounds[3:] - bounds[:3]) + bounds[:3]

        centers = descale(prediction[..., :3], "bounds_translations")
        sizes = 2 * descale(prediction[..., 3:6], "bounds_sizes").clamp_min(1e-4)
        if self.angle_dim == 2:
            cos, sin = prediction[..., 6], prediction[..., 7]
            usable = (cos.square() + sin.square()) > 1e-12
            angles = torch.atan2(torch.where(usable, sin, torch.zeros_like(sin)),
                                 torch.where(usable, cos, torch.ones_like(cos)))
        else:
            low, high = self.stats["bounds_angles"]
            angles = (prediction[..., 6] + 1) / 2 * (high - low) + low
        return torch.cat((sizes, centers, angles.unsqueeze(-1)), -1)

    def object_mask(self, prediction):
        if prediction.shape[-1] == self.bbox_dim:
            return torch.ones(prediction.shape[:2], dtype=torch.bool, device=prediction.device)
        if self.objectness_dim:
            return prediction[..., self.bbox_dim + self.class_dim] >= 0
        # The final class channel is the signed empty indicator, as in the
        # repository's delete_empty_from_network_samples (zero means empty).
        return prediction[..., self.bbox_dim + self.class_dim - 1] < 0

    def room_bounds(self, prediction, room_bounds):
        if room_bounds is None:
            room_bounds = self.config.get("room_bounds")
        if room_bounds is None:
            raise ValueError("Room guidance requires room_bounds [xmin,zmin,xmax,zmax]; "
                             "generate_diffusion.py supplies bounds from the floor plan")
        bounds = torch.as_tensor(room_bounds, dtype=prediction.dtype, device=prediction.device).detach()
        if bounds.shape == (4,):
            bounds = bounds.unsqueeze(0)
        if bounds.ndim != 2 or bounds.shape[-1] != 4 or bounds.shape[0] not in (1, prediction.shape[0]):
            raise ValueError("room_bounds must have shape [4], [1,4], or [B,4]")
        if not torch.isfinite(bounds).all() or not (bounds[:, 2:] > bounds[:, :2]).all():
            raise ValueError("room_bounds must be finite with max > min")
        return bounds.expand(prediction.shape[0], -1)

    def gradient(self, prediction, timestep, num_timesteps, room_bounds=None):
        active = self.active_constraints(timestep, num_timesteps)
        if not active or float(self.config.get("strength", 20)) == 0:
            return None
        with torch.enable_grad():
            x0 = prediction.detach().requires_grad_(True)
            # Sizes and all semantic/shape channels are constants for guidance.
            pose_only = self.preserve_baseline(x0, x0.detach())
            boxes = self.decode_boxes(pose_only)
            mask = self.object_mask(x0.detach())
            bounds = self.room_bounds(x0, room_bounds) if set(active) - {"collision"} else None
            losses = {}
            for name, cfg in active.items():
                options = {key: value for key, value in cfg.items()
                           if key not in ("enabled", "weight", "interval", "start_ratio")}
                if name == "collision":
                    losses[name] = compute_collision_guidance_loss(boxes, mask, **options)
                elif name == "room_boundary":
                    losses[name] = compute_room_boundary_loss(boxes, mask, bounds, **options)
                else:
                    losses[name] = compute_walkability_loss(boxes, mask, bounds, **options)
            total = sum(losses[name] * float(active[name]["weight"]) for name in losses)
            if not torch.isfinite(total):
                raise FloatingPointError("Nonfinite physical guidance loss at t={}".format(timestep))
            gradient, = torch.autograd.grad(total, x0)
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite physical guidance gradient at t={}".format(timestep))
        norms = gradient.norm(dim=-1, keepdim=True)
        clip = float(self.config.get("grad_clip", 10.1))
        if clip > 0:
            gradient = gradient * (clip / norms.clamp_min(1e-12)).clamp_max(1)
        self.step_stats.append(dict(timestep=int(timestep), total_loss=float(total.detach()),
                                    grad_norm_max=float(norms.max()),
                                    **{name: float(value.detach()) for name, value in losses.items()}))
        return gradient.detach()

    def preserve_baseline(self, prediction, baseline):
        """Keep only translations/yaw from prediction; freeze all other channels."""
        return torch.cat((prediction[..., :3], baseline[..., 3:6],
                          prediction[..., 6:self.bbox_dim],
                          baseline[..., self.bbox_dim:]), dim=-1)
