"""
Script to resolve bounding box collisions for DiffuScene scenes in:
bedroom, diningroom, and livingroom.

Reads:
  - {room}/collision_params_merged.json
  - 3d_front_processed/{category}_objfeats_32_64/*/boxes.npz  (for room bounds)
Writes:
  - {room}/collision_params_resolved.json

Key details:
  1. Box representation in resolve_collision.py: [l, h, w, cx, cy, cz].
     We concatenate sizes (dx, dy, dz) and translations (cx, cy, cz).
  2. Angles in DiffuScene JSON are stored in RADIANS.
     The OBB solver in resolve_collision.py expects DEGREES (converted via rad * 180 / pi).
  3. Returns updated translations [cx, cy, cz] and preserves all original JSON keys
     (scene_ids, class_labels, sizes, angles, and updated translations).
  4. Automatically computes and reports collision metrics before and after resolving.
  5. Room bounds are injected from floor_plan_vertices in boxes.npz so that
     the room containment step in the resolver is active.
"""

import json
import math
import os
import re
import sys
import numpy as np
import torch

# Import OBB collision resolution solver from resolve_collision.py
from resolve_collision import resolve_bbox_collisions_obb, _obb_sat_xz

# ── Room bounds lookup ────────────────────────────────────────────────────────

# Mapping from output folder name to 3d_front_processed subfolder
CATEGORY_TO_PROCESSED = {
    "bedroom":    "bedrooms_objfeats_32_64",
    "livingroom": "livingrooms_objfeats_32_64",
    "diningroom": "diningrooms_objfeats_32_64",
}

def _extract_base_id(scene_id):
    """
    Extract the room base ID from a DiffuScene scene_id.
    E.g. 'MasterBedroom-33296_0_000' -> 'MasterBedroom-33296'
         'LivingDiningRoom-2065_0_000' -> 'LivingDiningRoom-2065'
         'DiningRoom-15534_1_001' -> 'DiningRoom-15534'
    """
    # Pattern: RoomType-Number followed by _idx_idx
    m = re.match(r'^(.+?-\d+)_\d+_\d+$', scene_id)
    if m:
        return m.group(1)
    return scene_id


def _build_room_bounds_cache(category, base_dir):
    """
    Pre-scan the 3d_front_processed folder for this category and build a
    dict mapping base_id -> room_bounds tuple.
    This avoids repeated os.listdir calls for every scene.
    """
    processed_subdir = CATEGORY_TO_PROCESSED.get(category)
    if processed_subdir is None:
        return {}

    cat_dir = os.path.join(base_dir, "3d_front_processed", processed_subdir)
    if not os.path.isdir(cat_dir):
        print(f"  WARNING: Processed data dir not found: {cat_dir}")
        return {}

    cache = {}
    for d in os.listdir(cat_dir):
        boxes_path = os.path.join(cat_dir, d, "boxes.npz")
        if not os.path.isfile(boxes_path):
            continue

        # Folder name format: UUID_RoomType-Number
        # Extract the RoomType-Number part (everything after the UUID_)
        parts = d.split("_", 1)
        if len(parts) == 2:
            base_id = parts[1]  # e.g. "MasterBedroom-33296"
        else:
            base_id = d

        if base_id in cache:
            continue  # already loaded

        try:
            data = np.load(boxes_path, allow_pickle=True)
            if "floor_plan_vertices" not in data:
                continue
            vertices = data["floor_plan_vertices"]
            # vertices shape: (V, 3) with columns [x, y, z], y is up
            xs = vertices[:, 0]
            zs = vertices[:, 2]

            cxL = 0.0
            czL = 0.0
            hxL = float((xs.max() - xs.min()) / 2.0)
            hzL = float((zs.max() - zs.min()) / 2.0)
            # Axis-aligned room (floor plans are already in world coordinates)
            axL = np.array([1.0, 0.0])
            azL = np.array([0.0, 1.0])

            cache[base_id] = (cxL, czL, hxL, hzL, axL, azL)
        except Exception as e:
            print(f"  WARNING: Failed to load {boxes_path}: {e}")

    print(f"  Built room bounds cache: {len(cache)} rooms from {cat_dir}")
    return cache


# ── Collision stats ───────────────────────────────────────────────────────────

def compute_collision_stats(data):
    """
    Compute OBB collision rate (fraction of scenes with collisions and fraction of colliding pairs)
    using the exact SAT logic to evaluate resolution effectiveness.
    """
    total_scenes = len(data["scene_ids"])
    collided_scenes = 0
    total_pairs = 0
    collided_pairs = 0

    for i in range(total_scenes):
        sizes = np.array(data["sizes"][i], dtype=np.float32) * 2.0
        trans = np.array(data["translations"][i], dtype=np.float32)
        angles_rad = np.array(data["angles"][i], dtype=np.float32)
        angles_deg = angles_rad * (180.0 / np.pi)

        N = len(sizes)
        if N < 2:
            continue

        scene_has_col = False
        for a in range(N):
            for b in range(a + 1, N):
                # Vertical check
                hy_a = float(sizes[a, 1]) / 2.0
                cy_a = float(trans[a, 1])
                hy_b = float(sizes[b, 1]) / 2.0
                cy_b = float(trans[b, 1])

                if (cy_a + hy_a) <= (cy_b - hy_b) or (cy_b + hy_b) <= (cy_a - hy_a):
                    continue

                box_a = [sizes[a, 0], sizes[a, 1], sizes[a, 2], trans[a, 0], trans[a, 1], trans[a, 2]]
                box_b = [sizes[b, 0], sizes[b, 1], sizes[b, 2], trans[b, 0], trans[b, 1], trans[b, 2]]

                ang_a = float(angles_deg[a].reshape(-1)[0])
                ang_b = float(angles_deg[b].reshape(-1)[0])

                is_col, depth, _ = _obb_sat_xz(box_a, ang_a, box_b, ang_b)
                if is_col and depth > 1e-4:
                    scene_has_col = True
                    collided_pairs += 1
                total_pairs += 1

        if scene_has_col:
            collided_scenes += 1

    scene_rate = (collided_scenes / total_scenes) if total_scenes > 0 else 0.0
    pair_rate = (collided_pairs / total_pairs) if total_pairs > 0 else 0.0
    return scene_rate, pair_rate, collided_scenes, total_scenes, collided_pairs, total_pairs


# ── Per-room resolution ──────────────────────────────────────────────────────

def resolve_room(room_dir, category, base_dir, max_iter=500, push_eps=0.02):
    input_path = os.path.join(room_dir, "collision_params_merged.json")
    output_path = os.path.join(room_dir, "collision_params_resolved.json")

    if not os.path.exists(input_path):
        print(f"  [SKIP] File not found: {input_path}")
        return

    print(f"Loading {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Build room bounds cache from floor_plan_vertices
    bounds_cache = _build_room_bounds_cache(category, base_dir)

    # Initial stats
    sc_before, pr_before, cs_b, ts_b, cp_b, tp_b = compute_collision_stats(data)
    print(f"  Before resolution:")
    print(f"    Collided scenes: {cs_b}/{ts_b} ({sc_before * 100:.2f}%)")
    print(f"    Collided pairs:  {cp_b}/{tp_b} ({pr_before * 100:.2f}%)")

    num_scenes = len(data["scene_ids"])
    resolved_translations = []
    bounds_found = 0
    bounds_missed = 0

    for i in range(num_scenes):
        sizes = torch.tensor(data["sizes"][i], dtype=torch.float32) * 2.0
        trans = torch.tensor(data["translations"][i], dtype=torch.float32)
        angles_rad = torch.tensor(data["angles"][i], dtype=torch.float32)

        # Convert radians to degrees for resolve_bbox_collisions_obb
        angles_deg = angles_rad * (180.0 / np.pi)

        # Concatenate to [l, h, w, x, y, z]
        boxes = torch.cat([sizes, trans], dim=-1)

        # Objectness is not present in DiffuScene output
        objectness = None
        if "objectness" in data:
            objectness = torch.tensor(data["objectness"][i], dtype=torch.float32)

        class_labels = None
        if "class_labels" in data:
            class_labels = np.array(data["class_labels"][i], dtype=np.float32)

        # Look up room bounds from floor_plan_vertices
        scene_id = data["scene_ids"][i]
        base_id = _extract_base_id(scene_id)
        room_bounds = bounds_cache.get(base_id)
        if room_bounds is not None:
            bounds_found += 1
        else:
            bounds_missed += 1

        resolved_boxes = resolve_bbox_collisions_obb(
            boxes=boxes,
            angles_pred=angles_deg,
            objectness_mask=objectness,
            class_labels=class_labels,
            max_iter=max_iter,
            push_eps=push_eps,
            verbose=False,
            external_room_bounds=room_bounds,
        )

        # Extract updated translations [cx, cy, cz]
        res_trans = resolved_boxes[:, 3:].cpu().numpy().tolist()
        resolved_translations.append(res_trans)

    print(f"  Room bounds: found={bounds_found}, missed={bounds_missed}")

    # Update translations in data dict
    data["translations"] = resolved_translations

    # After stats
    sc_after, pr_after, cs_a, ts_a, cp_a, tp_a = compute_collision_stats(data)
    print(f"  After resolution:")
    print(f"    Collided scenes: {cs_a}/{ts_a} ({sc_after * 100:.2f}%)")
    print(f"    Collided pairs:  {cp_a}/{tp_a} ({pr_after * 100:.2f}%)")

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"  Successfully saved to: {output_path} ({file_size_kb:.1f} KB)\n")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    rooms = ["bedroom", "diningroom", "livingroom"]

    print("=" * 65)
    print("STARTING OBB COLLISION RESOLUTION FOR DIFFUSCENE ROOMS")
    print("  (with room bounds from floor_plan_vertices)")
    print("=" * 65)

    for room in rooms:
        room_dir = os.path.join(base_dir, room)
        print(f"\n[{room.upper()}] Processing folder: {room_dir}")
        resolve_room(room_dir, category=room, base_dir=base_dir)

    print("=" * 65)
    print("ALL ROOMS PROCESSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    main()
