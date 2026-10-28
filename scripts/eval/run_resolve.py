"""Resolve furniture boxes exported by scripts/generate_diffusion.py on CPU.

Input/output sizes remain half extents and angles remain renderer yaw in radians.
Only X/Z translations change. This wrapper does not impose room containment.
"""

import argparse
import json
from pathlib import Path
import sys

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from resolve_collision import resolve_bbox_collisions_obb


def resolve_json(input_path, output_path, max_iter=500, push_eps=0.02):
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Use a separate output file to preserve the original layout")
    with input_path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    scene_ids = data["scene_ids"]
    if len(scene_ids) != len(set(scene_ids)):
        raise ValueError("Duplicate scene_ids in input JSON; use a clean generation output")
    for key in ("sizes", "translations", "angles"):
        if len(data[key]) != len(scene_ids):
            raise ValueError("Scene count mismatch for {}".format(key))

    print("Processing {} scenes on CPU".format(len(scene_ids)))
    for i, scene_id in enumerate(scene_ids):
        sizes = torch.tensor(data["sizes"][i], dtype=torch.float32).reshape(-1, 3)
        translations = torch.tensor(data["translations"][i], dtype=torch.float32).reshape(-1, 3)
        angles = torch.tensor(data["angles"][i], dtype=torch.float32).flatten()
        if sizes.shape != translations.shape or len(angles) != len(sizes):
            raise ValueError("Object count mismatch in {}".format(scene_id))
        if not all(torch.isfinite(value).all() for value in (sizes, translations, angles)):
            raise ValueError("Nonfinite geometry in {}".format(scene_id))
        if (sizes <= 0).any():
            raise ValueError("Nonpositive sizes in {}".format(scene_id))

        # DiffuScene exports half extents. The solver consumes full extents,
        # with positive yaw opposite to the renderer's row-vector transform.
        boxes = torch.cat((sizes * 2.0, translations), dim=-1)
        angles_deg = -torch.rad2deg(angles)
        print("Resolving {}/{}: {}".format(i + 1, len(scene_ids), scene_id))
        # Generator output already excludes empty slots and contains furniture
        # only: class 0 is furniture here, not EchoScene's layout/floor class.
        resolved = resolve_bbox_collisions_obb(
            boxes=boxes, angles_pred=angles_deg,
            max_iter=max_iter, push_eps=push_eps, verbose=True,
        )
        data["translations"][i] = resolved[:, 3:6].tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle)
    print("Saved resolved layout to {}".format(output_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_file", required=True, help="Clean DiffuScene collision_params JSON")
    parser.add_argument("--out_file", required=True, help="Separate resolved JSON output")
    parser.add_argument("--max_iter", type=int, default=500)
    parser.add_argument("--push_eps", type=float, default=0.02, help="Extra clearance in meters")
    args = parser.parse_args()
    if args.max_iter < 1 or not 0 <= args.push_eps < float("inf"):
        parser.error("max_iter must be positive and push_eps finite and nonnegative")
    resolve_json(args.in_file, args.out_file, args.max_iter, args.push_eps)


if __name__ == "__main__":
    main()
