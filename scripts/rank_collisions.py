"""Rank collision severity using rotated boxes from DiffuScene JSON.

    python rank_collisions.py --json-path /path/to/collision_params.json --size-scale 2

Standalone: only NumPy is required. No mesh, GPU, Torch or summary needed.
Sizes * size_scale gives full dimensions; translations unchanged; angles in radians.
Uses SAT on XZ and vertical overlap. Depth is horizontal box separation distance.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def box_geometry(size, center, angle):
    c, s = np.cos(angle), np.sin(angle)
    axes = np.array([[c, s], [-s, c]])
    offsets = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]) * size[[0, 2]] / 2
    return offsets @ axes + center[[0, 2]], axes


def collision_pairs(sizes, translations, angles, valid_indices, epsilon):
    boxes = {i: box_geometry(sizes[i], translations[i], angles[i]) for i in valid_indices}
    pairs = []
    for offset, i in enumerate(valid_indices):
        for j in valid_indices[offset + 1:]:
            y_overlap = min(translations[i, 1] + sizes[i, 1] / 2,
                            translations[j, 1] + sizes[j, 1] / 2) - max(
                                translations[i, 1] - sizes[i, 1] / 2,
                                translations[j, 1] - sizes[j, 1] / 2)
            if y_overlap <= epsilon:
                continue
            corners_i, axes_i = boxes[i]
            corners_j, axes_j = boxes[j]
            depth = float("inf")
            for axis in np.concatenate((axes_i, axes_j)):
                pi, pj = corners_i @ axis, corners_j @ axis
                # Directional separation distances handle contained boxes too.
                overlap = min(pi.max() - pj.min(), pj.max() - pi.min())
                if overlap <= epsilon:
                    break
                depth = min(depth, float(overlap))
            else:
                pairs.append({"i": i, "j": j, "depth": depth,
                              "y_overlap": float(y_overlap)})
    return sorted(pairs, key=lambda pair: pair["depth"], reverse=True)


def rank_json(source, size_scale=2.0, epsilon=1e-4):
    data = json.loads(source.read_text(encoding="utf-8"))
    for key in ("sizes", "translations", "angles"):
        if len(data[key]) != len(data["scene_ids"]):
            raise ValueError(f"{source}: {key} length does not match scene_ids")
    rows = []
    for index, scene_id in enumerate(data["scene_ids"]):
        sizes = np.asarray(data["sizes"][index], dtype=float).reshape(-1, 3) * size_scale
        translations = np.asarray(data["translations"][index], dtype=float).reshape(-1, 3)
        angles = np.asarray(data["angles"][index], dtype=float).reshape(-1)
        n = len(sizes)
        if translations.shape != sizes.shape or len(angles) != n:
            raise ValueError(f"{scene_id}: sizes, translations and angles must align")
        if not all(np.isfinite(a).all() for a in (sizes, translations, angles)) or np.any(sizes < 0):
            raise ValueError(f"{scene_id}: invalid box values")
        valid = list(range(n))
        if "objectness" in data and n:
            objectness = np.asarray(data["objectness"][index]).reshape(n, -1)[:, 0]
            valid = [i for i in valid if objectness[i] > 0]
        pairs = collision_pairs(sizes, translations, angles, valid, epsilon)
        if "class_labels" in data:
            for pair in pairs:
                pair["class_i"] = int(np.argmax(data["class_labels"][index][pair["i"]]))
                pair["class_j"] = int(np.argmax(data["class_labels"][index][pair["j"]]))
        collided = {pair[key] for pair in pairs for key in ("i", "j")}
        rows.append({
            "source_json": str(source.resolve()), "scene_index": index,
            "scene_id": scene_id, "object_count": len(valid),
            "collided_object_count": len(collided),
            "collision_rate": len(collided) / len(valid) if valid else 0.0,
            "pair_count": len(pairs),
            "max_depth": max((pair["depth"] for pair in pairs), default=0.0),
            "total_depth": sum(pair["depth"] for pair in pairs), "pairs": pairs,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--json-path", "--json_path", "--scene-json", type=Path,
                        help="Collision parameter JSON; no mesh files needed")
    inputs.add_argument("--folders", nargs="+", type=Path,
                        help="Room folders; defaults to bedroom diningroom livingroom")
    parser.add_argument("--stage", choices=("before", "after", "both"),
                        help="Folder mode: select merged/resolved JSON (default: both)")
    parser.add_argument("--size-scale", "--size_scale", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--sort", choices=("max_depth", "total_depth", "pair_count", "collided_object_count", "collision_rate"), default="max_depth")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("collision_ranking.json"))
    args = parser.parse_args()
    if not np.isfinite(args.size_scale) or args.size_scale <= 0:
        parser.error("--size-scale must be finite and positive")
    if not np.isfinite(args.epsilon) or args.epsilon < 0:
        parser.error("--epsilon must be finite and nonnegative")
    if args.top < 1:
        parser.error("--top must be positive")
    if args.json_path is not None:
        sources = [args.json_path]
    else:
        folders = args.folders or [Path(p) for p in ("bedroom", "diningroom", "livingroom")]
        stages = ("before", "after") if args.stage in (None, "both") else (args.stage,)
        sources = [folder / ("collision_params_merged.json" if stage == "before" else "collision_params_resolved.json")
                   for folder in folders for stage in stages]
    for source in sources:
        if not source.is_file():
            parser.error(f"JSON file not found: {source}")
        if source.resolve() == args.output.resolve():
            parser.error("--output must differ from input JSON")
    all_rows = []
    for source in sources:
        rows = rank_json(source, args.size_scale, args.epsilon)
        rows.sort(key=lambda row: (row[args.sort], row["total_depth"], row["pair_count"]), reverse=True)
        colliding = [row for row in rows if row["pair_count"]]
        print(f"\n{source} [size_scale={args.size_scale:g}]: {len(colliding)}/{len(rows)} scenes with collisions")
        print("scene_id | pairs | collided_objects | max_depth | total_depth")
        for row in colliding[:args.top]:
            print(f"{row['scene_id']} | {row['pair_count']} | {row['collided_object_count']}/{row['object_count']} | {row['max_depth']:.6f} | {row['total_depth']:.6f}")
        all_rows.extend(rows)
    report = {
        "method": "obb_sat_xz_with_vertical_overlap", "size_scale": args.size_scale,
        "epsilon": args.epsilon, "sort": args.sort,
        "notes": "sizes * size_scale gives full dimensions; translations unchanged; angles in radians. Skips objectness <= 0 when provided. No class exclusions. Depth is horizontal box separation distance, in input coordinate units. Object indices are zero-based. All scenes are saved; --top only limits console output.",
        "scenes": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull scene and object-pair details: {args.output.resolve()}")


if __name__ == "__main__":
    main()
