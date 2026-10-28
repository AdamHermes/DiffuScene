"""Rank individual scenes using the same OBB metrics as compute_collision_metrics.

Run from the repository root:
    python scripts/rank_collisions.py --folders bedroom diningroom livingroom
    python scripts/rank_collisions.py --json-path /path/to/collision_params.json

Depth is the horizontal SAT depth returned by the existing solver (meters
when source coordinates use meters), with a separate vertical overlap check.
This checks exported boxes, not mesh triangle intersections.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compute_collision_metrics import compute_pairwise_penetrations


def rank_folder(folder, stage):
    filename = "collision_params_merged.json" if stage == "before" else "collision_params_resolved.json"
    source = folder / filename
    mesh_dir = folder / ("scene_mesh" if stage == "before" else "scene_mesh_resolved")
    return rank_json(source, stage, mesh_dir)


def rank_json(source, stage="input", mesh_dir=None):
    if not source.is_file():
        raise FileNotFoundError(source)
    data = json.loads(source.read_text(encoding="utf-8"))
    folder = source.parent
    rows = []
    for index, scene_id in enumerate(data["scene_ids"]):
        sizes = np.asarray(data["sizes"][index], dtype=np.float32)
        trans = np.asarray(data["translations"][index], dtype=np.float32)
        angles = np.asarray(data["angles"][index], dtype=np.float32) * (180.0 / np.pi)
        pairs = compute_pairwise_penetrations(sizes, trans, angles)
        labels = data.get("class_labels")
        for pair in pairs:
            # Preserve original, zero-based object indices from the JSON.
            if labels is not None:
                pair["class_i"] = int(np.argmax(labels[index][pair["i"]]))
                pair["class_j"] = int(np.argmax(labels[index][pair["j"]]))
        pairs.sort(key=lambda pair: pair["depth"], reverse=True)
        mesh_path = mesh_dir / f"{scene_id}.ply" if mesh_dir is not None else None
        image_path = folder / f"{scene_id}.png"
        rows.append({
            "folder": str(folder.resolve()),
            "stage": stage,
            "source_json": str(source.resolve()),
            "scene_index": index,
            "scene_id": scene_id,
            "mesh_path": str(mesh_path.resolve()) if mesh_path is not None and mesh_path.is_file() else None,
            "image_path": str(image_path.resolve()) if image_path.is_file() and stage == "before" else None,
            "object_count": len(sizes),
            "pair_count": len(pairs),
            "max_depth": max((pair["depth"] for pair in pairs), default=0.0),
            "total_depth": sum(pair["depth"] for pair in pairs),
            "pairs": pairs,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--folders", nargs="+", type=Path, help="Room folders; defaults to bedroom diningroom livingroom")
    inputs.add_argument("--json-path", "--json_path", type=Path, help="Evaluate this JSON directly, with any filename")
    parser.add_argument("--mesh-dir", type=Path, help="Optional folder containing <scene_id>.ply; requires --json-path")
    parser.add_argument("--stage", choices=("before", "after", "both"), help="Folder mode: defaults to both. JSON mode: optional before/after label")
    parser.add_argument("--sort", choices=("max_depth", "total_depth", "pair_count"), default="max_depth")
    parser.add_argument("--top", type=int, default=10, help="Number of colliding scenes displayed per folder and stage")
    parser.add_argument("--output", type=Path, default=Path("collision_ranking.json"))
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top must be positive")
    if args.mesh_dir is not None and args.json_path is None:
        parser.error("--mesh-dir requires --json-path")
    if args.json_path is not None and args.stage == "both":
        parser.error("--json-path evaluates one file; --stage must be before or after, or omitted")
    if args.json_path is not None:
        jobs = [(args.json_path, args.stage or "input", args.mesh_dir)]
    else:
        folders = args.folders or [Path(p) for p in ("bedroom", "diningroom", "livingroom")]
        stages = ("before", "after") if args.stage in (None, "both") else (args.stage,)
        jobs = []
        for folder in folders:
            for stage in stages:
                filename = "collision_params_merged.json" if stage == "before" else "collision_params_resolved.json"
                mesh_dir = folder / ("scene_mesh" if stage == "before" else "scene_mesh_resolved")
                jobs.append((folder / filename, stage, mesh_dir))
    for source, _, _ in jobs:
        if not source.is_file():
            parser.error(f"JSON file not found: {source}")
    all_rows = []
    for source, stage, mesh_dir in jobs:
        rows = rank_json(source, stage, mesh_dir)
        rows.sort(key=lambda row: (row[args.sort], row["max_depth"], row["pair_count"]), reverse=True)
        colliding = [row for row in rows if row["pair_count"]]
        print(f"\n{source} [{stage}]: {len(colliding)}/{len(rows)} scenes with collisions")
        print("scene_id | pairs | max_depth | total_depth")
        for row in colliding[:args.top]:
            print(f"{row['scene_id']} | {row['pair_count']} | {row['max_depth']:.6f} | {row['total_depth']:.6f}")
            print(f"  mesh: {row['mesh_path'] or '(no matching exported mesh)'}")
        all_rows.extend(rows)
    report = {
        "method": "compute_collision_metrics.compute_pairwise_penetrations",
        "notes": "Uses sizes as stored, all objects, radians converted to degrees, vertical overlap and horizontal solver SAT depth > 1e-4; matches existing penetration report. Box approximation, not mesh intersections. Object indices are zero-based. Depth units follow input coordinates.",
        "sort": args.sort,
        "scenes": all_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nFull scene and object-pair details: {args.output.resolve()}")


if __name__ == "__main__":
    main()
