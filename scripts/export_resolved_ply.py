"""Export resolved PLY scenes by moving the generator's world-space OBJ meshes.

Only translations may differ between original and resolved JSON. Existing OBJ
vertices already include size and yaw: do not multiply them by size_scale again.
Requires numpy and trimesh; no GPU or renderer is needed.
Default color source: original scene PLY vertex colors, matched before moving
objects (requires scipy). If unavailable, converts OBJ visuals to vertex colors.
"""

import argparse
import json
from pathlib import Path
import re

import numpy as np


def load_scenes(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    ids = data["scene_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate scene_ids in {path}")
    if len(data["translations"]) != len(ids):
        raise ValueError(f"Scene count mismatch in {path}")
    return {sid: {key: data[key][i] for key in ("translations", "sizes", "angles") if key in data}
            for i, sid in enumerate(ids)}


def translation_deltas(original, resolved, scene_id):
    before = np.asarray(original["translations"], dtype=float).reshape(-1, 3)
    after = np.asarray(resolved["translations"], dtype=float).reshape(-1, 3)
    if before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError(f"{scene_id}: invalid or mismatched translations")
    for key in ("sizes", "angles"):
        if key in original and key in resolved:
            a, b = np.asarray(original[key]), np.asarray(resolved[key])
            if a.shape != b.shape or not np.allclose(a, b):
                raise ValueError(f"{scene_id}: {key} changed; this exporter supports translation-only resolution")
    return after - before


def matched_vertex_colors(vertices, color_tree, source_colors, tolerance=1e-4):
    """Transfer RGB(A) at the original coordinates, independent of vertex order."""
    distances, indices = color_tree.query(vertices)
    if len(distances) and np.max(distances) > tolerance:
        raise ValueError(
            f"OBJ vertices do not match original PLY (max distance {np.max(distances):.6g}); "
            "verify both exports are from the same generation, or use --color-source obj"
        )
    return np.asarray(source_colors)[indices].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", "--output_dir", type=Path, required=True,
                        help="Generation folder containing JSONs and scene_mesh")
    parser.add_argument("--orig-json", "--orig_json", default="collision_params_merged.json")
    parser.add_argument("--resolved-json", "--resolved_json", default="collision_params_resolved.json")
    parser.add_argument("--scene-mesh-dir", "--scene_mesh_dir", default="scene_mesh")
    parser.add_argument("--export-dir", default="scene_mesh_resolved")
    parser.add_argument("--color-source", choices=("auto", "ply", "obj"), default="auto",
                        help="auto: original PLY colors if present, otherwise OBJ colors; ply: require original PLY colors")
    args = parser.parse_args()
    base = args.output_dir
    original = load_scenes(base / args.orig_json)
    resolved = load_scenes(base / args.resolved_json)
    mesh_root, output = base / args.scene_mesh_dir, base / args.export_dir
    if output.resolve() == mesh_root.resolve():
        parser.error("Export directory must differ from original scene_mesh")

    # Validate scene correspondence and transforms before exporting anything.
    deltas = {}
    for sid, scene in resolved.items():
        if sid not in original:
            raise ValueError(f"Resolved scene missing from original JSON: {sid}")
        deltas[sid] = translation_deltas(original[sid], scene, sid)

    import trimesh

    output.mkdir(parents=True, exist_ok=True)
    exported = skipped = 0
    for sid in resolved:
        obj_dir = mesh_root / f"{sid}_individual_objs"
        if not obj_dir.is_dir():
            obj_dir = mesh_root / sid
        paths = sorted(obj_dir.glob("*.obj"))
        if not paths:
            print(f"[SKIP] No OBJ files for {sid}: {obj_dir}")
            skipped += 1
            continue
        color_tree = source_colors = None
        original_ply = mesh_root / f"{sid}.ply"
        if args.color_source != "obj" and original_ply.is_file():
            from scipy.spatial import cKDTree

            colored = trimesh.load(str(original_ply), force="mesh", process=False)
            if colored.visual.kind in ("vertex", "face"):
                color_tree = cKDTree(colored.vertices)
                source_colors = np.asarray(colored.visual.vertex_colors)
        if args.color_source == "ply" and color_tree is None:
            raise ValueError(f"Original colored PLY required but unavailable: {original_ply}")
        print(f"[COLOR] {sid}: {'original PLY vertex colors' if color_tree is not None else 'OBJ visuals'}")
        meshes, seen = [], set()
        for path in paths:
            match = re.fullmatch(r"object_(\d+)", path.stem)
            if match:
                index = int(match.group(1))
                if index >= len(deltas[sid]):
                    print(f"[SKIP] {path.name}: index outside JSON object count")
                    continue
                if index in seen:
                    raise ValueError(f"{sid}: duplicate object index {index}")
                seen.add(index)
            elif path.stem.startswith("object_"):
                raise ValueError(f"Cannot parse object index: {path}")
            mesh = trimesh.load(str(path), force="mesh", process=False)
            # Bake colors before translating and concatenating. PLY carries RGB(A),
            # while OBJ textures/materials must first be converted to vertex colors.
            if color_tree is not None:
                colors = matched_vertex_colors(mesh.vertices, color_tree, source_colors)
            elif mesh.visual.kind == "texture":
                colors = mesh.visual.to_color().vertex_colors
            else:
                if mesh.visual.kind is None:
                    print(f"[WARNING] No explicit color in {path}; using default vertex colors")
                colors = mesh.visual.vertex_colors
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=colors)
            if match:
                mesh.apply_translation(deltas[sid][index])
            meshes.append(mesh)
        missing = set(range(len(deltas[sid]))) - seen
        if missing:
            raise ValueError(f"{sid}: missing OBJ files for object indices {sorted(missing)}")
        if not meshes:
            print(f"[SKIP] No matching objects for {sid}")
            skipped += 1
            continue
        out_path = output / f"{sid}.ply"
        trimesh.util.concatenate(meshes).export(str(out_path))
        exported += 1
        print(f"[EXPORTED] {out_path}")
    print(f"Done: {exported} exported, {skipped} skipped. Output: {output}")


if __name__ == "__main__":
    main()
