import argparse
import os
import sys
import json
import numpy as np
import trimesh
from scipy.spatial import cKDTree

def load_and_translate_obj(obj_path, delta_t, ply_tree, ply_colors):
    try:
        mesh = trimesh.load(obj_path, process=False, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                return None
                
        # Get vertex colors using KDTree
        if len(mesh.vertices) > 0:
            distances, indices = ply_tree.query(mesh.vertices)
            mesh_colors = ply_colors[indices]
            # Convert to [0, 255] rgba if needed
            if mesh_colors.shape[1] == 3:
                mesh_colors = np.hstack([mesh_colors, np.full((len(mesh_colors), 1), 255, dtype=mesh_colors.dtype)])
            
            # Apply per-vertex colors!
            mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=mesh_colors)
            
        # Apply transformation
        if delta_t is not None and not np.allclose(delta_t, 0.0):
            mesh.apply_translation(delta_t)
            
        return mesh
    except Exception as e:
        print(f"  WARNING: Cannot load {obj_path}: {e}")
        return None

def main(argv):
    parser = argparse.ArgumentParser(description="Export resolved scenes as GLB with per-vertex colors from PLY")
    parser.add_argument("--output_dir", required=True, help="Path to merge folder containing collision_params_resolved.json")
    parser.add_argument("--export_dir", default="resolved_glb_colored", help="Output folder name for glb files")
    
    args = parser.parse_args(argv)
    
    orig_json = os.path.join(args.output_dir, "collision_params.json")
    res_json = os.path.join(args.output_dir, "collision_params_resolved.json")
    scene_mesh_dir = os.path.join(args.output_dir, "scene_mesh")
    export_out_dir = os.path.join(args.output_dir, args.export_dir)
    
    os.makedirs(export_out_dir, exist_ok=True)
    
    with open(orig_json, 'r') as f:
        orig_data = json.load(f)
    with open(res_json, 'r') as f:
        res_data = json.load(f)
        
    scene_ids = orig_data.get("scene_ids", [])
    
    exported_count = 0
    print(f"Found {len(scene_ids)} scenes. Exporting colored GLB files...")
    
    for i, scene_id in enumerate(scene_ids):
        print(f"[{i+1}/{len(scene_ids)}] Exporting {scene_id}...")
        
        orig_trans = orig_data["translations"][i]
        res_trans = res_data["translations"][i]
        
        obj_dir = os.path.join(scene_mesh_dir, f"{scene_id}_individual_objs")
        ply_path = os.path.join(scene_mesh_dir, f"{scene_id}.ply")
        
        if not os.path.exists(obj_dir) or not os.path.exists(ply_path):
            print(f"  WARNING: Missing obj_dir or ply_path for {scene_id}")
            continue
            
        # Load PLY for colors
        try:
            ply_mesh = trimesh.load(ply_path, process=False)
            ply_tree = cKDTree(ply_mesh.vertices)
            ply_colors = ply_mesh.visual.vertex_colors
        except Exception as e:
            print(f"  ERROR loading PLY {ply_path}: {e}")
            continue
            
        scene_to_export = trimesh.Scene()
        
        for obj_idx in range(len(orig_trans)):
            obj_path = os.path.join(obj_dir, f"object_{obj_idx:03d}.obj")
            if not os.path.exists(obj_path):
                continue
                
            delta_t = np.array(res_trans[obj_idx]) - np.array(orig_trans[obj_idx])
            mesh = load_and_translate_obj(obj_path, delta_t, ply_tree, ply_colors)
            if mesh is not None:
                scene_to_export.add_geometry(mesh)
                
        if not scene_to_export.is_empty:
            out_glb_path = os.path.join(export_out_dir, f"{scene_id}.glb")
            scene_to_export.export(out_glb_path)
            print(f"  -> Saved: {out_glb_path}")
            exported_count += 1
            
    print(f"\nDone! Exported: {exported_count} colored GLB files.")

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
