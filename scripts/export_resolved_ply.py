import os
import json
import numpy as np
import trimesh
import argparse
import glob

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="Path to merge dir (e.g. merge_d/merged)")
    args = parser.parse_args()
    
    orig_json = os.path.join(args.output_dir, "collision_params.json")
    resolved_json = os.path.join(args.output_dir, "collision_params_resolved.json")
    scene_mesh_dir = os.path.join(args.output_dir, "scene_mesh")
    out_ply_dir = os.path.join(args.output_dir, "scene_mesh_resolved")
    
    os.makedirs(out_ply_dir, exist_ok=True)
    
    with open(orig_json, 'r') as f:
        orig_data = json.load(f)
    with open(resolved_json, 'r') as f:
        res_data = json.load(f)
        
    scene_ids = res_data["scene_ids"]
    
    for scene_idx, scene_id in enumerate(scene_ids):
        print(f"Exporting {scene_id}...")
        
        orig_trans = orig_data["translations"][scene_idx]
        res_trans = res_data["translations"][scene_idx]
        
        # Load the original individual objs
        obj_dir = os.path.join(scene_mesh_dir, scene_id + "_individual_objs")
        if not os.path.exists(obj_dir):
            continue
            
        scene_meshes = []
        
        # We need to map object index to the correct object_*.obj file
        # They are usually named object_000.obj, object_001.obj
        # But wait, there might be floor/ceiling in the json that are skipped in obj exports.
        # Wait, the objects exported in `_individual_objs` map exactly to the indices in the json?
        # Let's check how many objects are in the json vs the folder.
        
        # In our previous render_resolved.py, we did:
        # obj_files = sorted(glob.glob(os.path.join(obj_dir, "object_*.obj")))
        # for i, obj_file in enumerate(obj_files):
        #     mesh = trimesh.load(obj_file, force='mesh')
        #     delta_t = np.array(res_trans[i]) - np.array(orig_trans[i])
        #     mesh.apply_translation(delta_t)
        #     scene_meshes.append(mesh)
        
        # Also include floor/ceiling/walls if they exist (solid_*_wire.png etc, but we only want meshes)
        
        # Actually, let's load ALL .obj files. If it's an object_*.obj, we apply the translation.
        # What if there are other .obj files like room structure? We just load them as is.
        obj_files = sorted(glob.glob(os.path.join(obj_dir, "object_*.obj")))
        for i, obj_file in enumerate(obj_files):
            mesh = trimesh.load(obj_file, force='mesh')
            if i < len(res_trans):
                delta_t = np.array(res_trans[i]) - np.array(orig_trans[i])
                mesh.apply_translation(delta_t)
            scene_meshes.append(mesh)
            
        # Add floor/ceiling if any
        other_objs = glob.glob(os.path.join(obj_dir, "*.obj"))
        for obj_file in other_objs:
            if "object_" not in os.path.basename(obj_file):
                mesh = trimesh.load(obj_file, force='mesh')
                scene_meshes.append(mesh)
                
        if len(scene_meshes) > 0:
            combined = trimesh.util.concatenate(scene_meshes)
            out_file = os.path.join(out_ply_dir, scene_id + ".ply")
            combined.export(out_file)

if __name__ == "__main__":
    main()
