import argparse
import os
import sys
import json
import numpy as np
import trimesh
import seaborn as sns

def load_and_translate_obj(obj_path, delta_t, color_rgb):
    try:
        # Load mesh simply without process to keep it clean
        mesh = trimesh.load(obj_path, process=False, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) > 0:
                mesh = list(mesh.geometry.values())[0]
            else:
                return None
                
        # color_rgb is typically in [0, 1] range, convert to [0, 255]
        rgba = [int(color_rgb[0]*255), int(color_rgb[1]*255), int(color_rgb[2]*255), 255]
        
        # Apply color as a simple material
        material = trimesh.visual.material.SimpleMaterial(diffuse=rgba)
        mesh.visual = trimesh.visual.TextureVisuals(material=material)
        mesh.visual.face_colors = rgba
        
        # Apply transformation
        if delta_t is not None and not np.allclose(delta_t, 0.0):
            mesh.apply_translation(delta_t)
            
        return mesh
    except Exception as e:
        print(f"  WARNING: Cannot load {obj_path}: {e}")
        return None

def main(argv):
    parser = argparse.ArgumentParser(description="Export resolved scenes as GLB with seaborn HLS colors")
    parser.add_argument("--output_dir", required=True, help="Path to merge folder containing collision_params_resolved.json")
    parser.add_argument("--export_dir", default="resolved_glb_hls", help="Output folder name for glb files")
    
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
        
    scene_ids = res_data["scene_ids"]
    class_labels_all = res_data.get("class_labels", [])
    
    if class_labels_all and len(class_labels_all) > 0 and len(class_labels_all[0]) > 0:
        n_classes = len(class_labels_all[0][0])
        color_palette = np.array(sns.color_palette('hls', max(n_classes - 2, 1)))
    else:
        color_palette = np.array(sns.color_palette('hls', 23))
        
    exported_count = 0
    print(f"Found {len(scene_ids)} scenes. Exporting HLS colored GLB files...")
    
    for i, scene_id in enumerate(scene_ids):
        print(f"[{i+1}/{len(scene_ids)}] Exporting {scene_id}...")
        
        orig_trans = orig_data["translations"][i]
        res_trans = res_data["translations"][i]
        
        obj_dir = os.path.join(scene_mesh_dir, f"{scene_id}_individual_objs")
        
        if not os.path.exists(obj_dir):
            print(f"  WARNING: Missing obj_dir for {scene_id}")
            continue
            
        scene_to_export = trimesh.Scene()
        
        scene_class_labels = class_labels_all[i] if i < len(class_labels_all) else []
        
        for obj_idx in range(len(orig_trans)):
            obj_path = os.path.join(obj_dir, f"object_{obj_idx:03d}.obj")
            if not os.path.exists(obj_path):
                continue
                
            if obj_idx < len(scene_class_labels):
                one_hot = np.array(scene_class_labels[obj_idx])
                class_index = int(one_hot.argmax())
                color_index = min(class_index, len(color_palette) - 1)
                color = color_palette[color_index, :]
            else:
                color = np.array([0.5, 0.5, 0.5])
                
            delta_t = np.array(res_trans[obj_idx]) - np.array(orig_trans[obj_idx])
            mesh = load_and_translate_obj(obj_path, delta_t, color)
            
            if mesh is not None:
                scene_to_export.add_geometry(mesh)
                
        if not scene_to_export.is_empty:
            out_glb_path = os.path.join(export_out_dir, f"{scene_id}.glb")
            scene_to_export.export(out_glb_path)
            print(f"  -> Saved: {out_glb_path}")
            exported_count += 1
            
    print(f"\nDone! Exported: {exported_count} HLS colored GLB files.")

if __name__ == "__main__":
    main(sys.argv[1:])
