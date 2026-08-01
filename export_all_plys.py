import json
import numpy as np
import trimesh
import os
import glob

def export_all():
    in_file = '/Users/lehoangan/Downloads/merge_d/merged/collision_params.json'
    res_file = '/Users/lehoangan/Downloads/merge_d/merged/collision_params_resolved.json'
    d1 = json.load(open(in_file))
    d2 = json.load(open(res_file))
    
    out_dir = '/Users/lehoangan/Downloads/merge_d/merged/scene_mesh_resolved'
    os.makedirs(out_dir, exist_ok=True)
    
    for idx, scene_id in enumerate(d1['scene_ids']):
        # Just export a few to prove it works
        if idx > 2:
            break
        
        obj_dir = f'/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/{scene_id}_individual_objs'
        obj_files = sorted(glob.glob(os.path.join(obj_dir, 'object_*.obj')))
        if not obj_files:
            continue
            
        meshes = []
        for i, f in enumerate(obj_files):
            m = trimesh.load(f, force='mesh')
            if i < len(d1['translations'][idx]):
                orig_t = np.array(d1['translations'][idx][i])
                res_t = np.array(d2['translations'][idx][i])
                m.apply_translation(res_t - orig_t)
            meshes.append(m)
            
        combined = trimesh.util.concatenate(meshes)
        out_path = os.path.join(out_dir, f'{scene_id}.ply')
        combined.export(out_path)
        print(f"Exported {out_path}")

export_all()
