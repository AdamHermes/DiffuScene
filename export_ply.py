import json
import numpy as np
import trimesh
import os
import glob

def main():
    d1 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params.json'))
    d2 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params_resolved.json'))
    idx = d1['scene_ids'].index('LivingDiningRoom-9790_13_013')
    
    obj_files = sorted(glob.glob('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013_individual_objs/object_*.obj'))
    
    meshes = []
    for i, f in enumerate(obj_files):
        m = trimesh.load(f, force='mesh')
        if i < len(d1['translations'][idx]):
            orig_t = np.array(d1['translations'][idx][i])
            res_t = np.array(d2['translations'][idx][i])
            m.apply_translation(res_t - orig_t)
        meshes.append(m)
        
    combined = trimesh.util.concatenate(meshes)
    out_dir = '/Users/lehoangan/Downloads/merge_d/merged/scene_mesh_resolved'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'LivingDiningRoom-9790_13_013.ply')
    combined.export(out_path)
    print(f"Exported to {out_path}")

if __name__ == '__main__':
    main()
