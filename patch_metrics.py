import os
import sys

def patch_file():
    target = 'compute_collision_metrics.py'
    with open(target, 'r', encoding='utf-8') as f:
        content = f.read()

    bounds_cache_code = """
CATEGORY_TO_PROCESSED = {
    "bedroom":    "bedrooms_objfeats_32_64",
    "livingroom": "livingrooms_objfeats_32_64",
    "diningroom": "diningrooms_objfeats_32_64",
}
def _extract_base_id(scene_id):
    import re
    m = re.match(r'^(.+?-\d+)_\d+_\d+$', scene_id)
    if m: return m.group(1)
    return scene_id

def _build_room_bounds_cache(category, base_dir):
    processed_subdir = CATEGORY_TO_PROCESSED.get(category)
    if processed_subdir is None: return {}
    cat_dir = os.path.join(base_dir, "3d_front_processed", processed_subdir)
    cache = {}
    if not os.path.isdir(cat_dir): return cache
    import numpy as np
    for d in os.listdir(cat_dir):
        boxes_path = os.path.join(cat_dir, d, "boxes.npz")
        if not os.path.isfile(boxes_path): continue
        parts = d.split("_", 1)
        base_id = parts[1] if len(parts) == 2 else d
        if base_id in cache: continue
        try:
            data = np.load(boxes_path, allow_pickle=True)
            if "floor_plan_vertices" not in data: continue
            vertices = data["floor_plan_vertices"]
            xs = vertices[:, 0]
            zs = vertices[:, 2]
            cxL = 0.0
            czL = 0.0
            hxL = float((xs.max() - xs.min()) / 2.0)
            hzL = float((zs.max() - zs.min()) / 2.0)
            axL = np.array([1.0, 0.0])
            azL = np.array([0.0, 1.0])
            cache[base_id] = (cxL, czL, hxL, hzL, axL, azL)
        except Exception: pass
    return cache
"""

    # 1. Insert bounds cache
    content = content.replace('import torch', 'import torch\n' + bounds_cache_code)

    # 2. Modify resolve_with_logging
    content = content.replace(
        'def resolve_with_logging(sizes_list, trans_list, angles_rad_list, class_labels_list,',
        'def resolve_with_logging(sizes_list, trans_list, angles_rad_list, class_labels_list, room_bounds=None,'
    )

    # 3. Modify inner room bounds assignment
    content = content.replace(
        '        if main_layout_idx != -1:',
        '        if room_bounds is not None:\n            pass\n        elif main_layout_idx != -1:'
    )

    # 4. Modify compute_convergence_metrics signature
    content = content.replace(
        'def compute_convergence_metrics(data_before, max_iter=500, push_eps=0.02):',
        'def compute_convergence_metrics(data_before, bounds_cache=None, max_iter=500, push_eps=0.02):'
    )

    # 5. Modify compute_convergence_metrics loop
    content = content.replace(
        '        cl = data_before["class_labels"][idx] if "class_labels" in data_before else None',
        '        cl = data_before["class_labels"][idx] if "class_labels" in data_before else None\n        rb = bounds_cache.get(_extract_base_id(data_before["scene_ids"][idx])) if bounds_cache else None'
    )

    # 6. Pass room bounds to resolve_with_logging
    content = content.replace(
        '            cl,',
        '            cl,\n            room_bounds=rb,'
    )

    # 7. Modify process_room signature
    content = content.replace(
        'def process_room(room_name, room_dir, max_iter=500, push_eps=0.02):',
        'def process_room(room_name, room_dir, base_dir, max_iter=500, push_eps=0.02):'
    )

    # 8. Add cache build to process_room
    content = content.replace(
        '    print(f"  Re-running solver with logging...")',
        '    print(f"  Re-running solver with logging...")\n    bounds_cache = _build_room_bounds_cache(room_name, base_dir)'
    )

    # 9. Pass bounds_cache to compute_convergence_metrics
    content = content.replace(
        'compute_convergence_metrics(data_before, max_iter=max_iter, push_eps=push_eps)',
        'compute_convergence_metrics(data_before, bounds_cache=bounds_cache, max_iter=max_iter, push_eps=push_eps)'
    )

    # 10. Update process_room call in main
    content = content.replace(
        '            room_metrics = process_room(room, room_dir, max_iter=args.max_iter, push_eps=args.push_eps)',
        '            room_metrics = process_room(room, room_dir, args.base_dir, max_iter=args.max_iter, push_eps=args.push_eps)'
    )
    
    # 11. Read actual resolved translations instead of fake re-run
    # Find the displacement section
    disp_section = """
    # ── Build resolved data dict for after-metrics ────────────────────────
    data_after = dict(data_before)
    data_after["translations"] = resolved_trans_all"""
    
    new_disp_section = """
    # ── Load actual resolved translations ────────────────────────
    resolved_path = os.path.join(room_dir, "collision_params_resolved.json")
    if os.path.exists(resolved_path):
        with open(resolved_path, "r", encoding="utf-8") as f:
            data_after = json.load(f)
        resolved_trans_all = data_after["translations"]
    else:
        data_after = dict(data_before)
        data_after["translations"] = resolved_trans_all
    """
    content = content.replace(disp_section, new_disp_section)

    with open(target, 'w', encoding='utf-8') as f:
        f.write(content)
        
    print("Patch applied successfully.")

if __name__ == '__main__':
    patch_file()
