import json
import numpy as np

d1 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params.json'))
d2 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params_resolved.json'))
idx = d1['scene_ids'].index('LivingDiningRoom-9790_13_013')

orig_t = np.array(d1['translations'][idx])
res_t = np.array(d2['translations'][idx])
diff = res_t - orig_t

for i in range(len(diff)):
    d = diff[i]
    if np.linalg.norm(d) > 1e-4:
        print(f"Object {i} moved by {d}")
