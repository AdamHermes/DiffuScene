import json
import numpy as np

d1 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params.json'))
d2 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params_resolved.json'))
idx = d1['scene_ids'].index('LivingDiningRoom-9790_13_013')

orig_t = np.array(d1['translations'][idx])
res_t = np.array(d2['translations'][idx])
sizes = np.array(d1['sizes'][idx]) * 2.0

for i in [0, 3, 5]:
    print(f"Object {i}:")
    print(f"  Orig t: {orig_t[i]}")
    print(f"  Res t:  {res_t[i]}")
    print(f"  Delta:  {res_t[i] - orig_t[i]}")
    print(f"  Size:   {sizes[i]}")
