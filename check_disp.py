
import json
import numpy as np
import math

rooms = ['bedroom', 'livingroom', 'diningroom']
for room in rooms:
    with open(f'{room}/collision_params_merged.json') as f:
        d1 = json.load(f)
    with open(f'{room}/collision_params_resolved.json') as f:
        d2 = json.load(f)
    
    all_disp = []
    moved = 0
    total = 0
    
    for i in range(len(d1['scene_ids'])):
        t1 = np.array(d1['translations'][i])
        t2 = np.array(d2['translations'][i])
        diff = t2 - t1
        dist = np.sqrt(diff[:,0]**2 + diff[:,2]**2)
        total += len(dist)
        m = dist[dist > 1e-5]
        moved += len(m)
        all_disp.extend(m.tolist())
        
    avg = np.mean(all_disp) * 100 if all_disp else 0
    print(f'{room.upper()}: Moved {moved}/{total} objects ({(moved/total)*100:.1f}%), Avg Displacement = {avg:.2f} cm')

